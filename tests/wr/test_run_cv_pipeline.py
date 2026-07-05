"""End-to-end smoke + compatibility test for ``src.shared.pipeline.run_cv_pipeline`` (WR).

``run_cv_pipeline`` is the expanding-window CV orchestrator wrapped by
``src/wr/run_pipeline.py::run_cv``. The non-CV E2E test
(``tests/wr/test_pipeline_e2e.py``) covers ``run_pipeline`` only — this file
fills that gap. Mirrors ``tests/qb/test_run_cv_pipeline.py``: drive
``run_cv_pipeline`` end-to-end, then assert the result dict carries the CV +
holdout-evaluation metrics consumers (``benchmark.py``,
``summarize_pipeline_result``) read.

The same fixture also serves as a regression guard against drift between
``run_pipeline`` and ``run_cv_pipeline`` — they share most of the inner
machinery (``_prepare_train_val``, ``_train_nn``, ``RidgeMultiTarget``) so
config-key renames or new mandatory cfg entries surface here first.

Unlike the QB CV test (which synthesizes data), WR slices the real
engineered parquets — the WR pipeline expects 100+ upstream feature
columns that would be impractical to synthesize. This mirrors WR's
existing E2E pattern in ``tests/wr/test_pipeline_e2e.py``.

Budget: < 180s on CPU (4 CV folds + final holdout NN training on a tiny
real-data slice). Per-test timeout widened to 180s for the same xdist-
threaded-timeout reason as the QB CV class (see its docstring).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_cv_pipeline
from src.wr.config import CONFIG_TINY, POSITION_CONFIG
from src.wr.data import filter_to_position
from src.wr.features import add_specific_features, fill_nans, get_feature_columns
from src.wr.targets import compute_targets
from tests._skip_helpers import require_splits

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


def _build_tiny_cfg() -> dict:
    """Assemble the tiny config with position-specific callables attached.

    Mirrors ``tests/wr/test_pipeline_e2e.py::_build_tiny_cfg`` so the CV
    path exercises the same shrunk hyperparameters the non-CV E2E uses
    (1-epoch NN, no attention/LGBM, 2-fold ridge CV).
    """
    cfg = dict(CONFIG_TINY)
    cfg.update(
        {
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "aggregate_fn": aggregate_fn_for("WR"),
        }
    )
    return cfg


def _load_cv_splits(n_players: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the engineered parquets to (full_df, test_df) for CV.

    ``expanding_window_folds`` defaults to ``CV_VAL_SEASONS = [2021, 2022,
    2023, 2024]`` so each fold trains on prior seasons (>= 2013) and
    validates on a single later season. ``full_df`` must therefore span
    2013-2024 (train.parquet covers 2013-2023; val.parquet covers 2024);
    ``test_df`` is the 2025 holdout. ``n_players`` defaults to 25 to keep
    the 4-fold + final-holdout NN train budget under 180s on CPU.
    """
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    wr_train_all = train[train["position"] == "WR"]

    # Top-n_players by game count — stable ordering for reproducible runs.
    top_players = (
        wr_train_all.groupby("player_id").size().sort_values(ascending=False).head(n_players).index
    )

    wr_train = wr_train_all[wr_train_all["player_id"].isin(top_players)].copy()

    val_full = pd.read_parquet(SPLITS_DIR / "val.parquet")
    wr_val = val_full[
        (val_full["position"] == "WR") & val_full["player_id"].isin(top_players)
    ].copy()

    test_full = pd.read_parquet(SPLITS_DIR / "test.parquet")
    wr_test = test_full[
        (test_full["position"] == "WR") & test_full["player_id"].isin(top_players)
    ].copy()

    full_df = pd.concat([wr_train, wr_val], ignore_index=True)
    return full_df, wr_test


@pytest.fixture(scope="module")
def cv_splits():
    """Load and cache the CV split once per module.

    Skipped when ``data/splits/*.parquet`` is absent (same gate as the
    non-CV E2E fixture).
    """
    require_splits(SPLITS_DIR)
    return _load_cv_splits()


@pytest.fixture(scope="module")
def cv_pipeline_run(cv_splits, tmp_path_factory):
    """Single CV invocation; module-scoped so all assertions reuse it."""
    full_df, test_df = cv_splits
    workdir = tmp_path_factory.mktemp("wr_cv_run")
    cfg = _build_tiny_cfg()

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        # Symlink data/ so weather_features._load_schedules and friends
        # can resolve ``data/raw/schedules_2012_2025.parquet`` relative to
        # the new cwd.
        data_link = workdir / "data"
        if not data_link.exists():
            data_link.symlink_to(Path(cwd) / "data", target_is_directory=True)

        np.random.seed(42)
        torch.manual_seed(42)
        t0 = time.time()
        result = run_cv_pipeline("WR", cfg, full_df.copy(), test_df.copy(), seed=42)
        result["_elapsed"] = time.time() - t0
        return result
    finally:
        os.chdir(cwd)


@pytest.mark.e2e
@pytest.mark.timeout(180)
class TestWRRunCVPipeline:
    """Per-test timeout bumped to 180s (vs the 60s global default) for the
    same xdist-threaded-timeout reason documented in
    ``tests/qb/test_run_cv_pipeline.py::TestQBRunCVPipeline``.
    """

    def test_completes_within_budget(self, cv_pipeline_run):
        """CV smoke must finish under the 180s class timeout on slow CI."""
        assert cv_pipeline_run["_elapsed"] < 180.0, (
            f"run_cv_pipeline took {cv_pipeline_run['_elapsed']:.1f}s (budget 180s)"
        )

    def test_result_has_cv_metrics(self, cv_pipeline_run):
        """``cv_metrics`` must report mean+std MAE/R^2 for ridge + nn across folds."""
        cv = cv_pipeline_run["cv_metrics"]
        assert "ridge" in cv
        assert "nn" in cv
        for model_key in ("ridge", "nn"):
            assert "total" in cv[model_key]
            for stat in ("mae_mean", "mae_std", "r2_mean", "r2_std"):
                assert stat in cv[model_key]["total"]
                assert np.isfinite(cv[model_key]["total"][stat])

    def test_per_fold_arrays_match_fold_count(self, cv_pipeline_run):
        """``mae_per_fold`` and ``r2_per_fold`` must have one entry per CV fold.

        With CV_VAL_SEASONS = [2021, 2022, 2023, 2024] we expect 4 folds.
        """
        cv = cv_pipeline_run["cv_metrics"]
        for model_key in ("ridge", "nn"):
            per_fold_mae = cv[model_key]["total"]["mae_per_fold"]
            per_fold_r2 = cv[model_key]["total"]["r2_per_fold"]
            assert len(per_fold_mae) == len(per_fold_r2)
            assert len(per_fold_mae) == 4

    def test_holdout_metrics_present_and_finite(self, cv_pipeline_run):
        """Final-holdout (2025) ridge + nn metrics must include 'total' MAE/R^2."""
        for key in ("ridge_metrics", "nn_metrics"):
            assert key in cv_pipeline_run
            total = cv_pipeline_run[key]["total"]
            assert np.isfinite(total["mae"])
            assert np.isfinite(total["r2"])

    def test_ranking_metrics_populated(self, cv_pipeline_run):
        """Ridge + NN ``season_avg_hit_rate`` must land in [0, 1]."""
        for key in ("ridge_ranking", "nn_ranking"):
            ranking = cv_pipeline_run[key]
            hit_rate = ranking["season_avg_hit_rate"]
            assert 0.0 <= hit_rate <= 1.0

    def test_best_cv_alphas_round_trip(self, cv_pipeline_run):
        """``best_cv_alphas`` must carry one alpha per non-special CV target."""
        best = cv_pipeline_run["best_cv_alphas"]
        # The tiny config doesn't declare two_stage / classification targets,
        # so every TARGETS entry should land in best_cv_alphas with positive
        # float alphas.
        for target in POSITION_CONFIG.targets:
            assert target in best
            assert best[target] > 0

    def test_artifacts_written(self, cv_pipeline_run):
        """``run_cv_pipeline`` writes NN weights, scaler, ridge models, and
        feature-importance figure under ``WR/outputs/`` relative to cwd.

        Sentinel-style check: ``ridge_metrics`` / ``history`` / ``sim_results``
        in the result dict imply the save block ran without raising. Asserting
        exact paths isn't worth the bookkeeping given we chdir'd back.
        """
        assert "history" in cv_pipeline_run
        assert "sim_results" in cv_pipeline_run

    def test_history_contains_train_curves(self, cv_pipeline_run):
        """``history`` from the final NN training must include train + val loss
        traces for ``plot_training_curves`` to render."""
        history = cv_pipeline_run["history"]
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) >= 1


@pytest.mark.unit
def test_run_wr_cv_pipeline_wrapper_dispatches_to_run_cv_pipeline(monkeypatch):
    """``WR.run.run_cv`` is the wrapper called by ``--cv``. Verify it forwards to
    ``run_cv_pipeline`` with the WR position + config — without paying for
    the real CV training.
    """
    import src.wr.run_pipeline as wr_pipe

    seen: list[dict] = []

    def _fake_cv(position, cfg, *args, **kwargs):
        seen.append({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"cv_metrics": {"ridge": {}, "nn": {}}}

    monkeypatch.setattr(wr_pipe, "run_cv_pipeline", _fake_cv)
    wr_pipe.run_cv(full_df="full", test_df="test", seed=7)
    assert len(seen) == 1
    assert seen[0]["position"] == "WR"
    assert seen[0]["cfg"] is wr_pipe.CONFIG
    # full_df, test_df, seed travel as positional args.
    assert seen[0]["args"][-1] == 7

    # Custom cfg overrides CONFIG via the ``or`` short-circuit.
    custom = {"custom": True, "targets": ["x"]}
    wr_pipe.run_cv(full_df=None, test_df=None, seed=11, config=custom)
    assert seen[1]["cfg"] == custom
