"""End-to-end smoke + compatibility test for ``src.shared.pipeline.run_cv_pipeline`` (RB).

``run_cv_pipeline`` is the expanding-window CV orchestrator wrapped by
``src/rb/run_pipeline.py::run_cv``. The non-CV E2E test
(``tests/rb/test_pipeline_e2e.py``) covers ``run_pipeline`` only — this file
fills that gap so RB matches the other five positions (QB/WR/TE/K/DST all
already ship a ``test_run_cv_pipeline.py``). Mirrors
``tests/wr/test_run_cv_pipeline.py``: drive ``run_cv_pipeline`` end-to-end,
then assert the result dict carries the CV + holdout-evaluation metrics
consumers (``benchmark.py``, ``summarize_pipeline_result``) read.

The same fixture also serves as a regression guard against drift between
``run_pipeline`` and ``run_cv_pipeline`` — they share most of the inner
machinery (``_prepare_train_val``, ``_train_nn``, ``RidgeMultiTarget``) so
config-key renames or new mandatory cfg entries surface here first.

Unlike the QB CV test (which synthesizes data), RB slices the real
engineered parquets — the RB pipeline expects 100+ upstream feature columns
that would be impractical to synthesize, and RB's gated-ordinal TD config
flows from ``build_pipeline_config`` (so the config is assembled via the
shared ``build_tiny_config`` helper, not a hand-rolled dict). This mirrors
RB's existing real-data pattern in ``tests/_pipeline_e2e_utils.py``.

Budget: < 180s on CPU (4 CV folds + final holdout NN training on a tiny
real-data slice). Per-test timeout widened to 180s for the same xdist-
threaded-timeout reason as the QB/WR CV classes (see their docstrings).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.rb.config import POSITION_CONFIG
from src.shared.pipeline import run_cv_pipeline
from tests._pipeline_e2e_utils import build_tiny_config
from tests._skip_helpers import require_splits

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


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
    rb_train_all = train[train["position"] == "RB"]

    # Top-n_players by game count — stable ordering for reproducible runs.
    top_players = (
        rb_train_all.groupby("player_id")
        .size()
        .sort_values(ascending=False, kind="mergesort")
        .head(n_players)
        .index
    )

    rb_train = rb_train_all[rb_train_all["player_id"].isin(top_players)].copy()

    val_full = pd.read_parquet(SPLITS_DIR / "val.parquet")
    rb_val = val_full[
        (val_full["position"] == "RB") & val_full["player_id"].isin(top_players)
    ].copy()

    test_full = pd.read_parquet(SPLITS_DIR / "test.parquet")
    rb_test = test_full[
        (test_full["position"] == "RB") & test_full["player_id"].isin(top_players)
    ].copy()

    full_df = pd.concat([rb_train, rb_val], ignore_index=True)
    return full_df, rb_test


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
    workdir = tmp_path_factory.mktemp("rb_cv_run")
    # ``build_tiny_config`` routes RB's POSITION_CONFIG (incl. the gated-ordinal
    # TD targets) through ``build_pipeline_config`` then shrinks the heavy
    # NN/ridge knobs and disables attention/LGBM, matching RB's run_pipeline E2E.
    cfg = build_tiny_config("RB")

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        # Symlink data/ so weather_features._load_schedules and friends can
        # resolve ``data/raw/schedules_2012_2025.parquet`` relative to cwd.
        data_link = workdir / "data"
        if not data_link.exists():
            data_link.symlink_to(Path(cwd) / "data", target_is_directory=True)

        np.random.seed(42)
        torch.manual_seed(42)
        t0 = time.time()
        result = run_cv_pipeline("RB", cfg, full_df.copy(), test_df.copy(), seed=42)
        result["_elapsed"] = time.time() - t0
        return result
    finally:
        os.chdir(cwd)


@pytest.mark.e2e
@pytest.mark.timeout(180)
class TestRBRunCVPipeline:
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
        """``best_cv_alphas`` must carry one alpha per non-special CV target.

        RB declares ``classification_targets`` / ``two_stage_targets`` (the
        gated-ordinal TD heads); ``run_cv_pipeline`` excludes those from Ridge
        alpha tuning, so only the remaining targets should appear in
        ``best_cv_alphas``.
        """
        best = cv_pipeline_run["best_cv_alphas"]
        cfg = build_tiny_config("RB")
        special = set(cfg.get("two_stage_targets", {})) | set(cfg.get("classification_targets", {}))
        cv_ridge_targets = [t for t in POSITION_CONFIG.targets if t not in special]
        assert cv_ridge_targets, "expected at least one non-special Ridge target"
        for target in cv_ridge_targets:
            assert target in best
            assert best[target] > 0

    def test_artifacts_written(self, cv_pipeline_run):
        """Sentinel-style check: ``history`` / ``sim_results`` in the result dict
        imply the final-holdout train + save block ran without raising."""
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
def test_run_rb_cv_pipeline_wrapper_dispatches_to_run_cv_pipeline(monkeypatch):
    """``RB.run.run_cv`` is the wrapper called by ``--cv``. Verify it forwards
    to ``run_cv_pipeline`` with the RB position + config — without paying for
    the real CV training.
    """
    import src.rb.run_pipeline as rb_pipe

    seen: list[dict] = []

    def _fake_cv(position, cfg, *args, **kwargs):
        seen.append({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"cv_metrics": {"ridge": {}, "nn": {}}}

    monkeypatch.setattr(rb_pipe, "run_cv_pipeline", _fake_cv)
    rb_pipe.run_cv(full_df="full", test_df="test", seed=7)
    assert len(seen) == 1
    assert seen[0]["position"] == "RB"
    assert seen[0]["cfg"] is rb_pipe.CONFIG
    # full_df, test_df, seed travel as positional args.
    assert seen[0]["args"][-1] == 7

    # Custom cfg overrides CONFIG via the ``or`` short-circuit.
    custom = {"custom": True, "targets": ["x"]}
    rb_pipe.run_cv(full_df=None, test_df=None, seed=11, config=custom)
    assert seen[1]["cfg"] == custom
