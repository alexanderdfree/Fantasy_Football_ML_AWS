"""End-to-end smoke test for the TE pipeline.

Runs the full ``src.shared.pipeline.run_pipeline`` with a shrunk config
(2-layer x 8-unit NN, 1 epoch, no attention/LightGBM) on a tiny
slice of real engineered data (50 players x 2 seasons). Asserts:

  - pipeline completes without exception,
  - test predictions exist, have correct shape, and are finite,
  - two independent runs with seed=42 produce bit-identical predictions.

Budget: < 20 seconds.

The pipeline hard-codes ``{POS}/outputs`` for artifact saves; we chdir into
a tmp workspace per run and symlink ``data/`` so schedule parquet reads keep
working without overwriting the checked-in production checkpoints.

Uses real engineered parquets (not synthetic) because ``build_features`` for
TE pulls in upstream merges (weather/Vegas, opponent rankings, depth chart)
that synthetic frames cannot reproduce — drift in any of those merges would
slip past a fully-synthetic test.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.pipeline import run_pipeline
from src.te.config import CONFIG_TINY, POSITION_CONFIG
from tests._skip_helpers import require_splits

TARGETS = POSITION_CONFIG.targets
from src.te.data import filter_to_position
from src.te.features import (
    add_specific_features,
    fill_nans,
    get_feature_columns,
)
from src.te.targets import compute_targets

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"

pytestmark = [
    pytest.mark.e2e,
    # Silence PerformanceWarning from build_features on tiny synthetic input.
    pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning"),
]


def _build_tiny_cfg() -> dict:
    """Bundle CONFIG_TINY with the TE callables required by run_pipeline."""
    return {
        **CONFIG_TINY,
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
    }


def _load_tiny_splits(
    n_players: int = 50,
    train_seasons=(2022, 2023),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice the real engineered parquets to a tiny deterministic subset.

    Uses real pre-engineered data because run_pipeline expects 100+ upstream
    feature columns that would be impractical to synthesize. Deterministic
    because we pick the ``n_players`` with the most games (stable ordering).
    """
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    te_train_all = train[train["position"] == "TE"]

    # Top-n_players by game count — stable because pandas sort is stable and
    # game counts have wide enough spread that ties don't matter at n=50.
    top_players = (
        te_train_all.groupby("player_id").size().sort_values(ascending=False).head(n_players).index
    )
    te_train = te_train_all[
        te_train_all["season"].isin(train_seasons) & te_train_all["player_id"].isin(top_players)
    ].copy()

    val_full = pd.read_parquet(SPLITS_DIR / "val.parquet")
    te_val = val_full[
        (val_full["position"] == "TE") & val_full["player_id"].isin(top_players)
    ].copy()

    test_full = pd.read_parquet(SPLITS_DIR / "test.parquet")
    te_test = test_full[
        (test_full["position"] == "TE") & test_full["player_id"].isin(top_players)
    ].copy()

    return te_train, te_val, te_test


@pytest.fixture(scope="module")
def real_tiny_splits():
    """Load and cache the tiny real-parquet slice once per module.

    Skipped when ``data/splits/*.parquet`` is absent (worktree clones, fresh
    CI checkouts before the data step). Splits are produced by the data-pull
    workflow documented in SETUP.md; tests cannot synthesize them because
    run_pipeline expects 100+ engineered upstream feature columns.
    """
    require_splits(SPLITS_DIR)
    return _load_tiny_splits()


def _run_tiny_pipeline(splits, workdir, seed: int = 42):
    """Run the TE pipeline on the supplied splits inside ``workdir``.

    chdir isolates ``TE/outputs`` writes; symlinked ``data/`` lets the
    pipeline read schedule parquets for weather features.
    """
    train, val, test = splits
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        data_link = workdir / "data"
        if not data_link.exists():
            data_link.symlink_to(Path(cwd) / "data", target_is_directory=True)
        # Pandas fragmentation warnings are signal noise from build_features on
        # tiny inputs — suppress only inside this helper to keep pytest output
        # readable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_pipeline(
                "TE",
                _build_tiny_cfg(),
                train.copy(),
                val.copy(),
                test.copy(),
                seed=seed,
            )
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Module-scoped pipeline runs — one shared run for shape/finite assertions,
# a second cached run for the cross-run bit-identity check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(real_tiny_splits, tmp_path_factory):
    """Single pipeline invocation shared across tests (saves ~6s per test)."""
    workdir = tmp_path_factory.mktemp("te_e2e_run1")
    return _run_tiny_pipeline(real_tiny_splits, workdir, seed=42)


@pytest.fixture(scope="module")
def pipeline_run_repeat(real_tiny_splits, pipeline_run, tmp_path_factory):
    """Second pipeline invocation with the same seed for bit-identity checks.

    Uses the same real-parquet slice as pipeline_run — the reproducibility
    contract is that two runs with the same seed and inputs must agree.
    """
    workdir = tmp_path_factory.mktemp("te_e2e_run2")
    return _run_tiny_pipeline(real_tiny_splits, workdir, seed=42)


class TestPipelineE2E:
    def test_pipeline_completes_and_produces_finite_predictions(
        self, real_tiny_splits, pipeline_run
    ):
        """Smoke: pipeline runs cleanly, predictions finite and correctly shaped."""
        _, _, test = real_tiny_splits
        result = pipeline_run

        # Structural assertions.
        assert "test_df" in result
        assert "ridge_metrics" in result
        assert "nn_metrics" in result
        test_df = result["test_df"]

        # Prediction columns exist and match test row count.
        for pred_col in ("pred_ridge_total", "pred_nn_total", "pred_baseline"):
            assert pred_col in test_df.columns, f"missing {pred_col}"
            arr = test_df[pred_col].values
            assert arr.shape == (len(test),), f"{pred_col} shape {arr.shape} != ({len(test)},)"
            assert np.isfinite(arr).all(), f"{pred_col} has NaN/Inf"

        # Per-target predictions also present and finite.
        for target in TARGETS:
            for model in ("ridge", "nn"):
                col = f"pred_{model}_{target}"
                assert col in test_df.columns
                assert np.isfinite(test_df[col].values).all()

    def test_pipeline_is_bit_identical_across_runs(self, pipeline_run, pipeline_run_repeat):
        """Two runs with seed=42 yield bit-identical predictions.

        Guards end-to-end reproducibility: a single nondeterministic op
        anywhere in the pipeline (Ridge, NN init, data loader shuffle)
        would break this.
        """
        test_a = pipeline_run["test_df"]
        test_b = pipeline_run_repeat["test_df"]

        # Row count and order must match.
        assert len(test_a) == len(test_b)

        # Every prediction column must be bit-identical (atol=0).
        for col in test_a.columns:
            if not col.startswith("pred_"):
                continue
            a = test_a[col].values
            b = test_b[col].values
            np.testing.assert_array_equal(
                a,
                b,
                err_msg=f"{col} differs across seeded runs",
            )
