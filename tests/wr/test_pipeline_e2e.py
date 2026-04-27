"""End-to-end pipeline smoke test for the WR position.

Runs the full ``src.shared.pipeline.run_pipeline`` with a shrunk config
(2-layer × 8-unit NN, 1 epoch, no attention/LightGBM) on a tiny
slice of real data (50 players × 2 seasons). Asserts:

  * pipeline completes without exception
  * predictions are finite with correct shape
  * bit-identical predictions across two runs with the same seed

Budget: < 20s.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_pipeline
from src.wr.config import CONFIG_TINY, TARGETS
from src.wr.data import filter_to_position
from src.wr.features import add_specific_features, fill_nans, get_feature_columns
from src.wr.targets import compute_targets

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
_ALL_TARGETS = tuple(TARGETS)


def _build_tiny_cfg() -> dict:
    """Assemble the tiny config with position-specific callables attached."""
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
    wr_train_all = train[train["position"] == "WR"]

    # Top-n_players by game count — stable because pandas sort is stable and
    # game counts have wide enough spread that ties don't matter at n=50.
    top_players = (
        wr_train_all.groupby("player_id").size().sort_values(ascending=False).head(n_players).index
    )
    wr_train = wr_train_all[
        wr_train_all["season"].isin(train_seasons) & wr_train_all["player_id"].isin(top_players)
    ].copy()

    val_full = pd.read_parquet(SPLITS_DIR / "val.parquet")
    wr_val = val_full[
        (val_full["position"] == "WR") & val_full["player_id"].isin(top_players)
    ].copy()

    test_full = pd.read_parquet(SPLITS_DIR / "test.parquet")
    wr_test = test_full[
        (test_full["position"] == "WR") & test_full["player_id"].isin(top_players)
    ].copy()

    return wr_train, wr_val, wr_test


@pytest.fixture(scope="module")
def tiny_splits():
    """Load and cache the tiny split once per module.

    Skipped when ``data/splits/*.parquet`` is absent (worktree clones,
    fresh CI checkouts before the data step). Splits are produced by
    the data-pull workflow documented in SETUP.md; tests cannot
    synthesize them because run_pipeline expects 100+ engineered
    upstream feature columns.
    """
    splits_dir = SPLITS_DIR
    if not (splits_dir / "train.parquet").exists():
        pytest.skip(f"engineered splits absent at {splits_dir} (run data pull — see SETUP.md)")
    return _load_tiny_splits()


def _run(tiny_splits, tmp_outputs_dir, seed: int = 42):
    """Run the WR pipeline inside ``tmp_outputs_dir`` and return (predictions).

    The pipeline hard-codes ``WR/outputs`` for artifact saves; we ``chdir`` into
    a tmp workspace and symlink ``data/`` so the pipeline finds splits without
    polluting the checked-in outputs directory.
    """
    cfg = _build_tiny_cfg()
    train_df, val_df, test_df = tiny_splits

    cwd = os.getcwd()
    Path(tmp_outputs_dir).mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(tmp_outputs_dir)
        # Symlink data/ so schedule/roster parquet reads inside run_pipeline work
        data_link = Path(tmp_outputs_dir) / "data"
        if not data_link.exists():
            data_link.symlink_to(Path(cwd) / "data", target_is_directory=True)
        result = run_pipeline("WR", cfg, train_df, val_df, test_df, seed=seed)
    finally:
        os.chdir(cwd)
    return result


# ---------------------------------------------------------------------------
# Module-scoped pipeline runs — one run shared by finite/shape checks,
# a second run cached for the cross-run bit-identity check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(tiny_splits, tmp_path_factory):
    """Single pipeline invocation shared across tests (saves ~6s per test)."""
    workdir = tmp_path_factory.mktemp("wr_e2e_run1")
    return _run(tiny_splits, workdir, seed=42)


@pytest.fixture(scope="module")
def pipeline_run_repeat(tiny_splits, pipeline_run, tmp_path_factory):
    """Second pipeline invocation with the same seed for bit-identity checks."""
    workdir = tmp_path_factory.mktemp("wr_e2e_run2")
    return _run(tiny_splits, workdir, seed=42)


@pytest.mark.e2e
def test_pipeline_completes_and_predictions_finite(pipeline_run):
    """run_pipeline executes end-to-end with finite, correctly-shaped outputs."""
    result = pipeline_run

    assert "per_target_preds" in result
    ridge = result["per_target_preds"]["ridge"]
    nn = result["per_target_preds"]["nn"]

    expected_n = len(result["test_df"])
    for preds, name in [(ridge, "ridge"), (nn, "nn")]:
        for key in _ALL_TARGETS:
            arr = np.asarray(preds[key])
            assert arr.shape == (expected_n,), f"{name}.{key} has shape {arr.shape}"
            assert np.isfinite(arr).all(), f"{name}.{key} has non-finite values"


@pytest.mark.e2e
def test_pipeline_deterministic_across_runs(pipeline_run, pipeline_run_repeat):
    """Two runs with the same seed produce bit-identical predictions."""
    for backbone in ("ridge", "nn"):
        p1 = pipeline_run["per_target_preds"][backbone]
        p2 = pipeline_run_repeat["per_target_preds"][backbone]
        for key in _ALL_TARGETS:
            np.testing.assert_array_equal(
                np.asarray(p1[key]),
                np.asarray(p2[key]),
                err_msg=f"{backbone}.{key} differed across identical-seed runs",
            )
