"""End-to-end smoke test for the DST pipeline.

Exercises ``src.shared.pipeline.run_pipeline`` all the way through on a tiny
synthetic dataset — the only test in the DST suite that runs the full
orchestration (feature prep -> Ridge tuning -> NN training -> ranking /
backtest / artifact save).  This catches integration bugs (shape
mismatches, config drift, missing columns) that position-unit tests miss.

Design choices
--------------
* **Dataset** — 32 teams x 4 seasons x 17 weeks via the ``tiny_dataset``
  fixture (conftest.py).  Seasons 2022-2024 train/val, 2025 test.  Every
  team plays 17 games per season so the ``MIN_GAMES_PER_SEASON=6`` filter
  does not wipe rows.
* **Config** — ``CONFIG_TINY`` in dst_config.py: 2-layer 8-unit NN,
  1 epoch, no LightGBM, no attention.  The rest of the production config
  (targets, ridge grids, loss weights) is kept so the test exercises
  representative code.
* **CWD override** — ``run_pipeline`` writes artifacts to ``DST/outputs``
  (a relative path).  We ``chdir`` into ``tmp_path`` so the real repo
  outputs are untouched.
* **Reproducibility** — two runs with the same seed must produce
  bit-identical test predictions; this is the strongest check for hidden
  non-determinism in the shared kernel.

Budget: < 20 s.  If the test ever exceeds that, the NN is too deep or
the dataset is too large — revisit CONFIG_TINY before raising the
timeout.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.dst.config import (
    ATTN_HISTORY_STATS,
    ATTN_STATIC_FEATURES,
    CONFIG_TINY,
    HUBER_DELTAS,
    LOSS_WEIGHTS,
    POISSON_TARGETS,
    RIDGE_ALPHA_GRIDS,
    SPECIFIC_FEATURES,
    TARGETS,
)
from src.dst.data import filter_to_position
from src.dst.features import (
    add_specific_features,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.dst.targets import compute_targets
from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_pipeline
from tests.dst.conftest import _build_tiny_dataset


def _build_synthetic_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal schedules DataFrame the pipeline can merge against.

    ``src.shared.weather_features._load_schedules`` expects a parquet on disk
    at ``data/raw/schedules_2012_2025.parquet``.  The E2E test chdirs
    to tmp_path (so artifact writes don't pollute the repo), which
    breaks that relative path.  We monkeypatch ``_load_schedules`` to
    return this synthetic frame instead — it has to carry the columns
    that ``_build_team_schedule_lookup`` reads (see weather_features.py).
    """
    # One row per (season, week, home_team, away_team) combo in the tiny
    # dataset.  Derive home/away from the 'is_home' flag we stamped in.
    home = df[df["is_home"] == 1][
        [
            "season",
            "week",
            "team",
            "opponent_team",
            "spread_line",
            "total_line",
            "rest_days",
            "div_game",
            "is_dome",
        ]
    ].rename(columns={"team": "home_team", "opponent_team": "away_team", "rest_days": "home_rest"})
    away = df[df["is_home"] == 0][["season", "week", "team", "rest_days"]].rename(
        columns={"team": "away_team", "rest_days": "away_rest"}
    )
    sched = home.merge(away, on=["season", "week", "away_team"], how="left")
    sched["away_rest"] = sched["away_rest"].fillna(7).astype(int)
    sched["game_type"] = "REG"
    sched["roof"] = "outdoors"  # we leave is_dome flag inside the DST df
    sched["surface"] = "grass"
    sched["temp"] = 65.0
    sched["wind"] = 5.0
    # drop_duplicates because the source DST frame has both home+away rows
    sched = sched.drop_duplicates(subset=["season", "week", "home_team", "away_team"])
    return sched.reset_index(drop=True)


def _make_dst_tiny_cfg() -> dict:
    """Build the full DST cfg dict using ``CONFIG_TINY`` for training knobs.

    Non-training fields (targets, callables, alpha grids, loss/Huber)
    match the production DST config so the test covers the real dispatch
    paths in ``src.shared.pipeline.run_pipeline``.
    """
    cfg = {
        "targets": TARGETS,
        "ridge_alpha_grids": RIDGE_ALPHA_GRIDS,
        "specific_features": SPECIFIC_FEATURES,
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
        "compute_adjustment_fn": None,
        "loss_weights": LOSS_WEIGHTS,
        "huber_deltas": HUBER_DELTAS,
        "poisson_targets": POISSON_TARGETS,
    }
    cfg.update(CONFIG_TINY)
    return cfg


def _build_tiny_splits(seed: int = 42):
    """Build tiny synthetic DST splits ready for run_pipeline.

    DST's production pipeline computes features on the full dataset
    before splitting (compute_features); we mirror that here so
    rolling windows see the full season history.
    """
    df = _build_tiny_dataset(seed=seed)
    df = compute_targets(df)
    compute_features(df)
    # Split: 2022-2023 train, 2024 val, 2025 test (mirrors src.config values
    # but shifted to the tiny dataset range).
    train = df[df["season"].isin([2022, 2023])].copy()
    val = df[df["season"] == 2024].copy()
    test = df[df["season"] == 2025].copy()
    return train, val, test


@pytest.fixture(scope="module")
def tiny_cwd(tmp_path_factory, tiny_dataset):
    """Redirect CWD so run_pipeline's ``DST/outputs`` writes land in tmp_path.

    Also stubs out the schedule-parquet loader so it returns a synthetic
    frame keyed to the tiny dataset — the real parquet is resolved by a
    relative path (``data/raw/...``) that breaks after chdir.

    Module-scoped so the two cached pipeline runs share one workspace —
    ``pytest.MonkeyPatch()`` is used directly because the built-in
    ``monkeypatch`` fixture is function-scoped.
    """
    mp = pytest.MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("dst_e2e")
    mp.chdir(tmp_path)
    # Pre-create the directory the pipeline expects so any early file
    # operations don't fail before its os.makedirs call runs.
    os.makedirs(tmp_path / "DST" / "outputs" / "models", exist_ok=True)
    os.makedirs(tmp_path / "DST" / "outputs" / "figures", exist_ok=True)

    # Stub the weather_features schedule loader.  merge_schedule_features
    # short-circuits on the ``_schedule_merged`` column, so if a df already
    # has synthetic schedule-derived columns it is left alone.  We still
    # replace _load_schedules so the initial path-open never happens.
    synthetic_sched = _build_synthetic_schedules(tiny_dataset)
    from src.shared import weather_features as _wf

    mp.setattr(_wf, "_schedule_cache", synthetic_sched)
    mp.setattr(_wf, "_load_schedules", lambda: synthetic_sched)
    try:
        yield tmp_path
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# Module-scoped pipeline runs — one shared run for smoke/shape/metrics
# assertions, a second cached run for the cross-run bit-identity check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(tiny_cwd):
    """Single pipeline invocation shared across tests (saves ~6s per test)."""
    train, val, test = _build_tiny_splits(seed=42)
    cfg = _make_dst_tiny_cfg()
    return run_pipeline("DST", cfg, train, val, test, seed=42)


@pytest.fixture(scope="module")
def pipeline_run_repeat(tiny_cwd, pipeline_run):
    """Second pipeline invocation with the same seed for bit-identity checks.

    Fresh splits are rebuilt to mirror the reproducibility contract:
    deterministic re-builds with the same seed must agree bitwise.
    """
    train, val, test = _build_tiny_splits(seed=42)
    cfg = _make_dst_tiny_cfg()
    return run_pipeline("DST", cfg, train, val, test, seed=42)


@pytest.mark.e2e
@pytest.mark.timeout(60)
class TestDSTPipelineE2E:
    """Full-pipeline smoke + reproducibility tests."""

    def test_pipeline_runs_without_exception(self, pipeline_run):
        """run_pipeline(CONFIG_TINY) must complete cleanly."""
        result = pipeline_run
        assert result is not None
        assert "ridge_metrics" in result
        assert "nn_metrics" in result
        assert "test_df" in result

    def test_predictions_shape_and_finite(self, pipeline_run):
        """Per-target predictions must match test-row count and be finite."""
        result = pipeline_run

        n_test = len(result["test_df"])
        for model_key in ("ridge", "nn"):
            preds = result["per_target_preds"][model_key]
            for t in TARGETS:
                assert preds[t].shape == (n_test,)
                assert np.isfinite(preds[t]).all(), f"{model_key}.{t} contains NaN/inf"

    def test_same_seed_bit_identical_predictions(self, pipeline_run, pipeline_run_repeat):
        """Two runs with seed=42 must produce bit-identical Ridge + NN preds.

        Strongest check for hidden non-determinism in the shared kernel
        (dataloader shuffle, dropout masks, dict-iteration order, ...).
        """
        for t in TARGETS:
            # Ridge is deterministic — exact equality expected.
            np.testing.assert_allclose(
                pipeline_run["per_target_preds"]["ridge"][t],
                pipeline_run_repeat["per_target_preds"]["ridge"][t],
                atol=0.0,
                rtol=0.0,
                err_msg=f"Ridge {t} drifted across runs with same seed",
            )
            # NN reproducibility — should be bit-identical on CPU with the same
            # seed, but we allow atol=1e-6 in case BLAS thread scheduling
            # introduces last-bit noise on some platforms.
            np.testing.assert_allclose(
                pipeline_run["per_target_preds"]["nn"][t],
                pipeline_run_repeat["per_target_preds"]["nn"][t],
                atol=1e-6,
                rtol=0.0,
                err_msg=f"NN {t} drifted >1e-6 across runs with same seed",
            )

    def test_ridge_metrics_structure(self, pipeline_run):
        """Ridge metrics must include MAE/R2 for every target + total."""
        ridge_metrics = pipeline_run["ridge_metrics"]
        for key in list(TARGETS) + ["total"]:
            assert key in ridge_metrics, f"Ridge metrics missing '{key}'"
            for metric in ("mae", "rmse", "r2"):
                assert metric in ridge_metrics[key]
                assert np.isfinite(ridge_metrics[key][metric])

    def test_attention_nn_trains_and_predicts(self, tiny_cwd):
        """Smoke test for the attention path — enables train_attention_nn on
        the tiny config and asserts the attention model produces finite
        per-target predictions. Protects against broken wiring in
        ``get_attn_static_columns`` (per-position whitelist) and the
        per-game opp columns on the tiny dataset."""
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_tiny_cfg()
        # Flip attention on + provide required attn_* keys. aggregate_fn is
        # still set for serving + ranking metric reporting, but the NN itself
        # trains on raw-stat heads only.
        cfg["train_attention_nn"] = True
        cfg["attn_history_stats"] = ATTN_HISTORY_STATS
        cfg["attn_static_features"] = ATTN_STATIC_FEATURES
        cfg["attn_max_seq_len"] = 17
        cfg["attn_d_model"] = 8
        cfg["attn_n_heads"] = 2
        cfg["attn_encoder_hidden_dim"] = 8
        cfg["attn_positional_encoding"] = True
        cfg["attn_gated_fusion"] = False
        cfg["attn_gated"] = False
        cfg["attn_dropout"] = 0.0
        cfg["aggregate_fn"] = aggregate_fn_for("DST")

        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        assert "attn_nn_metrics" in result
        attn_preds = result["per_target_preds"]["attn_nn"]
        n_test = len(result["test_df"])
        for t in TARGETS:
            assert attn_preds[t].shape == (n_test,)
            assert np.isfinite(attn_preds[t]).all(), f"Attention NN.{t} contains NaN/inf"
