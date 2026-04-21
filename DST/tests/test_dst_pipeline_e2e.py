"""End-to-end smoke test for the DST pipeline.

Exercises ``shared.pipeline.run_pipeline`` all the way through on a tiny
synthetic dataset — the only test in the DST suite that runs the full
orchestration (feature prep -> Ridge tuning -> NN training -> ranking /
backtest / artifact save).  This catches integration bugs (shape
mismatches, config drift, missing columns) that position-unit tests miss.

Design choices
--------------
* **Dataset** — 32 teams x 4 seasons x 17 weeks via the ``tiny_dst_dataset``
  fixture (conftest.py).  Seasons 2022-2024 train/val, 2025 test.  Every
  team plays 17 games per season so the ``MIN_GAMES_PER_SEASON=6`` filter
  does not wipe rows.
* **Config** — ``DST_CONFIG_TINY`` in dst_config.py: 2-layer 8-unit NN,
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
the dataset is too large — revisit DST_CONFIG_TINY before raising the
timeout.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from DST.dst_config import (
    DST_ATTN_HISTORY_STATS,
    DST_CONFIG_TINY,
    DST_HUBER_DELTAS,
    DST_LOSS_WEIGHTS,
    DST_RIDGE_ALPHA_GRIDS,
    DST_SPECIFIC_FEATURES,
    DST_TARGETS,
)
from DST.dst_data import filter_to_dst
from DST.dst_features import (
    add_dst_specific_features,
    compute_dst_features,
    fill_dst_nans,
    get_dst_feature_columns,
)
from DST.dst_targets import compute_dst_targets
from DST.tests.conftest import _build_tiny_dst_dataset
from shared.aggregate_targets import aggregate_fn_for
from shared.pipeline import run_pipeline


def _build_synthetic_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal schedules DataFrame the pipeline can merge against.

    ``shared.weather_features._load_schedules`` expects a parquet on disk
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
    """Build the full DST cfg dict using ``DST_CONFIG_TINY`` for training knobs.

    Non-training fields (targets, callables, alpha grids, loss/Huber)
    match the production DST config so the test covers the real dispatch
    paths in ``shared.pipeline.run_pipeline``.
    """
    cfg = {
        "targets": DST_TARGETS,
        "ridge_alpha_grids": DST_RIDGE_ALPHA_GRIDS,
        "specific_features": DST_SPECIFIC_FEATURES,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": None,
        "loss_weights": DST_LOSS_WEIGHTS,
        "huber_deltas": DST_HUBER_DELTAS,
    }
    cfg.update(DST_CONFIG_TINY)
    return cfg


def _build_tiny_splits(seed: int = 42):
    """Build tiny synthetic DST splits ready for run_pipeline.

    DST's production pipeline computes features on the full dataset
    before splitting (compute_dst_features); we mirror that here so
    rolling windows see the full season history.
    """
    df = _build_tiny_dst_dataset(seed=seed)
    df = compute_dst_targets(df)
    compute_dst_features(df)
    # Split: 2022-2023 train, 2024 val, 2025 test (mirrors src.config values
    # but shifted to the tiny dataset range).
    train = df[df["season"].isin([2022, 2023])].copy()
    val = df[df["season"] == 2024].copy()
    test = df[df["season"] == 2025].copy()
    return train, val, test


@pytest.fixture
def tiny_cwd(tmp_path, monkeypatch, tiny_dst_dataset):
    """Redirect CWD so run_pipeline's ``DST/outputs`` writes land in tmp_path.

    Also stubs out the schedule-parquet loader so it returns a synthetic
    frame keyed to the tiny dataset — the real parquet is resolved by a
    relative path (``data/raw/...``) that breaks after chdir.
    """
    monkeypatch.chdir(tmp_path)
    # Pre-create the directory the pipeline expects so any early file
    # operations don't fail before its os.makedirs call runs.
    os.makedirs(tmp_path / "DST" / "outputs" / "models", exist_ok=True)
    os.makedirs(tmp_path / "DST" / "outputs" / "figures", exist_ok=True)

    # Stub the weather_features schedule loader.  merge_schedule_features
    # short-circuits on the ``_schedule_merged`` column, so if a df already
    # has synthetic schedule-derived columns it is left alone.  We still
    # replace _load_schedules so the initial path-open never happens.
    synthetic_sched = _build_synthetic_schedules(tiny_dst_dataset)
    from shared import weather_features as _wf

    monkeypatch.setattr(_wf, "_schedule_cache", synthetic_sched)
    monkeypatch.setattr(_wf, "_load_schedules", lambda: synthetic_sched)
    return tmp_path


@pytest.mark.e2e
@pytest.mark.timeout(60)
class TestDSTPipelineE2E:
    """Full-pipeline smoke + reproducibility tests."""

    def test_pipeline_runs_without_exception(self, tiny_cwd):
        """run_pipeline(DST_CONFIG_TINY) must complete cleanly."""
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_tiny_cfg()

        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        assert result is not None
        assert "ridge_metrics" in result
        assert "nn_metrics" in result
        assert "test_df" in result

    def test_predictions_shape_and_finite(self, tiny_cwd):
        """Predicted totals must match test-row count and contain only finite values."""
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_tiny_cfg()

        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        n_test = len(result["test_df"])
        for model_key in ("ridge", "nn"):
            preds = result["per_target_preds"][model_key]
            assert "total" in preds
            assert preds["total"].shape == (n_test,), (
                f"{model_key} total-pred shape {preds['total'].shape} != ({n_test},)"
            )
            assert np.isfinite(preds["total"]).all(), f"{model_key} total contains NaN/inf"
            for t in DST_TARGETS:
                assert preds[t].shape == (n_test,)
                assert np.isfinite(preds[t]).all(), f"{model_key}.{t} contains NaN/inf"

    def test_same_seed_bit_identical_predictions(self, tiny_cwd):
        """Two runs with seed=42 must produce bit-identical Ridge + NN totals.

        Strongest check for hidden non-determinism in the shared kernel
        (dataloader shuffle, dropout masks, dict-iteration order, ...).
        """
        cfg = _make_dst_tiny_cfg()

        # Fresh splits per run — the dataset builder is deterministic too
        train1, val1, test1 = _build_tiny_splits(seed=42)
        r1 = run_pipeline("DST", cfg, train1, val1, test1, seed=42)

        train2, val2, test2 = _build_tiny_splits(seed=42)
        r2 = run_pipeline("DST", cfg, train2, val2, test2, seed=42)

        # Ridge is deterministic — exact equality expected.
        np.testing.assert_allclose(
            r1["per_target_preds"]["ridge"]["total"],
            r2["per_target_preds"]["ridge"]["total"],
            atol=0.0,
            rtol=0.0,
            err_msg="Ridge predictions drifted across runs with same seed",
        )
        # NN reproducibility — should be bit-identical on CPU with the same
        # seed, but we allow atol=1e-6 in case BLAS thread scheduling
        # introduces last-bit noise on some platforms.
        np.testing.assert_allclose(
            r1["per_target_preds"]["nn"]["total"],
            r2["per_target_preds"]["nn"]["total"],
            atol=1e-6,
            rtol=0.0,
            err_msg="NN predictions drifted >1e-6 across runs with same seed",
        )

    def test_ridge_metrics_structure(self, tiny_cwd):
        """Ridge metrics must include MAE/R2 for every target + total."""
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_tiny_cfg()
        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        ridge_metrics = result["ridge_metrics"]
        for key in list(DST_TARGETS) + ["total"]:
            assert key in ridge_metrics, f"Ridge metrics missing '{key}'"
            for metric in ("mae", "rmse", "r2"):
                assert metric in ridge_metrics[key]
                assert np.isfinite(ridge_metrics[key][metric])

    def test_attention_nn_trains_and_predicts(self, tiny_cwd):
        """Smoke test for the attention path — enables train_attention_nn on
        the tiny config and asserts the attention model produces finite
        per-target + total predictions. Protects against broken wiring in
        ``get_attn_static_columns`` (DST suffix branch), the per-game opp
        columns on the tiny dataset, and the aggregate_fn hook."""
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_tiny_cfg()
        # Flip attention on + provide required attn_* keys (plus aggregate_fn
        # so training supervises on fantasy_points via the tier-aware DST
        # aggregator, matching run_dst_pipeline.py).
        cfg["train_attention_nn"] = True
        cfg["attn_history_stats"] = DST_ATTN_HISTORY_STATS
        cfg["attn_max_seq_len"] = 17
        cfg["attn_d_model"] = 8
        cfg["attn_n_heads"] = 2
        cfg["attn_encoder_hidden_dim"] = 8
        cfg["attn_positional_encoding"] = True
        cfg["attn_gated_fusion"] = False
        cfg["attn_gated_td"] = False
        cfg["attn_dropout"] = 0.0
        cfg["aggregate_fn"] = aggregate_fn_for("DST")

        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        assert "attn_nn_metrics" in result
        attn_preds = result["per_target_preds"]["attn_nn"]
        n_test = len(result["test_df"])
        assert "total" in attn_preds
        assert attn_preds["total"].shape == (n_test,)
        assert np.isfinite(attn_preds["total"]).all(), "Attention NN total contains NaN/inf"
        for t in DST_TARGETS:
            assert attn_preds[t].shape == (n_test,)
            assert np.isfinite(attn_preds[t]).all(), f"Attention NN.{t} contains NaN/inf"
