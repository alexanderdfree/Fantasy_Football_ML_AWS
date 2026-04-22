"""Full-pipeline smoke test for the QB position.

Runs shared.pipeline.run_pipeline end-to-end on a tiny synthetic dataset
with a shrunk NN (1 layer x 8 units, 1 epoch) and asserts:

  - No exceptions
  - Predictions are finite (no NaN/Inf)
  - Output shapes match the test DataFrame length
  - Bit-identical across two runs with the same seed (Ridge: exact equality;
    NN: exact equality — one-epoch training with torch.manual_seed is
    deterministic on CPU)

Targets <20 s total wall clock. Verified via --durations=10.

This exercises the orchestration layer: filter_to_qb -> compute_qb_targets ->
add_qb_specific_features -> fill_qb_nans -> Ridge CV tuning -> Ridge fit ->
NN training -> weekly backtest. A regression anywhere in that chain breaks
this test first.

Notes:
  - This test must run from the project root because shared.weather_features
    and shared.pipeline load data/raw/schedules_2012_2025.parquet with a
    relative path.
  - It writes model artifacts under QB/outputs/; those files are regenerated
    on the next real pipeline run, so overwriting them is acceptable.
"""

import time

import numpy as np
import pandas as pd
import pytest
import torch

# Real NFL team codes (schedule parquet lookup requires these to match)
TEAMS = ["BUF", "KC", "DAL", "PHI", "SF", "LA", "ATL", "CIN"]


def _generate_qb_season(season, seed, n_players=25, n_weeks=17):
    """Build a synthetic QB-season DataFrame with columns run_pipeline needs."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_players + 1):
        team = TEAMS[pid % len(TEAMS)]
        opp = TEAMS[(pid + 1) % len(TEAMS)]
        for wk in range(1, n_weeks + 1):
            rows.append(
                {
                    "player_id": f"QB{pid:02d}",
                    "position": "QB",
                    "position_group": "QB",
                    "season": season,
                    "week": wk,
                    "recent_team": team,
                    "opponent_team": opp,
                    "completions": int(rng.integers(10, 30)),
                    "attempts": int(rng.integers(20, 45)),
                    "passing_yards": float(rng.integers(150, 400)),
                    "passing_tds": int(rng.integers(0, 4)),
                    "interceptions": int(rng.integers(0, 3)),
                    "sacks": int(rng.integers(0, 5)),
                    "sack_yards": float(rng.integers(0, 30)),
                    "sack_fumbles_lost": 0,
                    "rushing_yards": float(rng.integers(0, 60)),
                    "rushing_tds": int(rng.integers(0, 2)),
                    "rushing_fumbles_lost": 0,
                    "rushing_first_downs": int(rng.integers(0, 3)),
                    "rushing_epa": float(rng.uniform(-3, 5)),
                    "receiving_yards": 0.0,
                    "receiving_tds": 0,
                    "receiving_fumbles_lost": 0,
                    "receptions": 0,
                    "carries": int(rng.integers(0, 8)),
                    "targets": 0,
                    "passing_air_yards": float(rng.integers(100, 350)),
                    "passing_yards_after_catch": float(rng.integers(50, 200)),
                    "passing_first_downs": int(rng.integers(5, 20)),
                    "passing_epa": float(rng.uniform(-10, 20)),
                    "snap_pct": 0.95,
                    "pos_QB": 1,
                    "pos_RB": 0,
                    "pos_WR": 0,
                    "pos_TE": 0,
                }
            )
    df = pd.DataFrame(rows)
    # Fantasy points consistent with QB scoring so compute_qb_targets'
    # decomposition check passes without warnings.
    df["fantasy_points"] = (
        df["passing_yards"] * 0.04
        + df["rushing_yards"] * 0.1
        + df["passing_tds"] * 4
        + df["rushing_tds"] * 6
        + df["interceptions"] * -2
    )
    df["fantasy_points_ppr"] = df["fantasy_points"]
    return df


def _tiny_qb_config():
    """Shrunk copy of QB_CONFIG for E2E smoke.

    Changes from the production config:
      - 1-layer 8-unit NN backbone, 4-unit heads
      - 1 epoch, batch_size=16, patience=1
      - Attention NN and LightGBM disabled (cover those in unit tests)
      - Ridge CV reduced to 2 folds, 0 refine points
    """
    from QB.run_qb_pipeline import QB_CONFIG

    cfg = dict(QB_CONFIG)
    cfg.update(
        {
            "nn_backbone_layers": [8],
            "nn_head_hidden": 4,
            "nn_dropout": 0.0,
            "nn_epochs": 1,
            "nn_batch_size": 16,
            "nn_patience": 1,
            "train_attention_nn": False,
            "train_lightgbm": False,
            "ridge_cv_folds": 2,
            "ridge_refine_points": 0,
            "cosine_t0": 1,
            "cosine_t_mult": 1,
            "cosine_eta_min": 1e-5,
        }
    )
    return cfg


@pytest.fixture(scope="module")
def synthetic_splits():
    """Deterministic synthetic QB splits (25 players x 17 weeks per season).

    Train spans 2012-2022 (11 seasons) to match the real QB data range
    (data/raw/schedules_2012_2025.parquet) and to give the pipeline's
    expanding-window Ridge CV tuning enough unique seasons to build
    non-empty folds. val=2023 and test=2024 remain single-season.
    """
    train = pd.concat(
        [_generate_qb_season(season, seed=100 + (season - 2012)) for season in range(2012, 2023)],
        ignore_index=True,
    )
    val = _generate_qb_season(2023, seed=200)
    test = _generate_qb_season(2024, seed=201)
    return train, val, test


def _run_once(splits, seed=42):
    """Run the QB pipeline once with the given seed and return the result."""
    # Seed Python-level RNGs in case any component reads them before being
    # re-seeded inside run_pipeline.
    np.random.seed(seed)
    torch.manual_seed(seed)

    from shared.pipeline import run_pipeline

    train, val, test = splits
    cfg = _tiny_qb_config()
    # Pass defensive copies so the pipeline can't mutate the fixture across runs.
    return run_pipeline("QB", cfg, train.copy(), val.copy(), test.copy(), seed=seed)


@pytest.fixture(scope="module")
def pipeline_run(synthetic_splits):
    """Single pipeline invocation shared across tests (saves ~6s per test)."""
    t0 = time.time()
    result = _run_once(synthetic_splits, seed=42)
    result["_elapsed"] = time.time() - t0
    return result


@pytest.fixture(scope="module")
def pipeline_run_repeat(synthetic_splits, pipeline_run):
    """Second pipeline invocation with the same seed for bit-identity tests.

    Depends on pipeline_run so both share the synthetic_splits fixture.
    """
    return _run_once(synthetic_splits, seed=42)


@pytest.mark.e2e
class TestQBPipelineE2E:
    def test_pipeline_runs_without_exception(self, pipeline_run):
        """Smoke: run_pipeline must complete end-to-end on tiny synthetic data."""
        assert pipeline_run["_elapsed"] < 30.0, (
            f"E2E took {pipeline_run['_elapsed']:.1f}s (budget: 30s)"
        )
        for key in ("ridge_metrics", "nn_metrics", "per_target_preds", "sim_results", "history"):
            assert key in pipeline_run, f"Missing result key: {key}"

    def test_predictions_finite(self, pipeline_run):
        """Ridge and NN test predictions must not contain NaN or Inf."""
        for model_name in ("ridge", "nn"):
            preds = pipeline_run["per_target_preds"][model_name]
            for target, arr in preds.items():
                assert np.isfinite(arr).all(), (
                    f"{model_name}.{target} has NaN/Inf: "
                    f"{int(np.sum(~np.isfinite(arr)))} bad values"
                )

    def test_prediction_shapes(self, synthetic_splits, pipeline_run):
        """Prediction vectors must align with the test set length."""
        _, _, test = synthetic_splits
        expected_n = len(test)  # filter_to_qb keeps all 25 x 17 = 425 rows
        for model_name in ("ridge", "nn"):
            for target, arr in pipeline_run["per_target_preds"][model_name].items():
                assert arr.shape == (expected_n,), (
                    f"{model_name}.{target} shape {arr.shape} != ({expected_n},)"
                )

    def test_bit_identical_across_runs(self, pipeline_run, pipeline_run_repeat):
        """Two runs with the same seed produce bit-identical predictions.

        This is the strongest reproducibility signal — any non-determinism in
        training (unseeded RNG, non-deterministic GPU op, etc.) would surface
        here first.
        """
        # Ridge: closed-form, exact equality expected.
        ridge1 = pipeline_run["per_target_preds"]["ridge"]
        ridge2 = pipeline_run_repeat["per_target_preds"]["ridge"]
        for target in ridge1:
            np.testing.assert_array_equal(
                ridge1[target],
                ridge2[target],
                err_msg=f"Ridge {target} not bit-identical",
            )

        # NN: CPU training with torch.manual_seed is deterministic.
        nn1 = pipeline_run["per_target_preds"]["nn"]
        nn2 = pipeline_run_repeat["per_target_preds"]["nn"]
        for target in nn1:
            torch.testing.assert_close(
                torch.from_numpy(nn1[target]),
                torch.from_numpy(nn2[target]),
                atol=0,
                rtol=0,
                msg=f"NN {target} not bit-identical",
            )
