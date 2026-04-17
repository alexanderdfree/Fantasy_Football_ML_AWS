"""Tests for shared.backtest — run_weekly_simulation (WR context)."""

import numpy as np
import pandas as pd
import pytest

from shared.backtest import run_weekly_simulation


@pytest.mark.unit
class TestRunWeeklySimulation:
    def test_basic_structure(self, wr_sim_df):
        result = run_weekly_simulation(
            wr_sim_df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert "weekly_metrics" in result
        assert "weekly_ranking" in result
        assert "season_summary" in result

    def test_weekly_metrics_per_model(self, wr_sim_df_factory):
        df = wr_sim_df_factory(n_weeks=3)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["weekly_metrics"]
        assert len(result["weekly_metrics"]["Ridge"]) == 3

    def test_season_summary_keys(self, wr_sim_df):
        result = run_weekly_simulation(
            wr_sim_df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        summary = result["season_summary"]["Ridge"]
        assert "mae" in summary
        assert "rmse" in summary
        assert "r2" in summary

    def test_weekly_ranking_skips_small_weeks(self):
        rows = [
            {"week": 1, "player_id": "WR1", "fantasy_points": 10, "pred": 10},
            {"week": 1, "player_id": "WR2", "fantasy_points": 8, "pred": 8},
        ]
        df = pd.DataFrame(rows)
        result = run_weekly_simulation(
            df,
            pred_columns={"M": "pred"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_ranking"]["M"]) == 0
        assert len(result["weekly_metrics"]["M"]) == 1

    def test_multiple_models(self, wr_sim_df):
        result = run_weekly_simulation(
            wr_sim_df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["season_summary"]
        assert "NN" in result["season_summary"]

    def test_perfect_predictions(self, wr_sim_df_factory):
        df = wr_sim_df_factory(n_weeks=2, n_players=15)
        df["pred_perfect"] = df["fantasy_points"]
        result = run_weekly_simulation(
            df,
            pred_columns={"Perfect": "pred_perfect"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert pytest.approx(result["season_summary"]["Perfect"]["mae"], abs=1e-6) == 0.0
        for week_rank in result["weekly_ranking"]["Perfect"]:
            assert pytest.approx(week_rank["top_k_hit_rate"]) == 1.0

    def test_empty_pred_column(self):
        df = pd.DataFrame(columns=["week", "player_id", "fantasy_points", "pred_empty"])
        result = run_weekly_simulation(
            df,
            pred_columns={"Empty": "pred_empty"},
            true_col="fantasy_points",
        )
        assert np.isnan(result["season_summary"]["Empty"]["mae"])

    def test_single_week(self, wr_sim_df_factory):
        df = wr_sim_df_factory(n_weeks=1, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_metrics"]["Ridge"]) == 1
        assert len(result["weekly_ranking"]["Ridge"]) == 1

    def test_top_k_larger_than_players(self, wr_sim_df_factory):
        df = wr_sim_df_factory(n_weeks=2, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=50,
        )
        assert len(result["weekly_ranking"]["Ridge"]) == 0


@pytest.mark.unit
class TestBacktestDeterminism:
    """Identical inputs must yield identical weekly metrics and rankings.

    Guards shared/backtest.py against inadvertent non-determinism introduced
    by e.g. unstable sort order, dict iteration changes, or RNG leakage.
    """

    def test_identical_inputs_yield_identical_summary(self, wr_sim_df_factory):
        df1 = wr_sim_df_factory(seed=42)
        df2 = wr_sim_df_factory(seed=42)
        r1 = run_weekly_simulation(
            df1,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        r2 = run_weekly_simulation(
            df2,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        for model in ("Ridge", "NN"):
            for key in ("mae", "rmse", "r2"):
                assert r1["season_summary"][model][key] == r2["season_summary"][model][key]

    def test_weekly_metrics_bit_identical(self, wr_sim_df_factory):
        df1 = wr_sim_df_factory(seed=42)
        df2 = wr_sim_df_factory(seed=42)
        r1 = run_weekly_simulation(
            df1,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        r2 = run_weekly_simulation(
            df2,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert r1["weekly_metrics"]["Ridge"] == r2["weekly_metrics"]["Ridge"]

    def test_weekly_rankings_bit_identical(self, wr_sim_df_factory):
        df1 = wr_sim_df_factory(n_weeks=5, n_players=20, seed=42)
        df2 = wr_sim_df_factory(n_weeks=5, n_players=20, seed=42)
        r1 = run_weekly_simulation(
            df1,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        r2 = run_weekly_simulation(
            df2,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert r1["weekly_ranking"]["Ridge"] == r2["weekly_ranking"]["Ridge"]
