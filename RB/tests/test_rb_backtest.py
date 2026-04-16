"""Tests for shared.backtest — run_weekly_simulation (RB-scale)."""

import numpy as np
import pandas as pd
import pytest

from shared.backtest import run_weekly_simulation


@pytest.mark.unit
class TestRunWeeklySimulation:
    def test_basic_structure(self, sim_df_default):
        result = run_weekly_simulation(
            sim_df_default,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert "weekly_metrics" in result
        assert "weekly_ranking" in result
        assert "season_summary" in result

    def test_weekly_metrics_per_model(self, make_sim_df):
        df = make_sim_df(n_weeks=3, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["weekly_metrics"]
        assert len(result["weekly_metrics"]["Ridge"]) == 3

    def test_season_summary_keys(self, sim_df_default):
        result = run_weekly_simulation(
            sim_df_default,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        summary = result["season_summary"]["Ridge"]
        assert "mae" in summary
        assert "rmse" in summary
        assert "r2" in summary

    def test_weekly_ranking_skips_small_weeks(self):
        rows = [
            {"week": 1, "player_id": "P1", "fantasy_points": 10, "pred": 10},
            {"week": 1, "player_id": "P2", "fantasy_points": 8, "pred": 8},
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

    def test_multiple_models(self, sim_df_default):
        result = run_weekly_simulation(
            sim_df_default,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["season_summary"]
        assert "NN" in result["season_summary"]

    def test_perfect_predictions(self, make_sim_df):
        df = make_sim_df(n_weeks=2, n_players=15)
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

    def test_single_week(self, make_sim_df):
        df = make_sim_df(n_weeks=1, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_metrics"]["Ridge"]) == 1
        assert len(result["weekly_ranking"]["Ridge"]) == 1

    def test_top_k_larger_than_players(self, make_sim_df):
        df = make_sim_df(n_weeks=2, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=50,
        )
        assert len(result["weekly_ranking"]["Ridge"]) == 0


@pytest.mark.unit
class TestRunWeeklySimulationDeterminism:
    """Fixed-seed simulation should produce identical weekly rankings and
    season summaries across repeated calls. Regression-critical since the
    backtest is what downstream reports consume.
    """

    def test_identical_results_same_seed(self, make_sim_df):
        df1 = make_sim_df(n_weeks=4, n_players=15, seed=42)
        df2 = make_sim_df(n_weeks=4, n_players=15, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

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

        # Season summary: exact-match MAE / RMSE / R2 across repeated calls.
        for model in ("Ridge", "NN"):
            for metric in ("mae", "rmse", "r2"):
                assert r1["season_summary"][model][metric] == r2["season_summary"][model][metric]

        # Weekly rankings (top-K hit rate, Spearman) must line up week-by-week.
        for model in ("Ridge", "NN"):
            assert len(r1["weekly_ranking"][model]) == len(r2["weekly_ranking"][model])
            for a, b in zip(r1["weekly_ranking"][model], r2["weekly_ranking"][model]):
                assert a["week"] == b["week"]
                assert a["top_k_hit_rate"] == b["top_k_hit_rate"]
                # Spearman may be NaN; use np.allclose with equal_nan.
                assert np.allclose([a["spearman"]], [b["spearman"]], equal_nan=True)
