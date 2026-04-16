"""Tests for shared.backtest — run_weekly_simulation (QB context)."""

import numpy as np
import pandas as pd
import pytest

from shared.backtest import run_weekly_simulation


@pytest.mark.unit
class TestRunWeeklySimulation:
    def test_basic_structure(self, make_sim_df):
        df = make_sim_df()
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert "weekly_metrics" in result
        assert "weekly_ranking" in result
        assert "season_summary" in result

    def test_weekly_metrics_per_model(self, make_sim_df):
        df = make_sim_df(n_weeks=3)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["weekly_metrics"]
        assert len(result["weekly_metrics"]["Ridge"]) == 3

    def test_season_summary_keys(self, make_sim_df):
        df = make_sim_df()
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        summary = result["season_summary"]["Ridge"]
        assert "mae" in summary
        assert "rmse" in summary
        assert "r2" in summary

    def test_weekly_ranking_skips_small_weeks(self):
        rows = [
            {"week": 1, "player_id": "QB1", "fantasy_points": 20, "pred": 20},
            {"week": 1, "player_id": "QB2", "fantasy_points": 15, "pred": 15},
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

    def test_multiple_models(self, make_sim_df):
        df = make_sim_df()
        result = run_weekly_simulation(
            df,
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
class TestDeterminism:
    """Two runs with identical input and seed must produce identical results."""

    def test_identical_results_same_seed(self, make_sim_df):
        df_a = make_sim_df(n_weeks=4, n_players=15, seed=42)
        df_b = make_sim_df(n_weeks=4, n_players=15, seed=42)
        pd.testing.assert_frame_equal(df_a, df_b)

        result_a = run_weekly_simulation(
            df_a,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        result_b = run_weekly_simulation(
            df_b,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )

        # Weekly ranking must match exactly (same top-k membership, same spearman)
        for model in ("Ridge", "NN"):
            rank_a = result_a["weekly_ranking"][model]
            rank_b = result_b["weekly_ranking"][model]
            assert len(rank_a) == len(rank_b)
            for wa, wb in zip(rank_a, rank_b):
                assert wa["week"] == wb["week"]
                assert wa["top_k_hit_rate"] == wb["top_k_hit_rate"]
                # spearman may involve NaN; check via numpy equality
                np.testing.assert_equal(wa["spearman"], wb["spearman"])

        # Season summary must match exactly
        for model in ("Ridge", "NN"):
            sa = result_a["season_summary"][model]
            sb = result_b["season_summary"][model]
            assert sa.keys() == sb.keys()
            for k in sa:
                assert sa[k] == sb[k], f"{model}.{k}: {sa[k]} != {sb[k]}"
