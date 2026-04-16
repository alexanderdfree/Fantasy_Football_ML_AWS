"""Tests for shared.backtest — run_weekly_simulation (TE context)."""

import numpy as np
import pandas as pd
import pytest

from shared.backtest import run_weekly_simulation


@pytest.mark.unit
class TestRunWeeklySimulation:
    def test_basic_structure(self, te_sim_df):
        result = run_weekly_simulation(
            te_sim_df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert "weekly_metrics" in result
        assert "weekly_ranking" in result
        assert "season_summary" in result

    def test_weekly_metrics_per_model(self, te_sim_df_factory):
        df = te_sim_df_factory(n_weeks=3)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["weekly_metrics"]
        assert len(result["weekly_metrics"]["Ridge"]) == 3

    def test_season_summary_keys(self, te_sim_df):
        result = run_weekly_simulation(
            te_sim_df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        summary = result["season_summary"]["Ridge"]
        assert "mae" in summary
        assert "rmse" in summary
        assert "r2" in summary

    def test_weekly_ranking_skips_small_weeks(self):
        rows = [
            {"week": 1, "player_id": "TE1", "fantasy_points": 10, "pred": 10},
            {"week": 1, "player_id": "TE2", "fantasy_points": 8, "pred": 8},
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

    def test_multiple_models(self, te_sim_df):
        result = run_weekly_simulation(
            te_sim_df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["season_summary"]
        assert "NN" in result["season_summary"]

    def test_perfect_predictions(self, te_sim_df_factory):
        df = te_sim_df_factory(n_weeks=2, n_players=15)
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

    def test_single_week(self, te_sim_df_factory):
        df = te_sim_df_factory(n_weeks=1, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_metrics"]["Ridge"]) == 1
        assert len(result["weekly_ranking"]["Ridge"]) == 1

    def test_top_k_larger_than_players(self, te_sim_df_factory):
        df = te_sim_df_factory(n_weeks=2, n_players=15)
        result = run_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=50,
        )
        assert len(result["weekly_ranking"]["Ridge"]) == 0

    def test_determinism(self, te_sim_df_factory):
        """Same seed -> bit-identical simulation outputs.

        Guards against nondeterminism creeping into `run_weekly_simulation`
        (e.g., from a future change introducing `set`-based iteration order
        into per-week metrics).
        """
        df1 = te_sim_df_factory(n_weeks=3, n_players=15, seed=42)
        df2 = te_sim_df_factory(n_weeks=3, n_players=15, seed=42)

        result1 = run_weekly_simulation(
            df1,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        result2 = run_weekly_simulation(
            df2,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )

        # Season summaries equal down to machine precision.
        for model_name in result1["season_summary"]:
            s1 = result1["season_summary"][model_name]
            s2 = result2["season_summary"][model_name]
            for metric in ("mae", "rmse", "r2"):
                assert s1[metric] == pytest.approx(s2[metric], abs=1e-12), (
                    f"{model_name}.{metric} differs across seeded runs"
                )

        # Weekly metrics aligned tick-for-tick.
        for model_name in result1["weekly_metrics"]:
            wm1 = result1["weekly_metrics"][model_name]
            wm2 = result2["weekly_metrics"][model_name]
            assert len(wm1) == len(wm2)
            for w1, w2 in zip(wm1, wm2):
                assert w1["week"] == w2["week"]
                for metric in ("mae", "rmse", "r2"):
                    assert w1[metric] == pytest.approx(w2[metric], abs=1e-12)
