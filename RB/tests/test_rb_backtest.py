"""Tests for RB.rb_backtest — run_rb_weekly_simulation."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from RB.rb_backtest import run_rb_weekly_simulation


def _make_sim_df(n_weeks=4, n_players=15):
    """Create a test DataFrame for simulation."""
    rows = []
    np.random.seed(42)
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            fp = np.random.rand() * 20
            rows.append({
                "week": week,
                "player_id": f"P{pid}",
                "fantasy_points": fp,
                "pred_ridge": fp + np.random.randn() * 2,
                "pred_nn": fp + np.random.randn() * 3,
            })
    return pd.DataFrame(rows)


class TestRunRBWeeklySimulation:
    def test_basic_structure(self):
        df = _make_sim_df()
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert "weekly_metrics" in result
        assert "weekly_ranking" in result
        assert "season_summary" in result

    def test_weekly_metrics_per_model(self):
        df = _make_sim_df(n_weeks=3)
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["weekly_metrics"]
        assert len(result["weekly_metrics"]["Ridge"]) == 3

    def test_season_summary_keys(self):
        df = _make_sim_df()
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
        )
        summary = result["season_summary"]["Ridge"]
        assert "mae" in summary
        assert "rmse" in summary
        assert "r2" in summary

    def test_weekly_ranking_skips_small_weeks(self):
        """Weeks with fewer than top_k players should have no ranking entry."""
        rows = [
            {"week": 1, "player_id": "P1", "fantasy_points": 10, "pred": 10},
            {"week": 1, "player_id": "P2", "fantasy_points": 8, "pred": 8},
        ]
        df = pd.DataFrame(rows)
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"M": "pred"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_ranking"]["M"]) == 0
        # But weekly_metrics should still have the week
        assert len(result["weekly_metrics"]["M"]) == 1

    def test_multiple_models(self):
        df = _make_sim_df()
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge", "NN": "pred_nn"},
            true_col="fantasy_points",
        )
        assert "Ridge" in result["season_summary"]
        assert "NN" in result["season_summary"]

    def test_perfect_predictions(self):
        df = _make_sim_df(n_weeks=2, n_players=15)
        df["pred_perfect"] = df["fantasy_points"]
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Perfect": "pred_perfect"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert pytest.approx(result["season_summary"]["Perfect"]["mae"], abs=1e-6) == 0.0
        # All hit rates should be 1.0
        for week_rank in result["weekly_ranking"]["Perfect"]:
            assert pytest.approx(week_rank["top_k_hit_rate"]) == 1.0

    def test_empty_pred_column(self):
        """If no predictions exist for a model, season summary should be NaN."""
        df = pd.DataFrame(columns=["week", "player_id", "fantasy_points", "pred_empty"])
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Empty": "pred_empty"},
            true_col="fantasy_points",
        )
        assert np.isnan(result["season_summary"]["Empty"]["mae"])

    def test_single_week(self):
        df = _make_sim_df(n_weeks=1, n_players=15)
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=12,
        )
        assert len(result["weekly_metrics"]["Ridge"]) == 1
        assert len(result["weekly_ranking"]["Ridge"]) == 1

    def test_top_k_larger_than_players(self):
        """top_k=50 but only 15 players: ranking should be empty."""
        df = _make_sim_df(n_weeks=2, n_players=15)
        result = run_rb_weekly_simulation(
            df,
            pred_columns={"Ridge": "pred_ridge"},
            true_col="fantasy_points",
            top_k=50,
        )
        assert len(result["weekly_ranking"]["Ridge"]) == 0
