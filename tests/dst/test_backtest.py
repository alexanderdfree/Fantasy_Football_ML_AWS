"""Tests for shared.backtest — run_weekly_simulation (DST context).

DST is team-level — ``player_id`` holds the team code.  Fantasy-point
scale is 5–15 per week (see ``make_sim_df`` fixture in conftest.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.shared.backtest import run_weekly_simulation


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
            {"week": 1, "player_id": "KC", "fantasy_points": 10, "pred": 10},
            {"week": 1, "player_id": "SF", "fantasy_points": 8, "pred": 8},
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
class TestBacktestDeterminism:
    """Two runs with the same synthetic input must produce identical
    weekly equity curves.  Guards shared/backtest.py against silent
    non-determinism (e.g., dict-iteration order, unsorted groupby).
    """

    def test_same_seed_bit_identical(self, make_sim_df):
        df1 = make_sim_df(n_weeks=4, n_players=15, seed=42)
        df2 = make_sim_df(n_weeks=4, n_players=15, seed=42)

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

        # Weekly metrics identical
        w1 = r1["weekly_metrics"]["Ridge"]
        w2 = r2["weekly_metrics"]["Ridge"]
        assert len(w1) == len(w2)
        for m1, m2 in zip(w1, w2, strict=False):
            assert m1["week"] == m2["week"]
            np.testing.assert_allclose(m1["mae"], m2["mae"], atol=1e-12)
            np.testing.assert_allclose(m1["rmse"], m2["rmse"], atol=1e-12)

        # Season summary identical
        s1 = r1["season_summary"]["Ridge"]
        s2 = r2["season_summary"]["Ridge"]
        for key in ("mae", "rmse", "r2"):
            np.testing.assert_allclose(s1[key], s2[key], atol=1e-12)
