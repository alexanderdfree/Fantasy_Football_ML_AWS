"""Tests for RB.rb_evaluation — compute_rb_metrics, compute_rb_ranking_metrics."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from RB.rb_evaluation import compute_rb_metrics, compute_rb_ranking_metrics


# ---------------------------------------------------------------------------
# compute_rb_metrics
# ---------------------------------------------------------------------------

class TestComputeRBMetrics:
    def _make_dicts(self, n=50):
        np.random.seed(42)
        y_true = {
            "rushing_floor": np.random.rand(n) * 10,
            "receiving_floor": np.random.rand(n) * 8,
            "td_points": np.random.rand(n) * 6,
            "total": np.random.rand(n) * 20,
        }
        y_pred = {k: v + np.random.randn(n) * 0.5 for k, v in y_true.items()}
        return y_true, y_pred

    @patch("RB.rb_evaluation.compute_metrics")
    def test_calls_compute_metrics_for_each_target(self, mock_metrics):
        mock_metrics.return_value = {"mae": 1.0, "rmse": 1.5, "r2": 0.8}
        y_true, y_pred = self._make_dicts()
        result = compute_rb_metrics(y_true, y_pred)
        assert mock_metrics.call_count == 4  # total + 3 targets
        assert set(result.keys()) == {"total", "rushing_floor", "receiving_floor", "td_points"}

    @patch("RB.rb_evaluation.compute_metrics")
    def test_returns_correct_structure(self, mock_metrics):
        mock_metrics.return_value = {"mae": 2.0, "rmse": 3.0, "r2": 0.5}
        y_true, y_pred = self._make_dicts(10)
        result = compute_rb_metrics(y_true, y_pred)
        for target in result:
            assert "mae" in result[target]
            assert "rmse" in result[target]
            assert "r2" in result[target]

    @patch("RB.rb_evaluation.compute_metrics")
    def test_perfect_predictions(self, mock_metrics):
        mock_metrics.return_value = {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
        y = {
            "rushing_floor": np.array([1.0, 2.0]),
            "receiving_floor": np.array([3.0, 4.0]),
            "td_points": np.array([5.0, 6.0]),
            "total": np.array([9.0, 12.0]),
        }
        result = compute_rb_metrics(y, y)
        assert result["total"]["mae"] == 0.0


# ---------------------------------------------------------------------------
# compute_rb_ranking_metrics
# ---------------------------------------------------------------------------

class TestComputeRBRankingMetrics:
    def _make_test_df(self, n_weeks=3, n_players=15):
        """Create a test DataFrame with multiple weeks and players."""
        rows = []
        np.random.seed(42)
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                rows.append({
                    "week": week,
                    "player_id": f"P{pid}",
                    "pred_total": np.random.rand() * 20,
                    "fantasy_points": np.random.rand() * 20,
                })
        return pd.DataFrame(rows)

    def test_basic_structure(self):
        df = self._make_test_df()
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=5)
        assert "weekly" in result
        assert "season_avg_hit_rate" in result
        assert "season_avg_spearman" in result

    def test_weekly_count_matches(self):
        df = self._make_test_df(n_weeks=4, n_players=15)
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 4  # all weeks have >= 12 players

    def test_hit_rate_bounds(self):
        df = self._make_test_df(n_weeks=2, n_players=20)
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=10)
        for week_result in result["weekly"]:
            assert 0.0 <= week_result["top_k_hit_rate"] <= 1.0

    def test_perfect_predictions_hit_rate(self):
        """If predictions == actuals, hit rate should be 1.0."""
        df = self._make_test_df(n_weeks=1, n_players=15)
        df["pred_total"] = df["fantasy_points"]
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert pytest.approx(result["weekly"][0]["top_k_hit_rate"]) == 1.0

    def test_spearman_on_perfect_prediction(self):
        df = self._make_test_df(n_weeks=1, n_players=20)
        df["pred_total"] = df["fantasy_points"]
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=10)
        assert pytest.approx(result["weekly"][0]["spearman"], abs=0.01) == 1.0

    def test_weeks_with_fewer_than_top_k_skipped(self):
        """Weeks with < top_k players should be excluded."""
        rows = [
            {"week": 1, "player_id": "P1", "pred_total": 10, "fantasy_points": 10},
            {"week": 1, "player_id": "P2", "pred_total": 8, "fantasy_points": 8},
        ]
        df = pd.DataFrame(rows)
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 0
        assert result["season_avg_hit_rate"] == 0.0
        assert result["season_avg_spearman"] == 0.0

    def test_constant_predictions_spearman(self):
        """Constant predictions -> Spearman = NaN -> should return 0."""
        rows = []
        for pid in range(1, 16):
            rows.append({
                "week": 1,
                "player_id": f"P{pid}",
                "pred_total": 5.0,  # constant
                "fantasy_points": float(pid),
            })
        df = pd.DataFrame(rows)
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert result["weekly"][0]["spearman"] == 0.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["week", "player_id", "pred_total", "fantasy_points"])
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 0
        assert result["season_avg_hit_rate"] == 0.0

    def test_single_week(self):
        df = self._make_test_df(n_weeks=1, n_players=15)
        result = compute_rb_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 1
        assert result["season_avg_hit_rate"] == result["weekly"][0]["top_k_hit_rate"]
