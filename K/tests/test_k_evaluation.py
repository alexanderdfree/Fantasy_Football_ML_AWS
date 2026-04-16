"""Tests for shared.evaluation — compute_target_metrics, compute_ranking_metrics (K targets)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from shared.evaluation import compute_target_metrics, compute_ranking_metrics

K_TARGETS = ["fg_points", "pat_points"]


class TestComputeTargetMetrics:
    def _make_dicts(self, n=50):
        np.random.seed(42)
        y_true = {
            "fg_points": np.random.rand(n) * 10,
            "pat_points": np.random.rand(n) * 4,
            "total": np.random.rand(n) * 14,
        }
        y_pred = {k: v + np.random.randn(n) * 0.5 for k, v in y_true.items()}
        return y_true, y_pred

    @patch("shared.evaluation.compute_metrics")
    def test_calls_compute_metrics_for_each_target(self, mock_metrics):
        """Kickers have 2 targets, so 2+1(total) = 3 calls."""
        mock_metrics.return_value = {"mae": 1.0, "rmse": 1.5, "r2": 0.8}
        y_true, y_pred = self._make_dicts()
        result = compute_target_metrics(y_true, y_pred, K_TARGETS)
        assert mock_metrics.call_count == 3  # total + 2 targets
        assert set(result.keys()) == {"total", "fg_points", "pat_points"}

    @patch("shared.evaluation.compute_metrics")
    def test_returns_correct_structure(self, mock_metrics):
        mock_metrics.return_value = {"mae": 2.0, "rmse": 3.0, "r2": 0.5}
        y_true, y_pred = self._make_dicts(10)
        result = compute_target_metrics(y_true, y_pred, K_TARGETS)
        for target in result:
            assert "mae" in result[target]
            assert "rmse" in result[target]
            assert "r2" in result[target]

    @patch("shared.evaluation.compute_metrics")
    def test_perfect_predictions(self, mock_metrics):
        mock_metrics.return_value = {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
        y = {
            "fg_points": np.array([9.0, 12.0]),
            "pat_points": np.array([3.0, 2.0]),
            "total": np.array([12.0, 14.0]),
        }
        result = compute_target_metrics(y, y, K_TARGETS)
        assert result["total"]["mae"] == 0.0


class TestComputeRankingMetrics:
    def _make_test_df(self, n_weeks=3, n_players=15):
        rows = []
        np.random.seed(42)
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                rows.append({
                    "week": week,
                    "player_id": f"K{pid}",
                    "pred_total": np.random.rand() * 12,
                    "fantasy_points": np.random.rand() * 12,
                })
        return pd.DataFrame(rows)

    def test_basic_structure(self):
        df = self._make_test_df()
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=5)
        assert "weekly" in result
        assert "season_avg_hit_rate" in result
        assert "season_avg_spearman" in result

    def test_weekly_count_matches(self):
        df = self._make_test_df(n_weeks=4, n_players=15)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 4

    def test_hit_rate_bounds(self):
        df = self._make_test_df(n_weeks=2, n_players=20)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=10)
        for week_result in result["weekly"]:
            assert 0.0 <= week_result["top_k_hit_rate"] <= 1.0

    def test_perfect_predictions_hit_rate(self):
        df = self._make_test_df(n_weeks=1, n_players=15)
        df["pred_total"] = df["fantasy_points"]
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert pytest.approx(result["weekly"][0]["top_k_hit_rate"]) == 1.0

    def test_spearman_on_perfect_prediction(self):
        df = self._make_test_df(n_weeks=1, n_players=20)
        df["pred_total"] = df["fantasy_points"]
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=10)
        assert pytest.approx(result["weekly"][0]["spearman"], abs=0.01) == 1.0

    def test_weeks_with_fewer_than_top_k_skipped(self):
        rows = [
            {"week": 1, "player_id": "K1", "pred_total": 10, "fantasy_points": 10},
            {"week": 1, "player_id": "K2", "pred_total": 8, "fantasy_points": 8},
        ]
        df = pd.DataFrame(rows)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 0
        assert result["season_avg_hit_rate"] == 0.0
        assert result["season_avg_spearman"] == 0.0

    def test_constant_predictions_spearman(self):
        rows = []
        for pid in range(1, 16):
            rows.append({
                "week": 1,
                "player_id": f"K{pid}",
                "pred_total": 7.0,
                "fantasy_points": float(pid),
            })
        df = pd.DataFrame(rows)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert np.isnan(result["weekly"][0]["spearman"])

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["week", "player_id", "pred_total", "fantasy_points"])
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 0
        assert result["season_avg_hit_rate"] == 0.0

    def test_single_week(self):
        df = self._make_test_df(n_weeks=1, n_players=15)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 1
        assert result["season_avg_hit_rate"] == result["weekly"][0]["top_k_hit_rate"]
