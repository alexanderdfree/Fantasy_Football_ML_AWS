"""Tests for shared.evaluation — compute_target_metrics, compute_ranking_metrics (K targets)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.shared.evaluation import compute_ranking_metrics, compute_target_metrics

K_TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]
K_TARGETS_SET = set(K_TARGETS)


def _make_target_dicts(n=50):
    """Build y_true / y_pred dicts for compute_target_metrics tests."""
    rng = np.random.default_rng(42)
    y_true = {
        "fg_yard_points": rng.random(n) * 10,
        "pat_points": rng.random(n) * 4,
        "fg_misses": rng.random(n) * 2,
        "xp_misses": rng.random(n) * 1,
        "total": rng.random(n) * 14,
    }
    y_pred = {k: v + rng.standard_normal(n) * 0.5 for k, v in y_true.items()}
    return y_true, y_pred


@pytest.mark.unit
class TestComputeTargetMetrics:
    @patch("src.shared.evaluation.compute_metrics")
    def test_calls_compute_metrics_for_each_target(self, mock_metrics):
        """Kickers have 4 targets, so 4+1(total) = 5 calls."""
        mock_metrics.return_value = {"mae": 1.0, "rmse": 1.5, "r2": 0.8}
        y_true, y_pred = _make_target_dicts()
        result = compute_target_metrics(y_true, y_pred, K_TARGETS)
        assert mock_metrics.call_count == 5  # total + 4 targets
        assert set(result.keys()) == K_TARGETS_SET | {"total"}

    @patch("src.shared.evaluation.compute_metrics")
    def test_returns_correct_structure(self, mock_metrics):
        mock_metrics.return_value = {"mae": 2.0, "rmse": 3.0, "r2": 0.5}
        y_true, y_pred = _make_target_dicts(10)
        result = compute_target_metrics(y_true, y_pred, K_TARGETS)
        for target in result:
            assert "mae" in result[target]
            assert "rmse" in result[target]
            assert "r2" in result[target]

    @patch("src.shared.evaluation.compute_metrics")
    def test_perfect_predictions(self, mock_metrics):
        mock_metrics.return_value = {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
        y = {
            "fg_yard_points": np.array([9.0, 12.0]),
            "pat_points": np.array([3.0, 2.0]),
            "fg_misses": np.array([0.0, 1.0]),
            "xp_misses": np.array([0.0, 0.0]),
            "total": np.array([12.0, 14.0]),
        }
        result = compute_target_metrics(y, y, K_TARGETS)
        assert result["total"]["mae"] == 0.0


@pytest.mark.unit
class TestComputeRankingMetrics:
    def test_basic_structure(self, test_df):
        result = compute_ranking_metrics(test_df, "pred_total", "fantasy_points", top_k=5)
        assert "weekly" in result
        assert "season_avg_hit_rate" in result
        assert "season_avg_spearman" in result

    def test_weekly_count_matches(self, make_test_df):
        df = make_test_df(n_weeks=4, n_players=15)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 4

    def test_hit_rate_bounds(self, make_test_df):
        df = make_test_df(n_weeks=2, n_players=20)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=10)
        for week_result in result["weekly"]:
            assert 0.0 <= week_result["top_k_hit_rate"] <= 1.0

    def test_perfect_predictions_hit_rate(self, make_test_df):
        df = make_test_df(n_weeks=1, n_players=15).copy()
        df["pred_total"] = df["fantasy_points"]
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert pytest.approx(result["weekly"][0]["top_k_hit_rate"]) == 1.0

    def test_spearman_on_perfect_prediction(self, make_test_df):
        df = make_test_df(n_weeks=1, n_players=20).copy()
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
            rows.append(
                {
                    "week": 1,
                    "player_id": f"K{pid}",
                    "pred_total": 7.0,
                    "fantasy_points": float(pid),
                }
            )
        df = pd.DataFrame(rows)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert np.isnan(result["weekly"][0]["spearman"])

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["week", "player_id", "pred_total", "fantasy_points"])
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 0
        assert result["season_avg_hit_rate"] == 0.0

    def test_single_week(self, make_test_df):
        df = make_test_df(n_weeks=1, n_players=15)
        result = compute_ranking_metrics(df, "pred_total", "fantasy_points", top_k=12)
        assert len(result["weekly"]) == 1
        assert result["season_avg_hit_rate"] == result["weekly"][0]["top_k_hit_rate"]
