"""Weekly selection must never combine different seasons' player slates."""

import pandas as pd
import pytest

from src.analysis.analysis_expert_comparison import _compare_one_position
from src.shared.backtest import run_weekly_simulation
from src.shared.evaluation import compute_ranking_metrics

pytestmark = pytest.mark.unit


def _frame(seasons=(2024, 2025)):
    return pd.DataFrame(
        [
            {
                "player_id": f"P{i}",
                "season": season,
                "week": 1,
                "fantasy_points": float(i),
                "pred_total": float(i),
                "pred_attn_nn_total": float(i),
                "receiving_yards": float(i * 10),
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "fumbles_lost": 0.0,
            }
            for season in seasons
            for i in range(1, 25)
        ]
    )


@pytest.mark.parametrize("seasons", [(2024,), (2024, 2025)])
def test_perfect_rankings_are_per_season_week(seasons):
    result = compute_ranking_metrics(_frame(seasons), top_k=12)
    assert result["season_avg_hit_rate"] == 1.0
    assert len(result["weekly"]) == len(seasons)
    assert [row["season"] for row in result["weekly"]] == list(seasons)


def test_week_only_input_remains_supported():
    frame = _frame((2025,)).drop(columns="season")
    result = compute_ranking_metrics(frame, top_k=12)
    assert result["season_avg_hit_rate"] == 1.0
    assert "season" not in result["weekly"][0]


def test_backtest_keeps_distinct_season_slates():
    result = run_weekly_simulation(_frame(), {"perfect": "pred_total"})
    assert result["season_summary"]["perfect"]["mae"] == 0.0
    assert len(result["weekly_metrics"]["perfect"]) == 2
    assert [row["season"] for row in result["weekly_metrics"]["perfect"]] == [2024, 2025]
    assert [row["top_k_hit_rate"] for row in result["weekly_ranking"]["perfect"]] == [1.0, 1.0]


def test_expert_comparison_ranks_requested_seasons_independently():
    model = _frame()
    expert = model[["player_id", "season", "week", "pred_total"]].rename(
        columns={"pred_total": "expert_pred_total"}
    )
    result = _compare_one_position("WR", model, expert, "control", [2024, 2025], "ppr", 20, 0)
    assert result["model"]["mae"] == 0.0
    assert result["model"]["top_k_hit_rate"] == 1.0
    assert result["expert"]["top_k_hit_rate"] == 1.0
