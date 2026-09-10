import numpy as np
import pandas as pd
import pytest

from src.tuning.ab_air_yards import metric_fn

pytestmark = pytest.mark.unit


def test_qb_td_cohort_scores_passing_games_not_rare_receiving_games():
    frame = pd.DataFrame(
        {
            "passing_tds": [1, 0, 2, 0],
            "receiving_tds": [0, 1, 0, 0],
            "fantasy_points": [20.0, 6.0, 30.0, 2.0],
            "pred_ridge_total": [24.0, 106.0, 32.0, 2.0],
        }
    )
    metrics = metric_fn({"test_df": frame}, "QB")["Ridge"]
    assert metrics["tdgame_n"] == 2
    assert metrics["tdgame_bias"] == 3.0
    assert metrics["tdgame_rmse"] == pytest.approx(np.sqrt(10.0))


@pytest.mark.parametrize("position", ["WR", "TE", "RB"])
def test_receiving_positions_keep_receiving_td_cohort(position):
    frame = pd.DataFrame(
        {
            "passing_tds": [1, 0, 0, 0],
            "receiving_tds": [0, 1, 0, 1],
            "fantasy_points": [20.0, 10.0, 5.0, 30.0],
            "pred_ridge_total": [120.0, 8.0, 5.0, 34.0],
        }
    )
    metrics = metric_fn({"test_df": frame}, position)["Ridge"]
    assert metrics["tdgame_n"] == 2
    assert metrics["tdgame_bias"] == 1.0
    assert metrics["tdgame_rmse"] == pytest.approx(np.sqrt(10.0))


def test_missing_passing_column_does_not_substitute_qb_receiving_games():
    frame = pd.DataFrame(
        {"receiving_tds": [1, 0], "fantasy_points": [6.0, 0.0], "pred_ridge_total": [7.0, 0.0]}
    )
    metrics = metric_fn({"test_df": frame}, "QB")["Ridge"]
    assert "tdgame_n" not in metrics
    assert metrics["mae"] == 0.5
