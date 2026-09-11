"""Native display totals and certified comparison totals remain distinct."""

import numpy as np
import pandas as pd
import pytest

from src.contracts.serialization import _records_to_player_rows
from src.prediction import frames, historical
from src.shared.aggregate_targets import DST_TARGETS, predictions_to_fantasy_points
from src.shared.comparison_truth import attach_comparison_actuals

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
def test_dst_rows_keep_native_bonus_and_shared_missingness_separate(monkeypatch, scoring):
    from src.serving.app import create_app

    frame = pd.DataFrame({target: [0.0, 0.0] for target in DST_TARGETS})
    frame["yards_allowed"] = 349.0
    frame = attach_comparison_actuals(frame, "DST", pd.Series([True, False]))
    raw = {target: frame[target].to_numpy() for target in DST_TARGETS}
    total = predictions_to_fantasy_points("DST", raw)
    prediction = frames.PositionPredictions(
        frame,
        {"ridge": raw},
        {"ridge": dict.fromkeys(("ppr", "half_ppr", "standard"), total)},
        {},
        {},
        {},
    )
    monkeypatch.setattr(frames, "predict_position", lambda *args, **kwargs: prediction)
    results = pd.DataFrame(
        {
            "player_id": ["BUF", "KC"],
            "position": "DST",
            "recent_team": ["BUF", "KC"],
            "week": 1,
            "fantasy_points": 10.0,
            "fantasy_points_half_ppr": 10.0,
            "fantasy_points_standard": 10.0,
        }
    )
    with create_app().app_context():
        historical._apply_position_models(
            frame, frame, frame, "DST", results, opponent_weekly=frame
        )
    results[f"rotowire_pred_{scoring}"] = 12.0
    results[f"rotowire_comparison_pred_{scoring}"] = 2.0
    rows = _records_to_player_rows(results, scoring)
    assert rows[0]["actual"] == 10.0
    assert rows[0]["comparison_actual"] == 0.0
    assert rows[1]["comparison_actual"] is None
    assert rows[0]["ridge_pred"] == 10.0
    assert rows[0]["ridge_comparison_pred"] == 0.0
    assert rows[0]["rotowire_pred"] == 12.0
    assert rows[0]["rotowire_comparison_pred"] == 2.0
    assert rows[0]["nn_comparison_pred"] is None
    assert rows[0]["comparison_actual_basis"] == "shared_projected_components_v2"


def test_legacy_rows_do_not_infer_a_comparison_basis_from_native_totals():
    rows = _records_to_player_rows(
        pd.DataFrame(
            {
                "player_id": ["kicker"],
                "position": ["K"],
                "week": [1],
                "fantasy_points": [0.0],
                "nflcom_pred_ppr": [0.0],
            }
        )
    )
    assert rows[0]["actual"] == 0.0
    assert rows[0]["nflcom_pred"] == 0.0
    assert rows[0]["comparison_actual"] is None
    assert rows[0]["nflcom_comparison_pred"] is None
    assert rows[0]["comparison_actual_basis"] is None
    assert rows[0]["comparison_excluded_sources"] is None
