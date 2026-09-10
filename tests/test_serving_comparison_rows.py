"""Serving rows carry comparison truth separately from full fantasy scores."""

import numpy as np
import pandas as pd
import pytest

from src.serving.serialization import _records_to_player_rows
from src.shared.comparison_scoring import ACTUAL_BASIS, scoring_components

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scoring,expected", [("ppr", 10.0), ("half_ppr", 7.5), ("standard", 5.0)])
def test_comparison_actual_uses_raw_components_and_requested_format(scoring, expected):
    frame = pd.DataFrame(
        {
            "player_id": ["wr"],
            "position": ["WR"],
            "week": [1],
            "fantasy_points": [16.0],
            "fantasy_points_half_ppr": [13.5],
            "fantasy_points_standard": [11.0],
            "actual_receiving_yards": [50.0],
            "actual_receiving_tds": [0.0],
            "actual_receptions": [5.0],
            "actual_fumbles_lost": [0.0],
            "ridge_pred_ppr": [10.0],
        }
    )
    before = frame.copy(deep=True)
    row = _records_to_player_rows(frame, scoring)[0]
    assert row["comparison_actual"] == expected
    assert row["actual"] == expected + 6.0
    assert row["comparison_actual_basis"] == ACTUAL_BASIS
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
@pytest.mark.parametrize("missing", ["column", "nan"])
def test_missing_components_never_use_full_fantasy_fallback(position, missing):
    components = {f"actual_{target}": [0.0] for target in scoring_components(position)}
    column = next(iter(components))
    if missing == "column":
        del components[column]
    else:
        components[column] = [np.nan]
    frame = pd.DataFrame(
        {"position": [position], "week": [1], "fantasy_points": [99.0], **components}
    )
    row = _records_to_player_rows(frame)[0]
    assert row["actual"] == 99.0
    assert row["comparison_actual"] is None
    assert row["comparison_excluded_sources"] == (["nflcom"] if position == "K" else [])


def test_zero_component_actual_is_available():
    frame = pd.DataFrame(
        {
            "position": ["WR"],
            "week": [1],
            "fantasy_points": [3.0],
            **{f"actual_{target}": [0.0] for target in scoring_components("WR")},
        }
    )
    assert _records_to_player_rows(frame)[0]["comparison_actual"] == 0.0
