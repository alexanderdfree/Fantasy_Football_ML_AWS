"""The scoring intersection is symmetric and excludes unprojected actual stats."""

import numpy as np
import pandas as pd
import pytest

from src.shared.comparison_scoring import score_actual_components, scoring_components

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fmt,rec", [("ppr", 1.0), ("half_ppr", 0.5), ("standard", 0.0)])
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_shared_stat_totals_across_offense_and_formats(pos, fmt, rec):
    raw = pd.DataFrame(
        {
            "passing_yards": [250.0],
            "passing_tds": [2.0],
            "interceptions": [1.0],
            "rushing_yards": [40.0],
            "rushing_tds": [1.0],
            "receiving_yards": [50.0],
            "receiving_tds": [1.0],
            "receptions": [5.0],
            "fumbles_lost": [1.0],
        }
    )
    expected = {"QB": 24.0, "RB": 19.0 + 5 * rec, "WR": 9.0 + 5 * rec, "TE": 9.0 + 5 * rec}
    assert score_actual_components(raw, pos, fmt).iloc[0] == pytest.approx(expected[pos])
    served = raw.add_prefix("actual_").assign(fantasy_points=999.0)
    assert score_actual_components(served, pos, fmt, prefix="actual_").iloc[0] == pytest.approx(
        expected[pos]
    )


@pytest.mark.parametrize("pos", ["K", "DST"])
def test_missing_native_component_cannot_become_a_zero_or_tier_bonus(pos):
    raw = pd.DataFrame({name: [0.0, 0.0] for name in scoring_components(pos)})
    raw.loc[1, scoring_components(pos)[0]] = np.nan
    result = score_actual_components(raw, pos)
    assert np.isfinite(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert score_actual_components(raw.drop(columns=scoring_components(pos)[0]), pos).isna().all()
