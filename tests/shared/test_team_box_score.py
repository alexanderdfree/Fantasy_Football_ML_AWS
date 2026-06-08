"""Regression test for `_build_team_box_score_lookup` team-code normalization.

Schedules are the lone source still carrying legacy pre-relocation team codes
(OAK/SD/STL); `team_stats` uses modern codes (LV/LAC/LAR). Without normalizing
the schedule side before the `(season, week, team)` merge, pre-2017 OAK/SD/STL
rows fail to align and their `team_points_scored` is silently zeroed (#862).
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.shared.team_box_score as tbs

pytestmark = pytest.mark.unit


def test_legacy_schedule_team_codes_normalized_before_points_merge(monkeypatch):
    # 2015 Raiders game: schedules still say "OAK"; team_stats says "LV".
    schedules = pd.DataFrame(
        {
            "season": [2015],
            "week": [1],
            "home_team": ["OAK"],
            "away_team": ["KC"],
            "home_score": [20.0],
            "away_score": [14.0],
        }
    )
    team_stats = pd.DataFrame(
        {
            "team": ["LV", "KC"],
            "season": [2015, 2015],
            "week": [1, 1],
            "attempts": [30, 28],
            "completions": [20, 18],
            "passing_yards": [250.0, 240.0],
            "carries": [25, 22],
            "rushing_yards": [120.0, 110.0],
            "passing_interceptions": [1, 0],
            "rushing_fumbles_lost": [0, 0],
            "receiving_fumbles_lost": [0, 0],
            "sack_fumbles_lost": [0, 0],
        }
    )
    monkeypatch.setattr(tbs, "_load_schedules", lambda: schedules)
    monkeypatch.setattr(tbs, "load_team_week_stats", lambda seasons, cache_dir=None: team_stats)

    out = tbs._build_team_box_score_lookup()

    lv = out[(out["team"] == "LV") & (out["week"] == 1)].iloc[0]
    # OAK home_score=20 normalized → LV, so the merge aligns and points survive
    # (this read 0.0 before the fix).
    assert lv["team_points_scored"] == 20.0
    assert lv["opp_team_points_scored"] == 14.0
    # No stray legacy 'OAK' row leaks through.
    assert "OAK" not in set(out["team"])
