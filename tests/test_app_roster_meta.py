"""Age + rookie-status row enrichment (serving row-contract v7).

Covers the three layers added for the Age / Rookies filters:
- ``src.serving.roster_meta`` — parquet-backed (player, season) meta with the
  loader's stringified-column quirk and graceful degradation.
- ``serialization._records_to_player_rows`` — ``age``/``is_rookie`` on every
  player row (int / bool / None at the JSON boundary).
- ``/api/predictions`` — the fields reach the wire; DST rows serialize null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.serving.roster_meta as roster_meta
from src.serving.serialization import _records_to_player_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def meta_env(tmp_path, monkeypatch):
    """Point roster_meta at synthetic rosters/schedules parquets in tmp_path."""
    rosters = pd.DataFrame(
        {
            # Stringified object columns, mirroring loader._fetch_rosters'
            # astype(str) coercion (incl. the literal "None" for missing).
            "player_id": ["P1", "P2", "P3"],
            "season": [2025, 2025, 2025],
            "birth_date": ["2000-03-15", "1995-11-02", "None"],
            "entry_year": ["2025", "2017", "None"],
            "rookie_year": ["2025", "2017", "None"],
        }
    )
    schedules = pd.DataFrame(
        {
            "season": [2025],
            "week": [3],
            "gameday": ["2025-09-21"],
            "home_team": ["KC"],
            "away_team": ["BUF"],
        }
    )
    rosters_path = tmp_path / "rosters.parquet"
    schedules_path = tmp_path / "schedules.parquet"
    rosters.to_parquet(rosters_path)
    schedules.to_parquet(schedules_path)
    monkeypatch.setattr(roster_meta, "_rosters_path", lambda: str(rosters_path))
    monkeypatch.setattr(roster_meta, "_schedules_path", lambda: str(schedules_path))
    roster_meta.reset_caches()
    yield tmp_path
    roster_meta.reset_caches()


def _results_frame():
    return pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3", "KC"],
            "season": [2025, 2025, 2025, 2025],
            "week": [3, 3, 3, 3],
            "recent_team": ["KC", "BUF", "KC", "KC"],
            "position": ["WR", "QB", "TE", "DST"],
        }
    )


class TestAttachAgeAndRookie:
    def test_age_is_computed_at_kickoff(self, meta_env):
        out = roster_meta.attach_age_and_rookie(_results_frame())
        # P1 born 2000-03-15, game 2025-09-21 -> 25.
        assert out.loc[0, "age"] == 25.0
        # P2 born 1995-11-02 -> 29 (birthday not yet reached at kickoff).
        assert out.loc[1, "age"] == 29.0

    def test_rookie_flag_from_entry_year(self, meta_env):
        out = roster_meta.attach_age_and_rookie(_results_frame())
        assert out.loc[0, "is_rookie"] == 1.0  # entry_year == season
        assert out.loc[1, "is_rookie"] == 0.0  # 2017 veteran

    def test_unmatched_rows_stay_nan(self, meta_env):
        out = roster_meta.attach_age_and_rookie(_results_frame())
        # P3 has a stringified-"None" birth_date/entry_year; the DST team unit
        # has no roster row at all — both must degrade to NaN, not raise.
        assert np.isnan(out.loc[2, "age"])
        assert np.isnan(out.loc[2, "is_rookie"])
        assert np.isnan(out.loc[3, "age"])
        assert np.isnan(out.loc[3, "is_rookie"])

    def test_missing_parquets_degrade_to_nan(self, tmp_path, monkeypatch):
        monkeypatch.setattr(roster_meta, "_rosters_path", lambda: str(tmp_path / "absent.parquet"))
        monkeypatch.setattr(
            roster_meta, "_schedules_path", lambda: str(tmp_path / "absent2.parquet")
        )
        roster_meta.reset_caches()
        try:
            out = roster_meta.attach_age_and_rookie(_results_frame())
            assert out["age"].isna().all()
            assert out["is_rookie"].isna().all()
        finally:
            roster_meta.reset_caches()

    def test_missing_gameday_uses_season_fallback(self, meta_env, monkeypatch):
        # Week 10 has no schedule row -> nominal Dec 1 fallback; P1 (born
        # 2000-03-15) is 25 on 2025-12-01 as well, P2 turns 30 on Nov 2.
        frame = _results_frame().assign(week=10)
        out = roster_meta.attach_age_and_rookie(frame)
        assert out.loc[0, "age"] == 25.0
        assert out.loc[1, "age"] == 30.0


class TestRowSerialization:
    def test_rows_carry_int_age_and_bool_rookie(self):
        df = pd.DataFrame(
            {
                "player_id": ["a", "b"],
                "player_display_name": ["A", "B"],
                "position": ["WR", "DST"],
                "recent_team": ["KC", "KC"],
                "week": [1, 1],
                "fantasy_points": [10.0, 5.0],
                "age": [25.0, np.nan],
                "is_rookie": [1.0, np.nan],
            }
        )
        rows = _records_to_player_rows(df, scoring="ppr")
        assert rows[0]["age"] == 25 and isinstance(rows[0]["age"], int)
        assert rows[0]["is_rookie"] is True
        # DST / unmatched rows serialize null so the frontend feature-detect
        # (hide Age/Rookies when no row carries age) keeps working.
        assert rows[1]["age"] is None
        assert rows[1]["is_rookie"] is None


class TestPredictionsEndpoint:
    def test_api_rows_expose_age_and_rookie(self, client_with_data):
        resp = client_with_data.get("/api/predictions?position=WR&scoring=ppr")
        players = resp.get_json()["players"]
        assert players, "fixture should yield WR rows"
        assert all("age" in p and "is_rookie" in p for p in players)
        aged = [p for p in players if p["age"] is not None]
        assert aged, "synthetic fixture stamps WR ages"
        assert all(isinstance(p["age"], int) for p in aged)

    def test_dst_rows_serialize_null_age(self, client_with_data):
        resp = client_with_data.get("/api/predictions?position=DST&scoring=ppr")
        players = resp.get_json()["players"]
        assert players
        assert all(p["age"] is None and p["is_rookie"] is None for p in players)
