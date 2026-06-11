"""Unit tests for the nflreadpy → pandas boundary shim (``src/data/nfl_source.py``).

These mock ``nflreadpy``'s ``load_*`` functions with small Polars frames so the
shim's schema-reconciliation logic (renames, ``player_id`` aliasing, PBP column
projection, Polars→pandas conversion) is exercised without any network access.
This is where the legacy-weekly rename contract lives after the migration unified
the loader's weekly path (see tests/test_data_loader.py).
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import src.data.nfl_source as nfl_source


@pytest.mark.unit
def test_weekly_data_harmonizes_modern_schema(monkeypatch):
    """The modern ``stats_player`` columns get renamed to the legacy weekly
    names the pipeline keys on, and the result is a pandas frame."""
    modern = pl.DataFrame(
        {
            "player_id": ["P00"],
            "season": [2023],
            "week": [1],
            "team": ["KC"],
            "passing_interceptions": [1],
            "sacks_suffered": [2],
            "sack_yards_lost": [14],
        }
    )
    monkeypatch.setattr(
        nfl_source._nflreadpy,
        "load_player_stats",
        lambda seasons, summary_level="week": modern,
    )

    out = nfl_source.weekly_data([2023])

    assert isinstance(out, pd.DataFrame)
    assert {"recent_team", "interceptions", "sacks", "sack_yards"} <= set(out.columns)
    # Pre-rename names must not survive (they'd diverge from downstream schema).
    assert {"team", "passing_interceptions", "sacks_suffered", "sack_yards_lost"}.isdisjoint(
        out.columns
    )
    assert out.iloc[0]["recent_team"] == "KC"
    assert out.iloc[0]["interceptions"] == 1


@pytest.mark.unit
def test_rosters_adds_player_id_from_gsis(monkeypatch):
    """``load_rosters`` keys by ``gsis_id``; the shim adds the ``player_id``
    alias the loader's roster merge expects."""
    ros = pl.DataFrame(
        {"gsis_id": ["00-0000001"], "season": [2023], "position": ["QB"], "team": ["KC"]}
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_rosters", lambda seasons: ros)

    out = nfl_source.rosters([2023])

    assert isinstance(out, pd.DataFrame)
    assert "player_id" in out.columns
    assert out.iloc[0]["player_id"] == "00-0000001"


@pytest.mark.unit
def test_pbp_data_selects_only_requested_columns(monkeypatch):
    """``pbp_data`` projects to the requested columns before converting — this
    replaces nfl_data_py's removed ``columns=``/``downcast=`` params."""
    pbp = pl.DataFrame(
        {
            "season": [2023],
            "season_type": ["REG"],
            "week": [1],
            "posteam": ["KC"],
            "yardline_100": [5],
            "rusher_player_id": ["R1"],
            "receiver_player_id": [None],
            "pass_attempt": [0],
            "play_type": ["run"],
            "two_point_attempt": [0],
            "extra_unused_col": [99],
        }
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", lambda seasons: pbp)

    out = nfl_source.pbp_data([2023], nfl_source.PBP_REDZONE_COLS)

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(nfl_source.PBP_REDZONE_COLS)
    assert "extra_unused_col" not in out.columns


@pytest.mark.unit
def test_pbp_data_tolerates_missing_columns(monkeypatch):
    """A season missing a requested PBP column degrades to the available subset
    rather than raising (mirrors the per-year try/except in the consumers)."""
    pbp = pl.DataFrame({"season": [2023], "week": [1], "posteam": ["KC"]})
    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", lambda seasons: pbp)

    out = nfl_source.pbp_data([2023], nfl_source.PBP_REDZONE_COLS)

    assert set(out.columns) <= set(nfl_source.PBP_REDZONE_COLS)
    assert "season" in out.columns


@pytest.mark.unit
def test_teams_passthrough_has_logo_columns(monkeypatch):
    """``load_teams`` already exposes ``team_abbr`` + ``team_logo_espn``; the
    shim passes them through unchanged so dst/data.py's logo map is untouched."""
    teams = pl.DataFrame({"team_abbr": ["KC"], "team_logo_espn": ["https://example.test/kc.png"]})
    monkeypatch.setattr(nfl_source._nflreadpy, "load_teams", lambda: teams)

    out = nfl_source.teams()

    assert isinstance(out, pd.DataFrame)
    assert {"team_abbr", "team_logo_espn"} <= set(out.columns)
    assert out.iloc[0]["team_logo_espn"].endswith("kc.png")


@pytest.mark.unit
def test_rosters_weekly_adds_player_id_from_gsis(monkeypatch):
    """``load_rosters_weekly`` is the per-(player, week) status frame the
    inheritance out-set consumes (#1106); like ``rosters``, the shim adds the
    ``player_id`` alias from ``gsis_id``."""
    ros = pl.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001"],
            "season": [2023, 2023],
            "week": [1, 2],
            "position": ["RB", "RB"],
            "team": ["KC", "KC"],
            "status": ["ACT", "INA"],
        }
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_rosters_weekly", lambda seasons: ros)

    out = nfl_source.rosters_weekly([2023])

    assert isinstance(out, pd.DataFrame)
    assert "player_id" in out.columns
    assert len(out) == 2  # one row per week, not per season
    assert set(out["status"]) == {"ACT", "INA"}
