"""Unit tests for src/analysis/sleeper_loader.py.

The real loader hits Sleeper's API + nflverse — out of scope for unit tests.
These inject a stub ``reader`` (returns synthetic JSON records) and a stub
``player_ids`` crosswalk so the column mapping, the sleeper_id→gsis_id join, the
cache round-trip, the offense-only filter, and the 2018-floor guard are all
exercised without network. Import smoke included.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.analysis import sleeper_loader as mod

pytestmark = pytest.mark.unit


def _fake_reader(url: str) -> list:
    """Mimic Sleeper's /projections/nfl/{season}/{week} payload (per-record shape)."""
    m = re.search(r"/nfl/(\d+)/(\d+)\?", url)
    season, week = int(m.group(1)), int(m.group(2))
    return [
        {
            "player_id": "4881",
            "season": season,
            "week": week,
            "company": "rotowire",
            "player": {
                "position": "QB",
                "first_name": "Lamar",
                "last_name": "Jackson",
                "team_abbr": "BAL",
            },
            "stats": {
                "pass_yd": 210.0,
                "pass_td": 1.5,
                "pass_int": 0.6,
                "rush_yd": 50.0,
                "rush_td": 0.3,
                "fum_lost": 0.2,
                "pts_ppr": 22.0,
            },
        },
        {
            "player_id": "6904",
            "season": season,
            "week": week,
            "company": "rotowire",
            "player": {
                "position": "RB",
                "first_name": "Bijan",
                "last_name": "Robinson",
                "team_abbr": "ATL",
            },
            "stats": {
                "rush_yd": 75.0,
                "rush_td": 0.5,
                "rec": 4.0,
                "rec_yd": 35.0,
                "rec_td": 0.2,
                "fum_lost": 0.1,
                "pts_ppr": 17.0,
            },
        },
        {
            # DST record — ingested (team-keyed). "LAR" exercises the LAR->LA fixup;
            # player.team_abbr is null (the team is at the record top level).
            "player_id": "LAR",
            "season": season,
            "week": week,
            "company": "rotowire",
            "team": "LAR",
            "player": {"position": "DEF", "first_name": "", "last_name": "", "team_abbr": None},
            "stats": {
                "sack": 2.5,
                "int": 0.8,
                "fum_rec": 0.5,
                "ff": 0.7,
                "safe": 0.02,
                "def_td": 0.1,
                "blk_kick": 0.05,
                "st_td": 0.04,
                "pts_allow": 21.0,
                "yds_allow": 340.0,
                "pts_ppr": 7.0,
            },
        },
        {
            # Unprojected roster placeholder (no pts_ppr, no mapped stat) — must
            # be dropped so it isn't ingested as a confident 0.0 projection.
            "player_id": "99999",
            "season": season,
            "week": week,
            "company": "rotowire",
            "player": {
                "position": "QB",
                "first_name": "Bench",
                "last_name": "Qb",
                "team_abbr": "NYJ",
            },
            "stats": {"adp_dd_ppr": 999.0},
        },
    ]


def _fake_player_ids() -> pd.DataFrame:
    """Crosswalk subset: sleeper_id is float (as in ff_playerids); one NaN row."""
    return pd.DataFrame(
        {
            "sleeper_id": [4881.0, 6904.0, float("nan")],
            "gsis_id": ["00-0034796", "00-0038542", "00-0000000"],
            "name": ["Lamar Jackson", "Bijan Robinson", "Nobody"],
        }
    )


def test_module_imports_cleanly() -> None:
    assert hasattr(mod, "load_sleeper_projections")
    assert hasattr(mod, "load_sleeper_with_gsis_id")
    assert mod.SLEEPER_STAT_MAP["pass_yd"] == "passing_yards"
    assert mod._MIN_SEASON == 2018


def test_offense_mapping_and_placeholder_filter(tmp_path) -> None:
    df = mod.load_sleeper_projections(
        [2024], cache_dir=str(tmp_path), weeks=[1, 2], reader=_fake_reader
    )
    # Per week: QB + RB + DST (the unprojected QB placeholder is dropped) -> 6 rows.
    assert len(df) == 6
    assert set(df["position"]) == {"QB", "RB", "DST"}
    assert "99999" not in set(df["sleeper_player_id"])  # placeholder excluded
    qb = df[(df["position"] == "QB") & (df["week"] == 1)].iloc[0]
    assert qb["passing_yards"] == 210.0
    assert qb["passing_tds"] == 1.5
    assert qb["interceptions"] == 0.6
    assert qb["sleeper_player_id"] == "4881"
    # Cross-position stats are 0-filled (QB carries no receiving / DST stats here).
    assert qb["receiving_yards"] == 0.0
    assert qb["def_sacks"] == 0.0


def test_dst_mapping_and_team_fixup(tmp_path) -> None:
    df = mod.load_sleeper_projections(
        [2024], cache_dir=str(tmp_path), weeks=[1], reader=_fake_reader
    )
    dst = df[df["position"] == "DST"]
    assert len(dst) == 1
    row = dst.iloc[0]
    assert row["sleeper_player_id"] == "LA"  # LAR -> LA fixup (nflverse convention)
    assert row["team"] == "LA"
    assert row["def_sacks"] == 2.5
    assert row["def_ints"] == 0.8
    assert row["special_teams_tds"] == 0.04  # st_td -> special_teams_tds
    assert row["points_allowed"] == 21.0
    assert row["yards_allowed"] == 340.0
    assert row["passing_yards"] == 0.0  # offense targets 0-filled on a DST row


def test_dst_team_keyed_offense_gsis(tmp_path) -> None:
    df = mod.load_sleeper_with_gsis_id(
        [2024],
        cache_dir=str(tmp_path),
        weeks=[1],
        reader=_fake_reader,
        player_ids_loader=_fake_player_ids,
    )
    assert df[df["position"] == "DST"].iloc[0]["player_id"] == "LA"  # team-keyed, no gsis
    assert df[df["position"] == "QB"].iloc[0]["player_id"] == "00-0034796"  # offense gsis-bridged


def test_gsis_join(tmp_path) -> None:
    df = mod.load_sleeper_with_gsis_id(
        [2024],
        cache_dir=str(tmp_path),
        weeks=[1],
        reader=_fake_reader,
        player_ids_loader=_fake_player_ids,
    )
    by_sleeper = df.set_index("sleeper_player_id")
    assert by_sleeper.loc["4881", "player_id"] == "00-0034796"  # Lamar
    assert by_sleeper.loc["6904", "player_id"] == "00-0038542"  # Bijan


def test_pre_2018_seasons_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="2018"):
        mod.load_sleeper_projections([2017], cache_dir=str(tmp_path), reader=_fake_reader)


def test_cache_round_trip(tmp_path) -> None:
    df1 = mod.load_sleeper_projections(
        [2024], cache_dir=str(tmp_path), weeks=[1], reader=_fake_reader
    )

    def _boom(url):  # would fail if the network were hit again
        raise AssertionError("reader should not be called on a cache hit")

    df2 = mod.load_sleeper_projections([2024], cache_dir=str(tmp_path), weeks=[1], reader=_boom)
    pd.testing.assert_frame_equal(df1, df2)
