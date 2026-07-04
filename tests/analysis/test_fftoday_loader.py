"""Unit tests for src.analysis.fftoday_loader.

All tests run offline: the FFToday HTTP fetch is mocked via the loader's
injectable ``reader=`` kwarg (returning crafted FFToday-shaped HTML), and the
nflverse roster fetch is bypassed by passing ``rosters=`` directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import analysis_expert_comparison as aec
from src.analysis.fftoday_loader import (
    FFTODAY_POS_IDS,
    _parse_projection_html,
    load_fftoday_projections,
    load_fftoday_with_gsis_id,
)

pytestmark = pytest.mark.unit


# A FFToday WR row: [Chg, Player(anchor), Team, Opp, rAtt, rYd, rTD, Rec, recYd, recTD, FPts]
_WR_PAGE = """
<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td><td>Team</td><td>Opp</td>
<td>Att</td><td>Yard</td><td>TD</td><td>Rec</td><td>Yard</td><td>TD</td><td>FPts</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/2753/Calvin_Johnson?LeagueID=">Calvin Johnson</a></td>
<td>DET</td><td>MIN</td><td>0.0</td><td>0.0</td><td>0.0</td><td>8.0</td><td>110.0</td><td>1.0</td><td>17.0</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/1000/Jimmy_Graham?LeagueID=">Jimmy Graham</a></td>
<td>NO</td><td>ATL</td><td>0.0</td><td>0.0</td><td>0.0</td><td>9.0</td><td>90.0</td><td>1.0</td><td>15.0</td></tr>
</table>
"""

# A FFToday QB row: [Chg, Player, Team, Opp, Comp, Att, pYd, pTD, INT, rAtt, rYd, rTD, FPts]
_QB_PAGE = """
<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td><td>Team</td><td>Opp</td>
<td>Comp</td><td>Att</td><td>Yard</td><td>TD</td><td>INT</td><td>Att</td><td>Yard</td><td>TD</td><td>FPts</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/2515/Aaron_Rodgers?LeagueID=">Aaron Rodgers</a></td>
<td>GB</td><td>SF</td><td>24.0</td><td>37.0</td><td>290.0</td><td>2.0</td><td>0.0</td><td>4.0</td><td>18.0</td><td>0.0</td><td>24.5</td></tr>
</table>
"""

_EMPTY_PAGE = "<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td></tr></table>"


def _fake_reader(url: str) -> str:
    """Return crafted HTML keyed by the PosID in the URL (offense: QB=10, WR=30)."""
    posid = int(url.split("PosID=")[1].split("&")[0])
    if posid == FFTODAY_POS_IDS["WR"]:
        return _WR_PAGE
    if posid == FFTODAY_POS_IDS["QB"]:
        return _QB_PAGE
    return _EMPTY_PAGE  # RB/TE -> parsed empty -> skipped


# ---------- parsing --------------------------------------------------------------


def test_parse_wr_row_maps_receiving_columns():
    df = _parse_projection_html(_WR_PAGE, "WR")
    cj = df[df["player_name"] == "Calvin Johnson"].iloc[0]
    assert cj["team"] == "DET"
    assert cj["opponent"] == "MIN"
    assert cj["receptions"] == 8.0
    assert cj["receiving_yards"] == 110.0
    assert cj["receiving_tds"] == 1.0
    assert cj["fftoday_projected_pts"] == 17.0
    # WR carries no passing stats -> 0-filled, incl. the always-0 fumbles_lost.
    assert cj["passing_yards"] == 0.0
    assert cj["fumbles_lost"] == 0.0
    assert cj["fftoday_player_id"] == "2753"


def test_parse_qb_row_maps_passing_and_rushing_columns():
    df = _parse_projection_html(_QB_PAGE, "QB")
    ar = df[df["player_name"] == "Aaron Rodgers"].iloc[0]
    assert ar["team"] == "GB"  # correct historical team (not FantasyPros' current team)
    assert ar["passing_yards"] == 290.0
    assert ar["passing_tds"] == 2.0
    assert ar["interceptions"] == 0.0
    assert ar["rushing_yards"] == 18.0
    assert ar["rushing_tds"] == 0.0


def test_parse_skips_rows_without_a_player_anchor():
    # Header/spacer rows (no /stats/players anchor) must not become data rows.
    assert len(_parse_projection_html(_EMPTY_PAGE, "WR")) == 0


# ---------- load + cache ---------------------------------------------------------


def test_load_projections_injected_reader(tmp_path):
    df = load_fftoday_projections([2013], weeks=(1,), cache_dir=str(tmp_path), reader=_fake_reader)
    assert set(df["position"]) == {"QB", "WR"}  # RB/TE returned empty pages -> skipped
    assert {"player_name", "team", "season", "week", "position", "receptions"} <= set(df.columns)
    assert (df["season"] == 2013).all()
    # A second call hits the parquet cache (no reader needed).
    cached = load_fftoday_projections(
        [2013],
        weeks=(1,),
        cache_dir=str(tmp_path),
        reader=lambda u: (_ for _ in ()).throw(AssertionError),
    )
    assert len(cached) == len(df)


def test_min_season_guard():
    with pytest.raises(ValueError, match="archive starts at"):
        load_fftoday_projections([2009], weeks=(1,), reader=_fake_reader)


# ---------- gsis-id bridge -------------------------------------------------------


def _rosters():
    return pd.DataFrame(
        [
            {
                "player_name": "Calvin Johnson",
                "season": 2013,
                "team": "DET",
                "position": "WR",
                "player_id": "00-0026035",
            },
            {
                "player_name": "Aaron Rodgers",
                "season": 2013,
                "team": "GB",
                "position": "QB",
                "player_id": "00-0023459",
            },
            # Wrong team on purpose -> exercises the (name, season, position) single-id fallback.
            {
                "player_name": "Jimmy Graham",
                "season": 2013,
                "team": "XXX",
                "position": "WR",
                "player_id": "00-0027686",
            },
        ]
    )


def test_bridge_attaches_gsis_primary_and_fallback(tmp_path):
    joined = load_fftoday_with_gsis_id(
        [2013],
        cache_dir=str(tmp_path),
        rosters=_rosters(),
        min_match_rate=0.0,
        reader=_fake_reader,
    )
    by_name = joined.set_index("player_name")["player_id"].to_dict()
    assert by_name["Calvin Johnson"] == "00-0026035"  # primary (name+season+team+pos)
    assert by_name["Aaron Rodgers"] == "00-0023459"
    assert by_name["Jimmy Graham"] == "00-0027686"  # fallback despite wrong roster team
    # Only matched rows survive.
    assert joined["player_id"].notna().all()


def test_bridge_raises_below_min_match_rate(tmp_path):
    empty_rosters = pd.DataFrame(
        columns=["player_name", "season", "team", "position", "player_id"]
    ).astype({"season": int})
    with pytest.raises(RuntimeError, match="match rate"):
        load_fftoday_with_gsis_id(
            [2013],
            cache_dir=str(tmp_path),
            rosters=empty_rosters,
            min_match_rate=0.90,
            reader=_fake_reader,
        )


# ---------- expert registration --------------------------------------------------


def test_fftoday_registered_as_expert():
    experts = {e.name: e for e in aec._build_experts(nflcom_loader=None, sleeper_loader=None)}
    assert "fftoday" in experts
    ff = experts["fftoday"]
    assert ff.label == "FFToday"
    assert ff.skipped == frozenset({"K", "DST"})
    # Reuses the generic raw-stat -> PPR projector.
    assert ff.project is aec._project_sleeper_to_ppr
