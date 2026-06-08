"""Unit tests for the ESPN live-data parsers (src/serving/espn_live.py).

Pure-parser tests over fixture payloads shaped like the real ESPN responses —
no network. The key invariants: the spread-sign flip to nflverse convention,
team-code normalization, the espn_id-from-href extraction, and the active-roster
filtering.
"""

import pytest

from src.serving import espn_live


@pytest.mark.unit
def test_espn_to_nflverse_team_maps_only_the_two_outliers():
    assert espn_live.espn_to_nflverse_team("LAR") == "LA"
    assert espn_live.espn_to_nflverse_team("WSH") == "WAS"
    # The other 30 pass through unchanged.
    for code in ("KC", "SF", "NE", "JAX", "LAC", "LV", "GB"):
        assert espn_live.espn_to_nflverse_team(code) == code
    assert espn_live.espn_to_nflverse_team(None) is None


@pytest.mark.unit
def test_norm_espn_id_strips_float_suffix():
    assert espn_live._norm_espn_id("4678006") == "4678006"
    assert espn_live._norm_espn_id(4678006) == "4678006"
    assert espn_live._norm_espn_id(4678006.0) == "4678006"
    assert espn_live._norm_espn_id("nan") is None
    assert espn_live._norm_espn_id(None) is None
    assert espn_live._norm_espn_id("") is None


@pytest.mark.unit
def test_extract_espn_id_from_headshot_then_links():
    assert (
        espn_live._extract_espn_id(
            {"headshot": {"href": "https://a.espncdn.com/i/headshots/nfl/players/full/2578570.png"}}
        )
        == "2578570"
    )
    # No headshot → fall back to the player link.
    assert (
        espn_live._extract_espn_id(
            {"links": [{"href": "https://www.espn.com/nfl/player/_/id/3139477/patrick-mahomes"}]}
        )
        == "3139477"
    )
    assert espn_live._extract_espn_id({}) is None


@pytest.mark.unit
def test_parse_scoreboard_flips_spread_to_home_perspective():
    # ESPN: negative spread = home favored. nflverse: positive = home favored.
    payload = {
        "season": {"year": 2026},
        "week": {"number": 1},
        "events": [
            {
                "id": "401",
                "competitions": [
                    {
                        "status": {"type": {"name": "STATUS_SCHEDULED"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "SEA", "id": "26"}},
                            {"homeAway": "away", "team": {"abbreviation": "NE", "id": "17"}},
                        ],
                        "odds": [{"spread": -3.5, "overUnder": 44.5}],
                    }
                ],
            },
            {
                "id": "402",
                "competitions": [
                    {
                        "status": {"type": {"name": "STATUS_SCHEDULED"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "IND", "id": "11"}},
                            {"homeAway": "away", "team": {"abbreviation": "BAL", "id": "33"}},
                        ],
                        # away (BAL) favored → ESPN reports +3.5 (home perspective).
                        "odds": [{"spread": 3.5, "overUnder": 49.5}],
                    }
                ],
            },
        ],
    }
    games = espn_live._parse_scoreboard_games(payload)
    assert len(games) == 2
    sea = games[0]
    assert sea["home_team"] == "SEA" and sea["away_team"] == "NE"
    assert sea["is_scheduled"] is True
    # SEA home-favored: ESPN -3.5 → nflverse +3.5.
    assert sea["spread_line"] == 3.5
    assert sea["total_line"] == 44.5
    ind = games[1]
    # IND home-underdog: ESPN +3.5 → nflverse -3.5.
    assert ind["spread_line"] == -3.5


@pytest.mark.unit
def test_parse_scoreboard_handles_missing_odds():
    payload = {
        "season": {"year": 2026},
        "week": {"number": 1},
        "events": [
            {
                "id": "1",
                "competitions": [
                    {
                        "status": {"type": {"name": "STATUS_SCHEDULED"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "GB", "id": "9"}},
                            {"homeAway": "away", "team": {"abbreviation": "MIN", "id": "16"}},
                        ],
                        "odds": [],
                    }
                ],
            }
        ],
    }
    games = espn_live._parse_scoreboard_games(payload)
    assert games[0]["spread_line"] is None
    assert games[0]["total_line"] is None


@pytest.mark.unit
def test_parse_roster_skips_inactive_groups_and_nonskill():
    payload = {
        "athletes": [
            {
                "position": "offense",
                "items": [
                    {"id": "1", "displayName": "Star WR", "position": {"abbreviation": "WR"}},
                    {"id": "2", "displayName": "Center", "position": {"abbreviation": "C"}},
                ],
            },
            {
                "position": "injuredReserveOrOut",
                "items": [
                    {"id": "3", "displayName": "Hurt RB", "position": {"abbreviation": "RB"}}
                ],
            },
            {
                "position": "practiceSquad",
                "items": [{"id": "4", "displayName": "PS TE", "position": {"abbreviation": "TE"}}],
            },
        ]
    }
    players = espn_live._parse_roster_players(payload, team_code="SEA")
    ids = {p["espn_id"] for p in players}
    assert ids == {"1"}  # only the active skill player; C / IR / PS excluded
    assert players[0]["position"] == "WR"
    assert players[0]["recent_team"] == "SEA"


@pytest.mark.unit
def test_parse_injuries_extracts_id_position_team():
    payload = {
        "injuries": [
            {
                "displayName": "Arizona Cardinals",
                "injuries": [
                    {
                        "status": "Out",
                        "athlete": {
                            "displayName": "Some RB",
                            "position": {"abbreviation": "RB"},
                            "team": {"abbreviation": "WSH"},
                            "headshot": {
                                "href": "https://a.espncdn.com/i/headshots/nfl/players/full/12345.png"
                            },
                        },
                    }
                ],
            }
        ]
    }
    recs = espn_live._parse_injuries(payload)
    assert len(recs) == 1
    r = recs[0]
    assert r["espn_id"] == "12345"
    assert r["position"] == "RB"
    assert r["team"] == "WAS"  # WSH → WAS
    assert r["status"] == "Out"


@pytest.mark.unit
def test_injury_status_map_only_out_doubtful():
    m = espn_live._INJURY_STATUS_MAP
    assert m["out"] == "Out"
    assert m["injured reserve"] == "Out"
    assert m["doubtful"] == "Doubtful"
    assert "questionable" not in m
    assert "active" not in m
