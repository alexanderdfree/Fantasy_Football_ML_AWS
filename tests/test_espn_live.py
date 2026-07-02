"""Unit tests for the ESPN live-data parsers (src/serving/espn_live.py).

Pure-parser tests over fixture payloads shaped like the real ESPN responses —
no network. The key invariants: the spread-sign flip to nflverse convention,
team-code normalization, the espn_id-from-href extraction, and the active-roster
filtering.
"""

import pandas as pd
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


@pytest.mark.unit
def test_parse_depthchart_reranks_by_order_and_pools_wr():
    # ESPN's raw `rank` is grid-numbered (WRs 1, 4, 7); we re-rank by sorted
    # order. Defense + fullback are dropped; split WR slots pool into one WR list.
    payload = {
        "items": [
            {"name": "Base D", "positions": {"lde": {"athletes": [_dc(1, "999")]}}},
            {
                "name": "3WR 1TE",
                "positions": {
                    "qb": {"athletes": [_dc(2, "101"), _dc(1, "100")]},
                    "wr": {"athletes": [_dc(7, "202"), _dc(1, "200"), _dc(4, "201")]},
                    "fb": {"athletes": [_dc(1, "300")]},
                },
            },
        ]
    }
    entries = espn_live._parse_depthchart(payload)
    by_id = {e["espn_id"]: e for e in entries}
    assert "999" not in by_id and "300" not in by_id  # defense + FB excluded
    assert (by_id["100"]["position"], by_id["100"]["order"]) == ("QB", 1)
    assert by_id["101"]["order"] == 2
    # WR re-ranked by raw rank 1<4<7 -> order 1,2,3.
    assert [by_id[i]["order"] for i in ("200", "201", "202")] == [1, 2, 3]


@pytest.mark.unit
def test_fetch_depth_chart_ranks_clamps_to_three_and_maps_gsis(monkeypatch):
    payload = {
        "items": [
            {
                "positions": {
                    "qb": {"athletes": [_dc(1, "100"), _dc(2, "101"), _dc(3, "102"), _dc(4, "103")]}
                }
            }
        ]
    }
    monkeypatch.setattr(espn_live, "_get_json", lambda url: payload)
    monkeypatch.setattr(
        espn_live,
        "espn_to_gsis_map",
        lambda: {"100": "00-A", "101": "00-B", "102": "00-C", "103": "00-D"},
    )
    ranks = espn_live.fetch_depth_chart_ranks(2026, {"33": "BAL"})
    # Order 1/2/3 map straight; the 4th-string clamps to the nflverse [1,3] cap.
    assert ranks == {"00-A": 1.0, "00-B": 2.0, "00-C": 3.0, "00-D": 3.0}


@pytest.mark.unit
def test_fetch_depth_chart_ranks_falls_back_to_prior_season(monkeypatch):
    # The requested season's chart isn't posted yet (empty) — the offseason case
    # that's live right now — so the per-team fallback uses the prior season.
    prior = {"items": [{"positions": {"qb": {"athletes": [_dc(1, "100")]}}}]}

    def fake_get_json(url):
        return {"items": []} if "/seasons/2026/" in url else prior

    monkeypatch.setattr(espn_live, "_get_json", fake_get_json)
    monkeypatch.setattr(espn_live, "espn_to_gsis_map", lambda: {"100": "00-A"})
    assert espn_live.fetch_depth_chart_ranks(2026, {"33": "BAL"}) == {"00-A": 1.0}


@pytest.mark.unit
def test_fetch_injury_status_map_encodes_and_omits_active(monkeypatch):
    payload = {
        "injuries": [
            {
                "injuries": [
                    {"status": "Questionable", "athlete": _inj_ath("200")},
                    {"status": "Out", "athlete": _inj_ath("201")},
                    {"status": "Active", "athlete": _inj_ath("202")},
                ]
            }
        ]
    }
    monkeypatch.setattr(espn_live, "_get_json", lambda url: payload)
    monkeypatch.setattr(
        espn_live, "espn_to_gsis_map", lambda: {"200": "00-Q", "201": "00-O", "202": "00-ACT"}
    )
    m = espn_live.fetch_injury_status_map(2026, 1)
    # Questionable->0.5, Out->0.0 (matching loader.py's status_map); Active omitted
    # so that row keeps build_features' healthy default.
    assert m == {"00-Q": 0.5, "00-O": 0.0}


def _dc(rank, athlete_id):
    """A depthchart athlete entry (rank + an athlete $ref carrying the id)."""
    return {"rank": rank, "athlete": {"$ref": f"http://core/athletes/{athlete_id}?lang=en"}}


def _inj_ath(espn_id):
    """An injury athlete object (id is pulled from the headshot href)."""
    return {
        "position": {"abbreviation": "WR"},
        "team": {"abbreviation": "BAL"},
        "headshot": {"href": f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"},
    }


@pytest.mark.unit
def test_fetch_slate_handles_missing_odds(monkeypatch):
    """#1400 regression: one odds-less game must not TypeError the whole slate."""
    games = [
        {
            "game_id": "2026_05_NE_SEA",
            "season": 2026,
            "week": 5,
            "home_team": "SEA",
            "away_team": "NE",
            "home_team_id": "26",
            "away_team_id": "17",
            "spread_line": 3.5,
            "total_line": 44.5,
            "is_scheduled": True,
        },
        {
            # Odds off-board: the parser deliberately emits None (see
            # test_parse_scoreboard_handles_missing_odds).
            "game_id": "2026_05_MIN_GB",
            "season": 2026,
            "week": 5,
            "home_team": "GB",
            "away_team": "MIN",
            "home_team_id": "9",
            "away_team_id": "16",
            "spread_line": None,
            "total_line": None,
            "is_scheduled": True,
        },
    ]
    monkeypatch.setattr(espn_live, "fetch_games", lambda season, week: games)

    team_rows, sched_rows = espn_live.fetch_slate(2026, 5)

    assert len(team_rows) == 4
    by_team = team_rows.set_index("recent_team")
    # Priced game: away side negated as before.
    assert float(by_team.loc["SEA", "spread_line"]) == 3.5
    assert float(by_team.loc["NE", "spread_line"]) == -3.5
    # Odds-less game: both sides keep the parser's None (NaN in the frame).
    assert pd.isna(by_team.loc["GB", "spread_line"])
    assert pd.isna(by_team.loc["MIN", "spread_line"])
    assert len(sched_rows) == 2
    assert pd.isna(sched_rows.set_index("game_id").loc["2026_05_MIN_GB", "spread_line"])
