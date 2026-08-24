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
def test_get_json_sends_no_custom_user_agent(monkeypatch):
    """2026-08 incident pin: ESPN's WAF 403s custom (``ff-predictor/1.0``) and
    spoofed-browser UAs from non-browser clients, but passes honest tool UAs —
    ``_get_json`` must NOT re-add a custom User-Agent (urllib's default goes out)."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _Resp()

    monkeypatch.setattr(espn_live.urllib.request, "urlopen", fake_urlopen)
    assert espn_live._get_json("https://example.com/x") == {}
    assert not captured["req"].has_header("User-agent")


@pytest.mark.unit
def test_fetch_games_default_degrades_but_raise_on_error_raises(monkeypatch):
    def boom(url):
        raise RuntimeError("ESPN GET failed after 3 tries (403)")

    monkeypatch.setattr(espn_live, "_get_json", boom)
    assert espn_live.fetch_games(2026, 1) == []  # legacy defensive default
    with pytest.raises(RuntimeError):
        espn_live.fetch_games(2026, 1, raise_on_error=True)


@pytest.mark.unit
def test_next_unplayed_week_outage_raises_not_offseason(monkeypatch):
    """The 2026-08 silent-stale pin: when every scoreboard probe errors (ESPN
    403-ing the CI runner), ``next_unplayed_week`` must raise
    ``EspnUnreachableError`` rather than return None ("verified offseason") —
    None kept CI green while serving froze on a 20-day-old artifact."""
    calls = []

    def boom(url):
        calls.append(url)
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(espn_live, "_get_json", boom)
    with pytest.raises(espn_live.EspnUnreachableError):
        espn_live.next_unplayed_week(2026)
    # Early abort: a hard outage must not grind through all 2x18 weekly probes.
    assert len(calls) == espn_live._CONSECUTIVE_FAILURE_ABORT


@pytest.mark.unit
def test_next_unplayed_week_verified_offseason_returns_none(monkeypatch):
    # Every probe SUCCEEDS and none has a scheduled game -> verified offseason.
    monkeypatch.setattr(espn_live, "fetch_games", lambda s, w, raise_on_error=False: [])
    assert espn_live.next_unplayed_week(2026, lookahead_seasons=0) is None


@pytest.mark.unit
def test_next_unplayed_week_tolerates_isolated_failures(monkeypatch):
    # Two isolated probe failures (below the consecutive-abort threshold), then
    # a played week, then a scheduled one -> blips don't mask the real answer.
    seq = iter(["err", "err", "played", "sched"])

    def fake(season, week, raise_on_error=False):
        kind = next(seq)
        if kind == "err":
            raise RuntimeError("blip")
        return [{"is_scheduled": kind == "sched"}]

    monkeypatch.setattr(espn_live, "fetch_games", fake)
    assert espn_live.next_unplayed_week(2026) == (2026, 4)


@pytest.mark.unit
def test_next_unplayed_week_partial_failures_without_week_raises(monkeypatch):
    # Failures interleaved with successful-but-empty probes (never hitting the
    # consecutive abort) and no scheduled week found: offseason can't be
    # verified, so returning None would be a guess -> raise.
    state = {"n": 0}

    def fake(season, week, raise_on_error=False):
        state["n"] += 1
        if state["n"] % 2:
            raise RuntimeError("blip")
        return []

    monkeypatch.setattr(espn_live, "fetch_games", fake)
    with pytest.raises(espn_live.EspnUnreachableError):
        espn_live.next_unplayed_week(2026, lookahead_seasons=0)


@pytest.mark.unit
def test_get_json_passes_extra_headers_without_adding_ua(monkeypatch):
    """The fantasy API needs X-Fantasy-Filter; extra headers must pass through
    while the no-custom-User-Agent contract still holds."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _Resp()

    monkeypatch.setattr(espn_live.urllib.request, "urlopen", fake_urlopen)
    espn_live._get_json("https://example.com/x", headers={"X-Fantasy-Filter": '{"a":1}'})
    assert captured["req"].get_header("X-fantasy-filter") == '{"a":1}'
    assert not captured["req"].has_header("User-agent")


def _fantasy_stat(season, week, source, split, total, stats=None):
    return {
        "seasonId": season,
        "scoringPeriodId": week,
        "statSourceId": source,
        "statSplitTypeId": split,
        "appliedTotal": total,
        "stats": stats or {},
    }


def _fantasy_player(espn_id, pos_id, stats):
    return {"player": {"id": espn_id, "defaultPositionId": pos_id, "stats": stats}}


@pytest.mark.unit
def test_parse_fantasy_projections_selects_week_and_drops_placeholders():
    payload = {
        "players": [
            # QB with a genuine W1 projection (no receptions stat -> 0.0).
            _fantasy_player(100, 1, [_fantasy_stat(2026, 1, 1, 1, 19.23)]),
            # WR with a projection + receptions (id 53).
            _fantasy_player(200, 3, [_fantasy_stat(2026, 1, 1, 1, 19.97, {"53": 7.0087})]),
            # Only season-split (splitTypeId 0) + actuals (sourceId 0) -> skipped.
            _fantasy_player(
                300, 2, [_fantasy_stat(2026, 0, 1, 0, 250.0), _fantasy_stat(2026, 1, 0, 1, 12.0)]
            ),
            # 0.00 deep-bench placeholder -> "no genuine projection", dropped.
            _fantasy_player(400, 2, [_fantasy_stat(2026, 1, 1, 1, 0.0)]),
            # Defense (posId 16) -> out of scope.
            _fantasy_player(500, 16, [_fantasy_stat(2026, 1, 1, 1, 8.0)]),
            # Prior-season W1 entry only -> skipped.
            _fantasy_player(600, 3, [_fantasy_stat(2025, 1, 1, 1, 15.0)]),
        ]
    }
    recs = espn_live._parse_fantasy_projections(payload, 2026, 1)
    by_id = {r["espn_id"]: r for r in recs}
    assert set(by_id) == {"100", "200"}
    # Malformed receptions value degrades to 0.0 rather than raising.
    garbled = {
        "players": [_fantasy_player(700, 3, [_fantasy_stat(2026, 1, 1, 1, 9.0, {"53": "n/a"})])]
    }
    (rec,) = espn_live._parse_fantasy_projections(garbled, 2026, 1)
    assert rec["receptions"] == 0.0
    assert by_id["100"]["position"] == "QB"
    assert by_id["100"]["ppr_total"] == 19.23
    assert by_id["100"]["receptions"] == 0.0
    assert by_id["200"]["position"] == "WR"
    assert by_id["200"]["receptions"] == pytest.approx(7.0087)


@pytest.mark.unit
def test_fetch_fantasy_projections_maps_gsis_and_shapes_frame(monkeypatch):
    payload = {
        "players": [
            _fantasy_player(100, 1, [_fantasy_stat(2026, 1, 1, 1, 19.23)]),
            _fantasy_player(200, 3, [_fantasy_stat(2026, 1, 1, 1, 19.97, {"53": 7.0})]),
            # No crosswalk entry -> dropped with a logged count.
            _fantasy_player(999, 3, [_fantasy_stat(2026, 1, 1, 1, 9.0)]),
        ]
    }
    captured = {}

    def fake_get_json(url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return payload

    monkeypatch.setattr(espn_live, "_get_json", fake_get_json)
    monkeypatch.setattr(espn_live, "espn_to_gsis_map", lambda: {"100": "00-QB", "200": "00-WR"})
    df = espn_live.fetch_fantasy_projections(2026, 1)
    assert "leaguedefaults/3" in captured["url"] and "scoringPeriodId=1" in captured["url"]
    assert "X-Fantasy-Filter" in captured["headers"]
    assert sorted(df["player_id"]) == ["00-QB", "00-WR"]
    row = df.set_index("player_id").loc["00-WR"]
    assert (row["season"], row["week"]) == (2026, 1)
    assert row["espn_ppr_total"] == pytest.approx(19.97)
    assert row["espn_receptions"] == pytest.approx(7.0)


@pytest.mark.unit
def test_fetch_fantasy_projections_empty_on_failure(monkeypatch):
    def boom(url, headers=None):
        raise RuntimeError("fantasy API down")

    monkeypatch.setattr(espn_live, "_get_json", boom)
    assert espn_live.fetch_fantasy_projections(2026, 1).empty


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
