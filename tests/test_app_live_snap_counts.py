"""Actual Sleeper observations must bind to existing completed NFL player-games."""

import copy
import hashlib
import io
import json
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

from src.data import live_snap_counts as snaps

pytestmark = pytest.mark.unit


def _source(sid="10", *, count=31, total=61, team="SEA", opponent="NE", **changes):
    row = {
        "category": "stat",
        "sport": "nfl",
        "season": "2026",
        "week": 1,
        "season_type": "regular",
        "date": "2026-01-01",
        "game_id": "source-game-1",
        "player_id": sid,
        "team": team,
        "opponent": opponent,
        "company": "sportradar",
        "player": {"position": "WR", "team": team},
        "stats": {"off_snp": count, "tm_off_snp": total, "gp": 1},
    }
    row.update(changes)
    return row


def _response(payload, url="https://api.sleeper.com/stats/nfl/2026/1?season_type=regular"):
    return copy.deepcopy(payload), {
        "url": url,
        "retrieved_at": "2026-01-02T12:00:00+00:00",
        "sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
    }


@pytest.fixture
def inputs(monkeypatch):
    current = pd.DataFrame(
        [
            {
                "player_id": "g1",
                "season": 2026,
                "week": 1,
                "season_type": "REG",
                "position": "WR",
                "recent_team": "SEA",
                "opponent_team": "NE",
                "game_id": "2026_01_NE_SEA",
                "snap_pct": np.nan,
                "targets": 4,
            }
        ],
        index=pd.Index([71], name="original_index"),
    )
    current.attrs["source"] = {"preserve": ["native"]}
    schedules = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-01-01",
                "game_id": "2026_01_NE_SEA",
                "home_team": "SEA",
                "away_team": "NE",
                "home_score": 13,
                "away_score": 10,
            }
        ],
    )
    monkeypatch.setattr(
        snaps.nfl_source,
        "player_ids",
        lambda: pd.DataFrame({"sleeper_id": ["10", "11"], "gsis_id": ["g1", "g2"]}),
    )
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source()], url))
    return current, schedules


def test_fills_only_missing_existing_row_and_preserves_frame_identity(inputs):
    current, schedules = inputs
    before = current.copy(deep=True)
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(current, before)
    assert result.index.equals(current.index)
    assert result.columns.equals(current.columns)
    assert result.attrs == current.attrs
    assert result.at[71, "snap_pct"] == 0.51
    assert metadata["eligible_rows"] == metadata["observed_rows"] == 1
    assert metadata["recovered_rows"] == metadata["recovered_cells"] == 1
    assert metadata["requested_weeks"] == [1]
    assert metadata["status"] == "available"
    assert metadata["validated_games"] == ["2026_01_NE_SEA"]
    assert metadata["validated_source_game_ids"] == ["source-game-1"]
    assert metadata["companies"] == ["sportradar"]
    assert metadata["retrieved_at"] == "2026-01-02T12:00:00+00:00"
    assert len(metadata["receipts"][0]["sha256"]) == 64
    assert "off_snp" not in json.dumps(metadata)


@pytest.mark.parametrize("native", [0.0, 0.82])
def test_native_observations_never_query_or_get_overwritten(inputs, monkeypatch, native):
    current, schedules = inputs
    current["snap_pct"] = native
    monkeypatch.setattr(snaps, "_get_json", lambda url: pytest.fail("unneeded HTTP"))
    monkeypatch.setattr(
        snaps.nfl_source, "player_ids", lambda: pytest.fail("unneeded identity fetch")
    )
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["requested_weeks"] == []
    assert metadata["recovered_rows"] == 0
    assert metadata["status"] == "available"


def test_only_requests_weeks_with_missing_completed_skill_rows(inputs, monkeypatch):
    current, schedules = inputs
    second = current.iloc[0].to_dict() | {
        "player_id": "g2",
        "week": 2,
        "snap_pct": 0.5,
        "game_id": "2026_02_NE_SEA",
    }
    current = pd.concat([current, pd.DataFrame([second])])
    schedules = pd.concat([schedules, schedules.assign(week=2, game_id="2026_02_NE_SEA")])
    seen = []

    def fetch(url):
        seen.append(url)
        return _response([_source()], url)

    monkeypatch.setattr(snaps, "_get_json", fetch)
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert len(seen) == 1 and "/2026/1?season_type=regular" in seen[0]
    assert metadata["requested_weeks"] == [1]
    assert result.snap_pct.tolist() == [0.51, 0.5]
    assert result.index.tolist() == current.index.tolist()


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_all_existing_skill_positions_can_recover(inputs, monkeypatch, position):
    current, schedules = inputs
    current["position"] = position
    source = _source(player={"position": position})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.51


@pytest.mark.parametrize("count,expected", [(42, 0.52), (38, 0.48), (2, 0.02), (14, 0.18)])
def test_native_pfr_half_ties_use_exact_decimal_even_rounding(inputs, monkeypatch, count, expected):
    current, schedules = inputs
    current["targets"] = 0
    monkeypatch.setattr(
        snaps, "_get_json", lambda url: _response([_source(count=count, total=80)], url)
    )
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == expected
    assert metadata["recovered_rows"] == 1


@pytest.mark.parametrize("position", ["K", "DST", "CB", "FB", None])
def test_no_extra_positions_or_source_only_players_are_added(inputs, monkeypatch, position):
    current, schedules = inputs
    current["position"] = position
    monkeypatch.setattr(snaps, "_get_json", lambda url: pytest.fail("unneeded HTTP"))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["status"] == "not_required"


def test_unmatched_source_player_does_not_expand_existing_population(inputs, monkeypatch):
    current, schedules = inputs
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source("11")], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["recovered_rows"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "projection"),
        ("sport", "nba"),
        ("season", "2025"),
        ("season", True),
        ("week", 2),
        ("week", True),
        ("season_type", "post"),
        ("date", "2026-01-02"),
        ("date", None),
        ("date", "2026-01-01T00:00:00Z"),
        ("team", "NE"),
        ("opponent", "BUF"),
        ("game_id", ""),
        ("game_id", None),
        ("game_id", True),
        ("player_id", None),
        ("player_id", True),
        ("player", None),
        ("stats", None),
    ],
)
def test_source_discriminator_and_game_mismatches_remain_missing(inputs, monkeypatch, field, value):
    current, schedules = inputs
    source = _source(**{field: value})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["recovered_rows"] == 0


@pytest.mark.parametrize("field", ["off_snp", "tm_off_snp", "gp"])
@pytest.mark.parametrize("value", [None, True, False, "1", float("nan"), float("inf"), 1.25])
def test_snap_observation_requires_explicit_finite_integer_numbers(
    inputs, monkeypatch, field, value
):
    current, schedules = inputs
    source = _source()
    source["stats"][field] = value
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["rejected_rows"] == 1


@pytest.mark.parametrize("field", ["off_snp", "tm_off_snp", "gp"])
def test_absent_count_or_played_flag_is_not_observed_zero(inputs, monkeypatch, field):
    current, schedules = inputs
    source = _source()
    del source["stats"][field]
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["rejected_rows"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"off_snp": -1},
        {"tm_off_snp": 0},
        {"tm_off_snp": -1},
        {"off_snp": 62},
        {"gp": 0},
        {"gp": 2},
    ],
)
def test_impossible_counts_or_unplayed_row_rejected(inputs, monkeypatch, changes):
    current, schedules = inputs
    source = _source()
    source["stats"].update(changes)
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


def test_explicit_zero_needs_verified_played_identity_and_no_offensive_usage(inputs, monkeypatch):
    current, schedules = inputs
    current["targets"] = 0
    current["special_teams_tds"] = 1
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(count=0)], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.0
    assert metadata["recovered_rows"] == 1


@pytest.mark.parametrize(
    "side,key",
    [
        ("current", "targets"),
        ("current", "attempts"),
        ("current", "carries"),
        ("source", "rec_tgt"),
        ("source", "pass_att"),
        ("source", "rush_att"),
    ],
)
def test_zero_conflicting_with_offensive_usage_is_not_accepted(inputs, monkeypatch, side, key):
    current, schedules = inputs
    current["targets"] = 0
    source = _source(count=0)
    if side == "current":
        current[key] = 1
    else:
        source["stats"][key] = 1
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


@pytest.mark.parametrize(
    "side,key",
    [
        ("current", "attempts"),
        ("current", "carries"),
        ("current", "targets"),
        ("source", "pass_att"),
        ("source", "rush_att"),
        ("source", "rec_tgt"),
    ],
)
def test_nonzero_snaps_cannot_be_less_than_a_recorded_usage_counter(inputs, monkeypatch, side, key):
    current, schedules = inputs
    source = _source(count=1)
    if side == "current":
        current[key] = 30
    else:
        source["stats"][key] = 30
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["rejected_rows"] == 1


def test_usage_floor_uses_maximum_rather_than_sum(inputs, monkeypatch):
    current, schedules = inputs
    current["carries"] = current["targets"] = 10
    source = _source(count=10, total=100)
    source["stats"].update({"rush_att": 10, "rec_tgt": 10})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.1


def test_extreme_usage_counter_fails_closed_without_escaping_optional_boundary(inputs, monkeypatch):
    current, schedules = inputs
    source = _source()
    source["stats"]["rush_att"] = 10**400
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


@pytest.mark.parametrize(
    "pairs",
    [
        [("10", "g1"), ("10", "g2")],
        [("10", "g1"), ("11", "g1")],
        [(None, "g1")],
        [("10", None)],
        [(True, "g1")],
    ],
)
def test_identity_bridge_must_be_unique_in_both_directions(inputs, monkeypatch, pairs):
    current, schedules = inputs
    monkeypatch.setattr(
        snaps.nfl_source,
        "player_ids",
        lambda: pd.DataFrame(pairs, columns=["sleeper_id", "gsis_id"]),
    )
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["recovered_rows"] == 0


def test_identical_crosswalk_and_source_duplicates_do_not_fan_out(inputs, monkeypatch):
    current, schedules = inputs
    monkeypatch.setattr(
        snaps.nfl_source,
        "player_ids",
        lambda: pd.DataFrame({"sleeper_id": [10.0, "10"], "gsis_id": ["g1", "g1"]}),
    )
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(), _source()], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert len(result) == 1 and result.snap_pct.iloc[0] == 0.51
    assert metadata["recovered_rows"] == 1


@pytest.mark.parametrize("changes", [{"off_snp": 30}, {"tm_off_snp": 62}])
def test_conflicting_duplicate_observations_are_rejected(inputs, monkeypatch, changes):
    current, schedules = inputs
    duplicate = _source()
    duplicate["stats"].update(changes)
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(), duplicate], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["rejected_rows"] == 2


def test_invalid_duplicate_cannot_leave_a_valid_looking_winner(inputs, monkeypatch):
    current, schedules = inputs
    duplicate = _source(count=1.25)
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(), duplicate], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


def test_team_denominator_must_agree_across_matched_players(inputs, monkeypatch):
    current, schedules = inputs
    current = pd.concat([current, current.assign(player_id="g2")])
    monkeypatch.setattr(
        snaps, "_get_json", lambda url: _response([_source(), _source("11", total=62)], url)
    )
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["rejected_rows"] == 2


@pytest.mark.parametrize("conflict", ["denominator", "source_game"])
def test_unmapped_unmodeled_actual_records_still_check_game_integrity(
    inputs, monkeypatch, conflict
):
    current, schedules = inputs
    lineman = _source("unmapped-ol", count=61, player={"position": "OL"})
    if conflict == "denominator":
        lineman["stats"].update({"off_snp": 62, "tm_off_snp": 62})
    else:
        lineman["game_id"] = "conflicting-source-game"
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(), lineman], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["recovered_rows"] == 0
    assert metadata["rejected_rows"] == 2


def test_valid_unmapped_actual_lineman_can_confirm_without_expanding_population(
    inputs, monkeypatch
):
    current, schedules = inputs
    lineman = _source("unmapped-ol", count=61, player={"position": "OL"})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(), lineman], url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert len(result) == 1 and result.snap_pct.iloc[0] == 0.51
    assert metadata["recovered_rows"] == 1


def test_one_nfl_matchup_cannot_have_two_source_game_ids(inputs, monkeypatch):
    current, schedules = inputs
    current = pd.concat([current, current.assign(player_id="g2")])
    monkeypatch.setattr(
        snaps,
        "_get_json",
        lambda url: _response([_source(), _source("11", game_id="different-source-game")], url),
    )
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


def test_one_source_game_cannot_bind_two_nfl_matchups(inputs, monkeypatch):
    current, schedules = inputs
    current = pd.concat(
        [
            current,
            current.assign(
                player_id="g2", recent_team="BUF", opponent_team="MIA", game_id="2026_01_MIA_BUF"
            ),
        ]
    )
    schedules = pd.concat(
        [schedules, schedules.assign(home_team="BUF", away_team="MIA", game_id="2026_01_MIA_BUF")]
    )
    monkeypatch.setattr(
        snaps,
        "_get_json",
        lambda url: _response([_source(), _source("11", team="BUF", opponent="MIA")], url),
    )
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


@pytest.mark.parametrize(
    "schedule_changes",
    [
        {"home_score": None},
        {"away_score": np.nan},
        {"home_score": True},
        {"game_type": "POST"},
        {"gameday": "2099-01-01"},
        {"season": 2025},
    ],
)
def test_unplayed_future_other_season_and_postseason_never_fetch(
    inputs, monkeypatch, schedule_changes
):
    current, schedules = inputs
    for key, value in schedule_changes.items():
        schedules[key] = value
    monkeypatch.setattr(snaps, "_get_json", lambda url: pytest.fail("ineligible HTTP"))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    assert metadata["requested_weeks"] == []


def test_historical_replay_does_not_query_current_sources(inputs, monkeypatch):
    current, schedules = inputs
    monkeypatch.setattr(snaps, "_get_json", lambda url: pytest.fail("archive HTTP"))
    monkeypatch.setattr(
        snaps.nfl_source, "player_ids", lambda: pytest.fail("archive identity HTTP")
    )
    result, metadata = snaps.recover_snap_counts(current, schedules, max(snaps.SEASONS))
    pd.testing.assert_frame_equal(result, current)
    assert metadata["reason"] == "historical_replay"


def test_nullable_current_period_and_schedule_fields_remain_safe(inputs, monkeypatch):
    current, schedules = inputs
    current["season_type"] = pd.Series([pd.NA], index=current.index, dtype="string")
    monkeypatch.setattr(snaps, "_get_json", lambda url: pytest.fail("ineligible HTTP"))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    current["season_type"] = "REG"
    schedules["game_type"] = pd.NA
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


def test_missing_required_schema_and_absent_snap_column_fail_safely(inputs, monkeypatch):
    current, schedules = inputs
    for frame, calendar in [
        (current.drop(columns="opponent_team"), schedules),
        (current, schedules.drop(columns="game_id")),
    ]:
        result, metadata = snaps.recover_snap_counts(frame, calendar, 2026)
        pd.testing.assert_frame_equal(result, frame)
        assert metadata["status"] == "unavailable" and metadata["errors"]
    result, metadata = snaps.recover_snap_counts(current.drop(columns="snap_pct"), schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.51 and metadata["recovered_rows"] == 1


def test_conflicting_schedule_and_current_game_ids_do_not_fill(inputs, monkeypatch):
    current, schedules = inputs
    current["game_id"] = "wrong-game"
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()
    current["game_id"] = "2026_01_NE_SEA"
    schedules = pd.concat([schedules, schedules.assign(game_id="conflicting-game")])
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.isna().all()


def test_duplicate_current_player_week_is_not_arbitrarily_resolved(inputs):
    current, schedules = inputs
    current = pd.concat([current, current])
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)


@pytest.mark.parametrize(
    "source_position,roles,expected",
    [
        ("FB", None, True),
        ("DB", ["DB", "RB"], True),
        ("DB", ["DB", "WR"], False),
        ("DB", None, False),
        (None, ["RB"], False),
    ],
)
def test_source_roles_confirm_existing_position_without_retargeting_identity(
    inputs, monkeypatch, source_position, roles, expected
):
    current, schedules = inputs
    current["position"] = "RB"
    source = _source(player={"position": source_position, "fantasy_positions": roles, "team": None})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert pd.notna(result.snap_pct.iloc[0]) == expected
    assert result.position.tolist() == ["RB"]


def test_captured_hunter_dual_role_and_mutable_nested_team_are_valid(inputs, monkeypatch):
    current, schedules = inputs
    source = _source(player={"position": "DB", "fantasy_positions": ["DB", "WR"], "team": "BUF"})
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([source], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.51
    assert result.position.iloc[0] == "WR"


def test_repo_team_aliases_match_existing_game_without_changing_frame(inputs, monkeypatch):
    current, schedules = inputs
    current["recent_team"] = "LA"
    current["game_id"] = "2026_01_NE_LA"
    schedules["home_team"] = "LA"
    schedules["game_id"] = "2026_01_NE_LA"
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response([_source(team="LAR")], url))
    result, _ = snaps.recover_snap_counts(current, schedules, 2026)
    assert result.snap_pct.iloc[0] == 0.51
    assert result.recent_team.iloc[0] == "LA"


@pytest.mark.parametrize("payload", [{}, [None], ["bad"], "bad"])
def test_malformed_payload_leaves_existing_defaults_and_receipt(inputs, monkeypatch, payload):
    current, schedules = inputs
    monkeypatch.setattr(snaps, "_get_json", lambda url: _response(payload, url))
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["status"] == "unavailable"
    assert metadata["errors"] and metadata["receipts"]


def test_optional_fetch_outage_and_bad_crosswalk_leave_nan(inputs, monkeypatch):
    current, schedules = inputs

    def unavailable(url):
        raise URLError("temporary source outage")

    monkeypatch.setattr(snaps, "_get_json", unavailable)
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["status"] == "unavailable" and metadata["errors"]
    monkeypatch.setattr(snaps.nfl_source, "player_ids", lambda: pd.DataFrame())
    result, metadata = snaps.recover_snap_counts(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["requested_weeks"] == []
    assert metadata["errors"]


class _HTTPResponse(io.BytesIO):
    def __init__(
        self, body=b"[]", url="https://api.sleeper.com/stats/nfl/2026/1?season_type=regular"
    ):
        super().__init__(body)
        self.url = url

    def geturl(self):
        return self.url


def _opener(monkeypatch, responses):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(snaps, "build_opener", lambda *args: Opener())
    return seen


def test_http_helper_hashes_exact_bytes_and_uses_bounded_transport_retry(monkeypatch):
    body = b" [ ]\n"
    seen = _opener(monkeypatch, [URLError("retry"), _HTTPResponse(body)])
    payload, receipt = snaps._get_json(
        "https://api.sleeper.com/stats/nfl/2026/1?season_type=regular"
    )
    assert payload == []
    assert receipt["sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["bytes"] == len(body) and receipt["attempts"] == 2
    assert len(seen) == 2 and all(0 < timeout <= 15 for _, timeout in seen)
    assert all(request.get_header("User-agent") == "Final-Project/1.0" for request, _ in seen)
    assert receipt["user_agent"] == "Final-Project/1.0"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.sleeper.com/stats",
        "https://example.com/stats",
        "https://api.sleeper.com.evil.test/stats",
        "https://user@api.sleeper.com/stats",
    ],
)
def test_wrong_source_host_and_downgrade_are_rejected_before_request(monkeypatch, url):
    monkeypatch.setattr(snaps, "build_opener", lambda *args: pytest.fail("unsafe request"))
    with pytest.raises(ValueError, match="HTTPS"):
        snaps._get_json(url)


def test_http_final_response_host_is_checked_and_forbidden_is_not_retried(monkeypatch):
    seen = _opener(monkeypatch, [_HTTPResponse(url="http://api.sleeper.com/stats")])
    with pytest.raises(ValueError, match="HTTPS"):
        snaps._get_json("https://api.sleeper.com/stats")
    assert len(seen) == 1


def test_redirect_handler_does_not_follow_unsafe_redirects():
    from urllib.request import Request

    with pytest.raises(ValueError, match="HTTPS"):
        snaps._SafeRedirect().redirect_request(
            Request("https://api.sleeper.com/stats"),
            None,
            302,
            "Found",
            {},
            "https://example.com/elsewhere",
        )


def test_transient_http_failure_stops_after_two_attempts(monkeypatch):
    failures = [
        HTTPError("https://api.sleeper.com/stats", 503, "Unavailable", {}, None) for _ in range(2)
    ]
    seen = _opener(monkeypatch, failures)
    with pytest.raises(HTTPError):
        snaps._get_json("https://api.sleeper.com/stats")
    assert len(seen) == 2
    seen = _opener(
        monkeypatch, [HTTPError("https://api.sleeper.com/stats", 403, "Forbidden", {}, None)]
    )
    with pytest.raises(HTTPError):
        snaps._get_json("https://api.sleeper.com/stats")
    assert len(seen) == 1


@pytest.mark.parametrize("body", [b"not json", b'[{"stats":{},"stats":{}}]'])
def test_invalid_json_keeps_a_hashed_failure_receipt(monkeypatch, body):
    _opener(monkeypatch, [_HTTPResponse(body)])
    with pytest.raises(snaps._ResponseError) as exc:
        snaps._get_json("https://api.sleeper.com/stats")
    assert exc.value.receipt["sha256"] == hashlib.sha256(body).hexdigest()


def test_oversize_json_is_rejected_without_reading_unbounded_data(monkeypatch):
    monkeypatch.setattr(snaps, "_MAX_BYTES", 4)
    _opener(monkeypatch, [_HTTPResponse(b"[      ]")])
    with pytest.raises(snaps._ResponseError, match="size") as exc:
        snaps._get_json("https://api.sleeper.com/stats")
    assert exc.value.receipt["truncated"] is True
    assert exc.value.receipt["bytes"] == 5
