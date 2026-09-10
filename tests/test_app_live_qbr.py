"""Live QBR recovery uses the archive's exact upstream statistics and identities."""

import copy
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from src.serving import live_qbr

pytestmark = pytest.mark.unit


def _payload(week=1, game_id="101", player_id="10"):
    return {
        "currentValues": {"season": 2026, "week": week, "seasontype": 2, "qbrType": "weeks"},
        "pagination": {"pages": 1},
        "categories": [{"name": "general", "names": ["schedAdjQBR", "qbpaa", "qbr"]}],
        "athletes": [
            {
                "athlete": {"id": player_id},
                "game": {"id": game_id, "weekNumber": week, "date": "2026-01-01T00:00Z"},
                "categories": [{"name": "general", "totals": ["78.6", "3.0", "81.5"]}],
            }
        ],
    }


@pytest.fixture
def inputs(monkeypatch):
    current = pd.DataFrame(
        [
            {
                "player_id": "qb1",
                "season": 2026,
                "week": 1,
                "position": "QB",
                "recent_team": "SEA",
                "qbr_total": np.nan,
                "pts_added": np.nan,
            },
            {
                "player_id": "rb1",
                "season": 2026,
                "week": 1,
                "position": "RB",
                "recent_team": "SEA",
                "qbr_total": np.nan,
                "pts_added": np.nan,
            },
        ],
        index=[8, 9],
    )
    current.attrs["source"] = "preserve"
    schedules = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "espn": "101",
                "game_type": "REG",
                "gameday": "2026-01-01",
                "home_team": "SEA",
                "away_team": "NE",
                "home_score": 13,
                "away_score": 10,
            }
        ]
    )
    monkeypatch.setattr(
        live_qbr.nfl_source,
        "player_ids",
        lambda: pd.DataFrame({"espn_id": [10, 20], "gsis_id": ["qb1", "qb2"]}),
    )
    monkeypatch.setattr(live_qbr.espn_live, "_get_json", lambda url: _payload())
    return current, schedules


def test_recovers_exact_qbr_and_points_added_only_for_qbs(inputs):
    current, schedules = inputs
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    assert result.index.tolist() == [8, 9]
    assert result.attrs == current.attrs
    assert result.loc[8, "qbr_total"] == 78.6
    assert result.loc[8, "pts_added"] == 3.0
    assert result.loc[9, ["qbr_total", "pts_added"]].isna().all()
    assert current.qbr_total.isna().all()
    assert metadata["status"] == "available"
    assert metadata["recovered_rows"] == 1
    assert metadata["recovered_cells"] == 2
    assert metadata["validated_games"] == ["101"]
    assert metadata["retrieved_at"]


def test_preserves_archive_observation_when_filling_other_stat(inputs):
    current, schedules = inputs
    current.loc[8, "qbr_total"] = 77.0
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    assert result.loc[8, "qbr_total"] == 77.0
    assert result.loc[8, "pts_added"] == 3.0
    assert metadata["recovered_cells"] == 1


@pytest.mark.parametrize(
    "reason", ["already_observed", "future_game", "unplayed", "other_season", "no_qbs"]
)
def test_does_not_fetch_when_no_completed_qb_values_need_recovery(inputs, monkeypatch, reason):
    current, schedules = inputs
    if reason == "already_observed":
        current.loc[8, ["qbr_total", "pts_added"]] = [60.0, 0.1]
    elif reason == "future_game":
        schedules["gameday"] = "2999-01-01"
    elif reason == "unplayed":
        schedules["home_score"] = np.nan
    elif reason == "other_season":
        current["season"] = 2025
    else:
        current["position"] = "RB"
    monkeypatch.setattr(live_qbr.espn_live, "_get_json", lambda _: pytest.fail("unexpected fetch"))
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["requested_weeks"] == []
    assert metadata["retrieved_at"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_year",
        "wrong_week",
        "wrong_type",
        "future",
        "missing_stat",
        "nonfinite",
        "bad_scale",
        "conflict",
        "pagination",
    ],
)
def test_malformed_or_ambiguous_provider_response_keeps_missing_values(
    inputs, monkeypatch, mutation
):
    current, schedules = inputs
    payload = _payload()
    if mutation == "wrong_year":
        payload["currentValues"]["season"] = 2025
    elif mutation == "wrong_week":
        payload["athletes"][0]["game"]["weekNumber"] = 2
    elif mutation == "wrong_type":
        payload["currentValues"]["seasontype"] = 3
    elif mutation == "future":
        payload["athletes"][0]["game"]["date"] = "2999-01-01T00:00Z"
    elif mutation == "missing_stat":
        payload["categories"][0]["names"][0] = "unknown"
    elif mutation == "nonfinite":
        payload["athletes"][0]["categories"][0]["totals"][0] = "NaN"
    elif mutation == "bad_scale":
        payload["athletes"][0]["categories"][0]["totals"][0] = "101"
    elif mutation == "conflict":
        duplicate = copy.deepcopy(payload["athletes"][0])
        duplicate["categories"][0]["totals"][0] = "70"
        payload["athletes"].append(duplicate)
    else:
        payload["pagination"]["pages"] = 2
    monkeypatch.setattr(live_qbr.espn_live, "_get_json", lambda _: payload)
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["status"] == "unavailable"
    assert metadata["errors"]


def test_identical_provider_duplicates_cannot_fan_out_rows(inputs, monkeypatch):
    payload = _payload()
    payload["athletes"].append(copy.deepcopy(payload["athletes"][0]))
    monkeypatch.setattr(live_qbr.espn_live, "_get_json", lambda _: payload)
    result, metadata = live_qbr.recover_qbr(*inputs, 2026)
    assert len(result) == 2
    assert metadata["recovered_rows"] == 1


@pytest.mark.parametrize("scheduled", [False, True])
def test_rejects_unscheduled_game_and_valid_game_for_wrong_team(inputs, monkeypatch, scheduled):
    current, schedules = inputs
    if scheduled:
        other = schedules.iloc[0].copy()
        other[["espn", "home_team", "away_team"]] = ["102", "KC", "DEN"]
        schedules = pd.concat([schedules, other.to_frame().T], ignore_index=True)
    monkeypatch.setattr(live_qbr.espn_live, "_get_json", lambda _: _payload(game_id="102"))
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    pd.testing.assert_frame_equal(result, current)
    assert metadata["recovered_rows"] == 0
    assert metadata["rejected_rows"] == (0 if scheduled else 1)


def test_maps_stats_by_name_and_preserves_negative_points_added():
    payload = _payload()
    payload["categories"][0]["names"] = ["qbpaa", "qbr", "schedAdjQBR"]
    payload["athletes"][0]["categories"][0]["totals"] = ["-3.0", "81.5", "78.6"]
    parsed = live_qbr._parse_qbr(payload, 2026, 1)
    assert parsed.iloc[0].qbr_total == 78.6
    assert parsed.iloc[0].pts_added == -3.0


def test_parallel_week_fetches_report_partial_coverage_on_outage(inputs, monkeypatch):
    current, schedules = inputs
    second = current.iloc[[0]].assign(week=2, player_id="qb2")
    current = pd.concat([current, second], ignore_index=True)
    schedules = pd.concat([schedules, schedules.assign(week=2, espn="102")], ignore_index=True)
    seen = []

    def fetch(url):
        params = parse_qs(urlparse(url).query)
        seen.append(params)
        if params["week"] == ["2"]:
            raise RuntimeError("provider unavailable")
        return _payload()

    monkeypatch.setattr(live_qbr.espn_live, "_get_json", fetch)
    result, metadata = live_qbr.recover_qbr(current, schedules, 2026)
    assert len(seen) == 2
    assert all(params["isqualified"] == ["true"] for params in seen)
    assert metadata["status"] == "partial"
    assert metadata["requested_weeks"] == [1, 2]
    assert metadata["observed_rows"] == 1
    assert metadata["eligible_qb_rows"] == 2
    assert result.loc[result.player_id.eq("qb2"), "qbr_total"].isna().all()
    assert metadata["errors"] == [{"week": 2, "error": "provider unavailable"}]


def test_identity_feed_failure_is_disclosed_without_filling(inputs, monkeypatch):
    def failure():
        raise RuntimeError("IDs unavailable")

    monkeypatch.setattr(live_qbr.nfl_source, "player_ids", failure)
    result, metadata = live_qbr.recover_qbr(*inputs, 2026)
    pd.testing.assert_frame_equal(result, inputs[0])
    assert metadata["status"] == "unavailable"
    assert metadata["errors"] == [{"week": None, "error": "IDs unavailable"}]
