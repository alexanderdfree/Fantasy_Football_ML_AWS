"""Freshness is tied to generation time; missing inputs remain visible to clients."""

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from src.serving import upcoming_status, upcoming_week

pytestmark = pytest.mark.unit
NOW = datetime(2026, 9, 10, 12, tzinfo=UTC)


@pytest.mark.parametrize("age_seconds", [0, 3600, 14400])
def test_fresh_artifact_including_maximum_age_boundary(age_seconds):
    result = upcoming_status.freshness(
        {"available": True, "generated_at": (NOW - timedelta(seconds=age_seconds)).isoformat()},
        now=NOW,
    )
    assert result == {
        "status": "fresh",
        "age_seconds": age_seconds,
        "max_age_seconds": 14400,
        "reason": None,
    }


@pytest.mark.parametrize("timestamp", [None, "", "not-a-date", "2026-09-10T12:00:00"])
def test_missing_or_corrupt_generation_time_is_never_fresh(timestamp):
    result = upcoming_status.freshness({"available": True, "generated_at": timestamp}, now=NOW)
    assert result["status"] == "stale"
    assert result["age_seconds"] is None
    assert result["reason"] == "missing_timestamp"


def test_artifact_from_future_is_unverified_not_fresh():
    result = upcoming_status.freshness(
        {"available": True, "generated_at": (NOW + timedelta(hours=1)).isoformat()}, now=NOW
    )
    assert result["status"] == "stale"
    assert result["reason"] == "invalid_timestamp"


def test_generation_does_not_reset_older_input_age():
    result = upcoming_status.freshness(
        {
            "available": True,
            "generated_at": (NOW - timedelta(hours=3, minutes=45)).isoformat(),
            "inputs_fetched_at": (NOW - timedelta(hours=4, minutes=5)).isoformat(),
        },
        now=NOW,
    )
    assert result["status"] == "stale"
    assert result["reason"] == "refresh_overdue"
    assert result["age_seconds"] == 14700


def test_injury_source_age_survives_fetch_and_artifact_generation():
    result = upcoming_status.freshness(
        {
            "available": True,
            "generated_at": (NOW - timedelta(hours=3)).isoformat(),
            "inputs_fetched_at": (NOW - timedelta(hours=3, minutes=1)).isoformat(),
            "sources": {
                "injuries": {
                    "status": "available",
                    "source_updated_at": (NOW - timedelta(hours=7)).isoformat(),
                }
            },
        },
        now=NOW,
    )
    assert result["status"] == "stale"
    assert result["age_seconds"] == 7 * 3600


@pytest.mark.parametrize(
    "timestamp", [None, "invalid", "2026-09-10T11:00:00", "2026-09-10T13:00:00Z"]
)
def test_invalid_present_input_timestamp_is_not_hidden_by_recent_generation(timestamp):
    result = upcoming_status.freshness(
        {"available": True, "generated_at": NOW.isoformat(), "inputs_fetched_at": timestamp},
        now=NOW,
    )
    assert result["status"] == "stale"
    assert result["reason"] in {"missing_timestamp", "invalid_timestamp"}


def test_offseason_is_unavailable_and_still_retains_age():
    result = upcoming_status.freshness(
        {"available": False, "reason": "offseason", "generated_at": NOW.isoformat()}, now=NOW
    )
    assert result["status"] == "unavailable"
    assert result["age_seconds"] == 0
    assert result["reason"] == "offseason"


def test_redownloaded_old_artifact_stays_stale_and_preserves_predictions(client, monkeypatch):
    payload = {
        "available": True,
        "generated_at": "2026-08-01T00:00:00+00:00",
        "season": 2026,
        "week": 1,
        "scoring": {"ppr": [{"player_id": "00-1", "name": "Last good player", "nn_pred": 15.5}]},
    }
    upcoming_week._write_artifact(payload)
    path = upcoming_week._artifact_path()
    os.utime(path, None)  # Equivalent to just downloading the same old S3 object.
    monkeypatch.setattr(
        upcoming_week.core,
        "_ensure_base_data",
        lambda: pytest.fail("Serving must not build features"),
    )
    response = client.get("/api/upcoming_week")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache"
    body = response.get_json()
    assert body["freshness"]["status"] == "stale"
    assert body["freshness"]["reason"] == "refresh_overdue"
    assert body["data_quality"]["status"] == "unknown"
    assert body["generated_at"] == payload["generated_at"]
    assert body["scoring"] == payload["scoring"]
    with open(path) as stream:
        assert json.load(stream) == payload  # Read-time status never rewrites the artifact.


@pytest.mark.parametrize("contents", ["not json", "null", "[]"])
def test_unreadable_or_nonobject_artifact_returns_warming(client, contents):
    path = upcoming_week._artifact_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        stream.write(contents)
    response = client.get("/api/upcoming_week")
    assert response.status_code == 503
    assert response.get_json()["status"] == "warming"


@pytest.mark.parametrize(
    ("observed", "expected", "status"),
    [(0, 0, "not_needed"), (0, 10, "unavailable"), (1, 10, "partial"), (10, 10, "available")],
)
def test_history_coverage_requires_expected_player_games(observed, expected, status):
    assert upcoming_status.coverage(observed, expected) == {
        "observed": observed,
        "expected": expected,
        "status": status,
    }


def test_complete_coverage_and_no_games_yet_are_positive_controls():
    sources = {
        "injuries": {"status": "available"},
        "history": {
            "completed_games": 2,
            "coverage": {
                "snap_counts": {"status": "available", "observed": 10, "expected": 10},
                "ff_opportunity": {"status": "available", "observed": 10, "expected": 10},
                "qbr": {"status": "available", "observed": 2, "expected": 2},
            },
        },
        "practice": {"unknown_players": 0},
        "weather": {"coverage": {"forecast": 1, "covered_venue": 1}},
        "player_metadata": {"missing_players": 0},
    }
    assert upcoming_status.data_quality(sources) == {"status": "complete", "issues": []}
    sources["history"] = {
        "completed_games": 0,
        "coverage": {"snap_counts": {"status": "not_needed", "observed": 0, "expected": 0}},
    }
    assert upcoming_status.data_quality(sources)["status"] == "complete"


def test_missing_input_families_are_disclosed_independently():
    result = upcoming_status.data_quality(
        {
            "injuries": {"status": "unavailable"},
            "history": {
                "completed_games": 1,
                "coverage": {
                    "snap_counts": {"status": "partial", "observed": 2, "expected": 12},
                    "ff_opportunity": {"status": "unavailable", "observed": 0, "expected": 12},
                    "qbr": {"status": "unavailable", "observed": 0, "expected": 2},
                },
            },
            "practice": {"unknown_players": 4},
            "weather": {"coverage": {"forecast": 1, "unavailable": 1}},
            "player_metadata": {"missing_players": 3},
        }
    )
    assert result["status"] == "degraded"
    assert {issue["source"] for issue in result["issues"]} == {
        "injuries",
        "snap_counts",
        "ff_opportunity",
        "qbr",
        "practice",
        "weather",
        "player_metadata",
    }
    assert all(issue["message"] for issue in result["issues"])


def test_legacy_history_without_coverage_is_not_called_complete():
    result = upcoming_status.data_quality(
        {"injuries": {"status": "available"}, "history": {"completed_games": 2}}
    )
    assert result["status"] == "degraded"
    assert "history" in {issue["source"] for issue in result["issues"]}
