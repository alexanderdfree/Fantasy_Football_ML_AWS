"""Exercise publication boundaries with real injury parsing and artifact writes."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.serving import espn_live
from src.serving import upcoming_week as live
from src.serving.serialization import _pred_col

pytestmark = pytest.mark.unit
TEAMS = {"12": "KC", "24": "LAC"}


def injury_payload(*, items=None, now=None):
    now = now or datetime.now(UTC)
    return {
        "status": "success",
        "season": {"year": 2026},
        "timestamp": now.isoformat(),
        "injuries": [
            {"id": "12", "injuries": items or []},
            {"id": "24", "injuries": []},
        ],
    }


def injury_item(espn_id="10", status="Out", position="RB"):
    return {
        "status": status,
        "athlete": {
            "headshot": {
                "href": f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"
            },
            "displayName": f"Player {espn_id}",
            "team": {"abbreviation": "KC"},
            "position": {"abbreviation": position},
        },
    }


@pytest.fixture
def builder_boundary(monkeypatch, tmp_path):
    """Stop only heavy inference/source boundaries; retain publication logic."""
    monkeypatch.setattr(live.core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "unit-test-no-network")
    monkeypatch.setattr(live, "_last_signature", "last-good-signature")
    good = {"available": True, "generated_at": "2026-09-09T12:00:00Z", "scoring": {"ppr": []}}
    live._write_artifact(good)
    uploads = Mock(return_value=True)
    monkeypatch.setattr(live, "upload_artifact_to_s3", uploads)
    monkeypatch.setattr(espn_live, "next_unplayed_week", lambda *args: (2026, 1))
    slate = pd.DataFrame(
        {
            "team_id": ["12", "24"],
            "recent_team": ["KC", "LAC"],
            "opponent_team": ["LAC", "KC"],
            "is_home": [1, 0],
            "spread_line": [3.0, -3.0],
            "total_line": [46.0, 46.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "game_id": ["2026_01_LAC_KC"],
            "season": [2026],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["LAC"],
        }
    )
    roster = pd.DataFrame(
        {
            "player_id": ["00-out", "00-qb", "00-rb", "00-wr", "00-te", "00-k"],
            "position": ["RB", "QB", "RB", "WR", "TE", "K"],
            "recent_team": ["KC", "KC", "KC", "LAC", "LAC", "KC"],
            "espn_name": ["Ruled out", "QB", "RB", "WR", "TE", "K"],
        }
    )
    monkeypatch.setattr(espn_live, "fetch_slate", lambda *args: (slate, schedule))
    monkeypatch.setattr(espn_live, "fetch_active_rosters", lambda *args, **kwargs: roster)
    weather = {"by_game": {}, "coverage": {"forecast": 1}}
    monkeypatch.setattr(
        live.live_schedule, "fetch_schedule_context", lambda *args: (schedule, weather)
    )
    monkeypatch.setattr(espn_live, "espn_to_gsis_map", lambda: {"10": "00-out", "11": "00-rb"})
    build = Mock(side_effect=AssertionError("Unverified injuries reached feature building"))
    monkeypatch.setattr(live, "build_upcoming_week_frame", build)
    return SimpleNamespace(
        good=good, uploads=uploads, build=build, roster=roster, schedule=schedule
    )


@pytest.mark.parametrize("upload_ok", [True, False])
def test_verified_offseason_is_published_and_failed_upload_does_not_commit_signature(
    builder_boundary, monkeypatch, upload_ok
):
    monkeypatch.setattr(espn_live, "next_unplayed_week", lambda *args: None)
    builder_boundary.uploads.return_value = upload_ok
    if upload_ok:
        result = live.refresh_upcoming_week_cache(force=True)
        assert result["available"] is False
        assert result["reason"] == "offseason"
        assert live._last_signature == "offseason"
    else:
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            live.refresh_upcoming_week_cache(force=True)
        assert live._last_signature == "last-good-signature"
    builder_boundary.uploads.assert_called_once_with()
    assert live.read_cached_artifact()["reason"] == "offseason"
    builder_boundary.build.assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "outage",
        "wrong_season",
        "unsuccessful",
        "missing_timestamp",
        "stale",
        "future",
        "missing_team",
        "malformed_blocks",
        "unknown_status",
        "missing_athlete",
    ],
)
def test_unverified_injury_report_aborts_before_replacing_or_publishing(
    builder_boundary, monkeypatch, fault
):
    payload = injury_payload()
    if fault == "wrong_season":
        payload["season"]["year"] = 2025
    elif fault == "unsuccessful":
        payload["status"] = "error"
    elif fault == "missing_timestamp":
        payload.pop("timestamp")
    elif fault == "stale":
        payload["timestamp"] = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    elif fault == "future":
        payload["timestamp"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    elif fault == "missing_team":
        payload["injuries"].pop()
    elif fault == "malformed_blocks":
        payload["injuries"][0]["injuries"] = None
    elif fault == "unknown_status":
        payload["injuries"][0]["injuries"] = [injury_item(status="Pending evaluation")]
    elif fault == "missing_athlete":
        payload["injuries"][0]["injuries"] = [{"status": "Out", "athlete": {}}]
    request = (
        Mock(side_effect=ConnectionError("offline"))
        if fault == "outage"
        else Mock(return_value=payload)
    )
    monkeypatch.setattr(espn_live, "_get_json", request)
    # Fail if validation tries downstream downloads before rejecting bad inputs.
    monkeypatch.setattr(
        live.nfl_source,
        "rosters_weekly",
        Mock(side_effect=AssertionError("Unverified injuries reached enrichment")),
    )
    with pytest.raises((ConnectionError, ValueError)):
        live.refresh_upcoming_week_cache(force=True)
    assert live.read_cached_artifact() == builder_boundary.good
    assert live._last_signature == "last-good-signature"
    builder_boundary.uploads.assert_not_called()
    builder_boundary.build.assert_not_called()
    request.assert_called_once()


def test_valid_empty_team_reports_are_verified_healthy_not_outages(monkeypatch):
    now = datetime(2026, 9, 10, 12, tzinfo=UTC)
    request = Mock(return_value=injury_payload(now=now))
    monkeypatch.setattr(espn_live, "_get_json", request)
    monkeypatch.setattr(espn_live, "espn_to_gsis_map", lambda: {})
    result = espn_live.fetch_injury_report(2026, 1, TEAMS, now=now)
    assert result.injuries.empty
    assert result.statuses == {}
    assert result.metadata["status"] == "available"
    assert result.metadata["covered_teams"] == ["KC", "LAC"]
    assert result.metadata["reported_players"] == 0
    request.assert_called_once()


def test_partial_scheduled_team_roster_preserves_last_good_publication(
    builder_boundary, monkeypatch
):
    partial = builder_boundary.roster.loc[builder_boundary.roster.recent_team.eq("KC")]
    monkeypatch.setattr(espn_live, "fetch_active_rosters", lambda *args, **kwargs: partial)
    with pytest.raises(RuntimeError, match="missing scheduled teams.*LAC"):
        live.refresh_upcoming_week_cache(force=True)
    assert live.read_cached_artifact() == builder_boundary.good
    builder_boundary.uploads.assert_not_called()
    builder_boundary.build.assert_not_called()


def test_one_injury_snapshot_drives_out_exclusion_and_status_in_published_six_positions(
    builder_boundary, monkeypatch
):
    payload = injury_payload(items=[injury_item(), injury_item("11", "Questionable")])
    request = Mock(return_value=payload)
    monkeypatch.setattr(espn_live, "_get_json", request)
    monkeypatch.setattr(live.nfl_source, "rosters_weekly", lambda *args: pd.DataFrame())
    monkeypatch.setattr(espn_live, "fetch_depth_chart_ranks", lambda *args: {})
    monkeypatch.setattr(
        live.practice_reports,
        "fetch_practice_report",
        lambda *args: SimpleNamespace(values={}, metadata={"unknown_players": 0}),
    )
    monkeypatch.setattr(live.live_sources, "fetch_contract_features", lambda *args: None)
    monkeypatch.setattr(live, "_fetch_upcoming_expert_frames", lambda *args: (None, None, None))
    monkeypatch.setattr(live.core, "_ensure_base_data", lambda: None)
    monkeypatch.setattr(live.core, "_compute_models_fingerprint", lambda: "test-models")
    monkeypatch.setattr(live.core, "_degraded_positions", lambda: [])
    monkeypatch.setattr(live.app_pkg, "_cache", {"k_kicks_df": pd.DataFrame()})
    special = SimpleNamespace(digest="special", source_status={})
    monkeypatch.setattr(
        live.upcoming_special_teams, "prepare_special_teams", lambda *args, **kwargs: special
    )
    seen = {}

    def build(season, week, slate, roster, **kwargs):
        seen["roster"] = roster.copy()
        seen["injuries"] = kwargs["injuries_df"].copy()
        seen["statuses"] = deepcopy(kwargs["game_status_map"])
        frame = roster.copy()
        frame.attrs["live_history_sources"] = {"completed_games": 0, "coverage": {}}
        return frame

    monkeypatch.setattr(live, "build_upcoming_week_frame", build)

    def infer(frame, roster, slate, season, week, **kwargs):
        seen["inference_roster"] = roster.copy()
        result = pd.concat(
            [
                roster,
                pd.DataFrame(
                    [
                        {
                            "player_id": "KC",
                            "position": "DST",
                            "recent_team": "KC",
                            "espn_name": "Kansas City",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        result["season"] = season
        result["week"] = week
        result["player_display_name"] = result["espn_name"]
        for scoring in ("ppr", "half_ppr", "standard"):
            for model in ("ridge", "nn", "attn_nn", "lgbm"):
                result[_pred_col(model, scoring)] = 12.5
        return result

    monkeypatch.setattr(live, "run_upcoming_inference", infer)
    monkeypatch.setattr(
        live.roster_meta,
        "attach_age_and_rookie",
        lambda results, **kwargs: results.assign(age=25, is_rookie=False),
    )
    result = live.refresh_upcoming_week_cache(force=True)
    assert "00-out" not in set(seen["roster"]["player_id"])
    assert "00-out" not in set(seen["inference_roster"]["player_id"])
    assert seen["injuries"].set_index("gsis_id").loc["00-out", "report_status"] == "Out"
    assert seen["statuses"] == {"00-out": 0.0, "00-rb": 0.5}
    assert {row["position"] for row in result["scoring"]["ppr"]} == {
        "QB",
        "RB",
        "WR",
        "TE",
        "K",
        "DST",
    }
    assert "00-out" not in {row["player_id"] for row in result["scoring"]["ppr"]}
    assert result["sources"]["injuries"]["source_updated_at"] == payload["timestamp"]
    assert result["sources"]["injuries"]["reported_players"] == 2
    assert live.read_cached_artifact() == result
    assert live._last_signature == result["input_signature"]
    builder_boundary.uploads.assert_called_once_with()
    request.assert_called_once()
