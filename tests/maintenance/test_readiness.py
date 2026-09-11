"""Missing games, changed schemas, late reports, and expert availability are visible."""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pandas as pd
import pytest

from src.maintenance import readiness, sources

pytestmark = pytest.mark.unit


def test_archive_checks_find_missing_games_even_with_current_season_and_schema(tmp_path):
    schedule = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["NYJ"],
            "away_team": ["BUF"],
            "home_score": [10],
            "away_score": [14],
        }
    )
    schedule.to_parquet(tmp_path / "schedules_2025_2025.parquet")
    players = pd.DataFrame(
        {
            "player_id": ["a"],
            "season": [2025],
            "week": [1],
            "recent_team": ["NYJ"],
            "season_type": ["REG"],
        }
    )
    path = tmp_path / "weekly_2025_2025.parquet"
    players.to_parquet(path)
    result = readiness.archive_checks(tmp_path, [2025])["player_stats"]
    assert result["schema"]["status"] == "valid"
    assert result["coverage"]["missing_required_seasons"] == []
    assert result["coverage"]["missing_team_games"] == 1
    assert result["readiness"] == "blocked"
    pd.concat([players, players.assign(player_id="b", recent_team="BUF")]).to_parquet(path)
    assert readiness.archive_checks(tmp_path, [2025])["player_stats"]["readiness"] == "ready"


def test_known_2012_snap_gap_is_distinct_from_a_missing_required_season(tmp_path):
    frame = pd.DataFrame(
        {"season": [2013], "week": [1], "team": ["NYJ"], "pfr_player_id": ["player"]}
    )
    frame.to_parquet(tmp_path / "snap_counts_2012_2013.parquet")
    assert readiness.archive_checks(tmp_path, [2012, 2013])["snap_counts"]["readiness"] == "ready"
    blocked = readiness.archive_checks(tmp_path, [2012, 2013, 2014])["snap_counts"]
    assert blocked["coverage"]["missing_required_seasons"] == [2014]
    assert blocked["readiness"] == "blocked"


def test_invalid_archive_columns_do_not_pass_on_row_count(tmp_path):
    pd.DataFrame({"season": [2025], "week": [1], "recent_team": ["NYJ"]}).to_parquet(
        tmp_path / "weekly_2025_2025.parquet"
    )
    result = readiness.archive_checks(tmp_path, [2025])["player_stats"]
    assert result["coverage"]["rows"] == 1
    assert result["schema"]["missing_columns"] == ["player_id"]
    assert result["readiness"] == "blocked"


def live_payload(now):
    return {
        "available": True,
        "season": 2026,
        "week": 1,
        "scoring": {
            "ppr": [
                {"player_id": p, "position": p, "team": "NYJ", "espn_pred": 10.0}
                for p in ["QB", "RB", "WR", "TE"]
            ]
        },
        "sources": {
            "roster": {"covered_teams": ["NYJ", "BUF"], "fetched_at": now.isoformat()},
            "injuries": {
                "covered_teams": ["NYJ", "BUF"],
                "source_updated_at": now.isoformat(),
                "season": 2026,
                "week": 1,
            },
            "practice": {"known_players": 12, "unknown_players": 2},
            "weather": {"games": 1, "coverage": {"forecast": 1}},
            "history": {
                "completed_games": 1,
                "coverage": {
                    "qbr": {"expected": 3, "observed": 3},
                    "snap_counts": {"expected": 20, "observed": 20},
                    "ff_opportunity": {"expected": 15, "observed": 0},
                },
            },
            "experts": {
                "espn": {"status": "available"},
                "rotowire": {"status": "unavailable"},
                "nflcom": {"status": "historical_only", "verified_archive_through_season": 2025},
            },
        },
    }


@pytest.mark.parametrize("mutation", ["missing_team", "stale", "wrong_week", "wrong_type"])
def test_live_injury_readiness_uses_actual_coverage_age_and_slate(mutation):
    now = datetime.now(UTC)
    payload = live_payload(now)
    assert readiness.live_checks(payload, now=now)["injuries"]["readiness"] == "ready"
    injury = payload["sources"]["injuries"]
    if mutation == "missing_team":
        injury["covered_teams"] = ["NYJ"]
    elif mutation == "stale":
        injury["source_updated_at"] = (now - timedelta(hours=5)).isoformat()
    elif mutation == "wrong_week":
        injury["week"] = 2
    else:
        injury["covered_teams"] = "NYJ,BUF"
    assert readiness.live_checks(payload, now=now)["injuries"]["readiness"] == "blocked"


def test_optional_provider_lag_and_expert_coverage_are_not_fake_zeros():
    now = datetime.now(UTC)
    result = readiness.live_checks(live_payload(now), now=now)
    assert result["practice"]["readiness"] == "partial"
    assert result["opportunity"]["readiness"] == "upstream_pending"
    assert result["espn"]["coverage"]["projected_rows"] == 4
    assert result["espn"]["readiness"] == "ready"
    assert result["rotowire"]["readiness"] == "unavailable"
    assert result["nflcom"]["readiness"] == "not_needed"
    assert result["nflcom"]["archive_through_season"] == 2025


def test_expert_value_schema_is_checked_as_well_as_transport_status():
    now = datetime.now(UTC)
    payload = live_payload(now)
    payload["scoring"]["ppr"][0]["espn_pred"] = "unknown"
    result = readiness.live_checks(payload, now=now)["espn"]
    assert result["schema"]["status"] == "incompatible"
    assert result["readiness"] == "unavailable"


def test_offseason_does_not_require_active_week_source_reports():
    result = readiness.live_checks({"available": False, "reason": "offseason"})
    assert set(result) >= {"espn", "rotowire", "nflcom", "injuries", "practice"}
    assert all(x["readiness"] == "not_needed" for x in result.values())


def test_changed_csv_schema_is_reported_before_consumption():
    def fetch(url):
        if urlsplit(url).hostname == "raw.githubusercontent.com":
            return b"renamed_column\nvalue", {}
        return (
            b'{"assets":[{"name":"data.parquet","size":1,"updated_at":"2026-09-10T00:00:00Z"}]}',
            {},
        )

    result = sources.check_sources(fetcher=fetch)
    assert result["sources"]["schedules"]["status"] == "fetch_failed"
    assert "schema missing" in result["sources"]["schedules"]["error"]
    assert all(s["policy"]["publication_expectation"] for s in result["sources"].values())
