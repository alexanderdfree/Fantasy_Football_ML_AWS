"""Per-source publication expectations and observed schema/coverage checks."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from src.maintenance.storage import timestamp

_NIGHTLY = "Nightly after completed games; recheck Thursday for NFL stat corrections"
POLICIES = {
    "player_stats": (
        _NIGHTLY,
        "weekly_[0-9]*.parquet",
        ("player_id", "season", "week", "recent_team"),
    ),
    "team_stats": (_NIGHTLY, "team_stats_*.parquet", ("season", "week", "team")),
    "pbp": (_NIGHTLY, "redzone_pbp_*.parquet", ("player_id", "season", "week")),
    "rosters": (
        "Daily, normally after 07:00 UTC",
        "rosters_[0-9]*.parquet",
        ("player_id", "season", "team"),
    ),
    "weekly_rosters": (
        "Daily, normally after 07:00 UTC",
        "rosters_weekly_*.parquet",
        ("player_id", "season", "week", "team"),
    ),
    "snap_counts": (
        "Provider-dependent; nflverse checks at 00/06/12/18 UTC",
        "snap_counts_*.parquet",
        ("season", "week", "team", "pfr_player_id"),
    ),
    "depth_charts": (
        "Daily, normally after 07:00 UTC; timestamped source since 2025",
        "depth_charts_*.parquet",
        ("gsis_id", "season", "week"),
    ),
    "injuries": (
        "Historical archive; live reports use ESPN/NFL.com below",
        "injuries_*.parquet",
        ("gsis_id", "season", "week"),
    ),
    "qbr": (
        "After completed games; live unqualified appearances may require ESPN recovery",
        "qbr_weekly_*.parquet",
        ("player_id", "season", "week", "qbr_total"),
    ),
    "contracts": (
        "Provider updates following signings; check daily for revisions",
        "contracts_*.parquet",
        ("player_id", "season"),
    ),
    "players": (
        "Provider identity updates; check daily",
        "player_metadata_*.parquet",
        ("gsis_id", "display_name"),
    ),
    "player_ids": (
        "Provider identity crosswalk updates; check daily",
        "player_id_bridge_*.parquet",
        ("gsis_id", "pfr_id"),
    ),
    "opportunity": (
        "Provider-dependent after games; unpublished live seasons remain unavailable",
        "ff_opportunity_*.parquet",
        ("player_id", "season", "week", "pass_yards_gained_exp"),
    ),
    "schedules": (
        "Approximately every five minutes during the season",
        "schedules_*.parquet",
        ("season", "week", "game_type", "home_team", "away_team", "home_score", "away_score"),
    ),
}


def policy(name):
    publication, _, columns = POLICIES[name]
    return {"publication_expectation": publication, "required_archive_columns": list(columns)}


def _team_keys(frame, column):
    # Reuse the production identity bridge's relocation/gamebook normalization.
    from src.data.identity import _TEAM_CODES

    values = frame[["season", "week", column]].dropna().copy()
    values[column] = values[column].replace(_TEAM_CODES)
    return set(values.itertuples(index=False, name=None))


def archive_checks(raw_dir, seasons=None):
    """Inspect the pinned files actually consumed, never infer coverage from mtimes."""
    import pandas as pd
    import pyarrow.parquet as pq

    from src.config import SEASONS

    expected_seasons = set(SEASONS if seasons is None else seasons)
    checks, frames = {}, {}
    for name, (_, pattern, required) in POLICIES.items():
        candidates = sorted(Path(raw_dir).glob(pattern))
        item = {**policy(name), "scope": "pinned_historical_archive", "readiness": "blocked"}
        checks[name] = item
        if len(candidates) != 1:
            item["schema"] = {"status": "missing_or_ambiguous", "matching_files": len(candidates)}
            continue
        try:
            file = pq.ParquetFile(candidates[0])
            columns = set(file.schema.names)
            missing = sorted(set(required) - columns)
            item["schema"] = {
                "status": "incompatible" if missing else "valid",
                "missing_columns": missing,
            }
            item["file"] = candidates[0].name
            item["coverage"] = {"rows": file.metadata.num_rows}
            if missing or not file.metadata.num_rows:
                continue
            selected = sorted(
                columns
                & {
                    "season",
                    "week",
                    "team",
                    "recent_team",
                    "game_type",
                    "season_type",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                }
            )
            frame = file.read(columns=selected).to_pandas()
            frames[name] = frame
            absent = []
            if "season" in frame:
                present = set(pd.to_numeric(frame.season, errors="coerce").dropna().astype(int))
                # 2012 snap counts are explicitly absent upstream (ADR-0026).
                required_years = expected_seasons - ({2012} if name == "snap_counts" else set())
                absent = sorted(required_years - present)
                item["coverage"].update(
                    seasons_present=sorted(present), missing_required_seasons=absent
                )
                if "week" in frame and present:
                    latest = frame[frame.season == max(present)]
                    item["coverage"]["latest_observed_week"] = (
                        int(latest.week.max()) if latest.week.notna().any() else None
                    )
            item["readiness"] = "blocked" if absent else "ready"
        except Exception as error:
            item["schema"] = {"status": "unreadable", "error": str(error)[:200]}
    schedule = frames.get("schedules")
    if schedule is not None:
        played = schedule[
            schedule.game_type.eq("REG") & schedule.home_score.notna() & schedule.away_score.notna()
        ]
        expected_games = _team_keys(played, "home_team") | _team_keys(played, "away_team")
        for name, team in [("player_stats", "recent_team"), ("team_stats", "team")]:
            frame = frames.get(name)
            if frame is None:
                continue
            if "season_type" in frame:
                frame = frame[frame.season_type.eq("REG")]
            missing_games = expected_games - _team_keys(frame, team)
            checks[name]["coverage"].update(
                expected_team_games=len(expected_games),
                missing_team_games=len(missing_games),
                missing_game_sample=[list(k) for k in sorted(missing_games)[:8]],
            )
            if missing_games:
                checks[name]["readiness"] = "blocked"
    return checks


def live_checks(payload, *, now=None):
    """Validate fetched feed metadata and expert rows from this actual inference run."""
    now = now or datetime.now(UTC)
    if payload.get("available") is False and payload.get("reason") == "offseason":
        return {
            name: {"readiness": "not_needed", "reason": "verified_offseason"}
            for name in (
                "roster",
                "injuries",
                "practice",
                "weather",
                "qbr",
                "snap_counts",
                "opportunity",
                "espn",
                "rotowire",
                "nflcom",
            )
        }
    sources = payload.get("sources", {})
    rows = payload.get("scoring", {}).get("ppr", [])
    raw_teams = sources.get("roster", {}).get("covered_teams", [])
    teams = (
        set(raw_teams)
        if isinstance(raw_teams, list) and all(isinstance(t, str) for t in raw_teams)
        else set()
    )
    checks = {}
    for name, expectation, fields, critical in [
        (
            "roster",
            "Current active roster for every scheduled team",
            ("covered_teams", "fetched_at"),
            True,
        ),
        (
            "injuries",
            "Current week, every scheduled team, source age <= four hours",
            ("covered_teams", "source_updated_at", "season", "week"),
            True,
        ),
        (
            "practice",
            "Game-relative official practice reports; missing reports remain unknown",
            ("known_players", "unknown_players"),
            False,
        ),
        (
            "weather",
            "Current kickoff forecast or verified covered venue",
            ("coverage", "games"),
            False,
        ),
    ]:
        source = sources.get(name, {})
        missing = [f for f in fields if f not in source]
        item = {
            "publication_expectation": expectation,
            "schema": {"status": "incompatible" if missing else "valid", "missing_fields": missing},
            "coverage": {},
            "readiness": "ready",
            "required": critical,
        }
        checks[name] = item
        if missing:
            item["readiness"] = "blocked" if critical else "unavailable"
            continue
        if name in {"roster", "injuries"}:
            valid = isinstance(source["covered_teams"], list) and all(
                isinstance(t, str) for t in source["covered_teams"]
            )
        elif name == "practice":
            valid = all(type(source[k]) is int and source[k] >= 0 for k in fields)
        else:
            valid = (
                isinstance(source["coverage"], dict)
                and type(source["games"]) is int
                and source["games"] >= 0
                and all(type(v) is int and v >= 0 for v in source["coverage"].values())
            )
        if not valid:
            item["schema"]["status"] = "incompatible"
            item["readiness"] = "blocked" if critical else "unavailable"
            continue
        if name in {"roster", "injuries"}:
            missing_teams = teams - set(source["covered_teams"])
            item["coverage"] = {
                "expected_teams": len(teams),
                "observed_teams": len(source["covered_teams"]),
                "missing_teams": sorted(missing_teams),
            }
            try:
                age = (
                    now
                    - timestamp(source["source_updated_at" if name == "injuries" else "fetched_at"])
                ).total_seconds()
            except (ValueError, TypeError, AttributeError):
                item["schema"]["status"] = "incompatible"
                item["readiness"] = "blocked"
                continue
            item["age_seconds"] = max(0, int(age))
            if missing_teams or not teams or not -300 <= age <= 4 * 3600:
                item["readiness"] = "blocked"
            if name == "injuries" and any(source[k] != payload.get(k) for k in ("season", "week")):
                item["readiness"] = "blocked"
                item["coverage"]["wrong_slate"] = True
        elif name == "practice":
            item["coverage"] = {k: source[k] for k in fields}
            item["readiness"] = "partial" if source["unknown_players"] else "ready"
        else:
            item["coverage"] = {"expected_games": source["games"], "by_status": source["coverage"]}
            good = sum(source["coverage"].get(k, 0) for k in ("forecast", "covered_venue"))
            item["readiness"] = "partial" if good < source["games"] else "ready"
    history = sources.get("history", {})
    for name, key in [
        ("qbr", "qbr"),
        ("snap_counts", "snap_counts"),
        ("opportunity", "ff_opportunity"),
    ]:
        coverage = history.get("coverage", {}).get(key)
        item = {
            "publication_expectation": "After provider publishes completed-game data",
            "required": False,
            "schema": {
                "status": "not_needed" if not history.get("completed_games") else "unavailable"
            },
            "coverage": coverage or {},
            "readiness": "not_needed" if not history.get("completed_games") else "unavailable",
        }
        if coverage:
            valid = all(
                isinstance(coverage.get(k), int) and coverage[k] >= 0
                for k in ("expected", "observed")
            )
            item["schema"] = {"status": "valid" if valid else "incompatible"}
            if valid:
                item["readiness"] = (
                    "ready" if coverage["observed"] >= coverage["expected"] else "upstream_pending"
                )
        checks[name] = item
    for name in ("espn", "rotowire", "nflcom"):
        provider = sources.get("experts", {}).get(name, {})
        eligible = [r for r in rows if r.get("position") in ("QB", "RB", "WR", "TE")]
        observed = [
            r
            for r in eligible
            if type(r.get(name + "_pred")) in (int, float) and math.isfinite(r[name + "_pred"])
        ]
        invalid_rows = any(
            r.get(name + "_pred") is not None
            and (
                type(r[name + "_pred"]) not in (int, float) or not math.isfinite(r[name + "_pred"])
            )
            for r in eligible
        )
        positions = sorted({r["position"] for r in observed})
        historical = provider.get("status") == "historical_only"
        # Expert feeds need not project every bench player. Record that denominator
        # without treating ordinary provider eligibility as fabricated zero points.
        checks[name] = {
            "publication_expectation": "Current pregame projections, provider-dependent updates",
            "required": False,
            "schema": {
                "status": "valid"
                if isinstance(provider.get("status"), str) and not invalid_rows
                else "incompatible"
            },
            "coverage": {
                "eligible_roster_rows": len(eligible),
                "projected_rows": len(observed),
                "positions": positions,
                "missing_positions": sorted({r["position"] for r in eligible} - set(positions)),
            },
            "readiness": "not_needed"
            if historical
            else "unavailable"
            if invalid_rows or not observed
            else "partial"
            if len(observed) < len(eligible)
            else "ready",
        }
        if historical:
            checks[name]["archive_through_season"] = provider.get("verified_archive_through_season")
    return checks


def complete_report(index_report, raw_dir, *, payload=None, seasons=None):
    archive = archive_checks(raw_dir, seasons)
    live = live_checks(payload) if payload is not None else {}
    blocked = any(c["readiness"] == "blocked" for c in (*archive.values(), *live.values()))
    degraded = any(c["readiness"] not in {"ready", "not_needed"} for c in live.values())
    return {
        **index_report,
        "archive_checks": archive,
        "live_checks": live,
        "readiness": "blocked" if blocked else "degraded" if degraded else "ready",
        "coverage_note": "Archive schema/seasons/completed-team-game coverage and this run's live/expert inputs are checked separately from upstream revision age.",
    }
