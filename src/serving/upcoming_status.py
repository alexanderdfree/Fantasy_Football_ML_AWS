"""Freshness and source coverage for precomputed upcoming-week artifacts."""

from __future__ import annotations

from datetime import UTC, datetime

MAX_ARTIFACT_AGE_SECONDS = 4 * 3600  # 3-hour builds plus build/download headroom.


def freshness(payload: dict, *, now: datetime | None = None) -> dict:
    """Evaluate generation time at read time, never the S3 download timestamp."""
    now = now or datetime.now(UTC)
    age = None
    reason = None
    timestamps = [payload.get("generated_at")]
    if "inputs_fetched_at" in payload:
        timestamps.append(payload["inputs_fetched_at"])
    injury = (payload.get("sources") or {}).get("injuries") or {}
    if injury.get("status") == "available":
        timestamps.append(injury.get("source_updated_at"))
    for value in timestamps:
        try:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("timestamp requires a timezone")
            elapsed = int((now - timestamp).total_seconds())
            age = max(age, elapsed) if age is not None else elapsed
            if elapsed < -300:
                reason = "invalid_timestamp"
        except (TypeError, ValueError, OverflowError):
            reason = "missing_timestamp"
    if reason is None and age is not None and age > MAX_ARTIFACT_AGE_SECONDS:
        reason = "refresh_overdue"
    status = "stale" if reason else "fresh"
    if payload.get("available") is False:
        status = "unavailable"
    return {
        "status": status,
        "age_seconds": max(0, age) if age is not None else None,
        "max_age_seconds": MAX_ARTIFACT_AGE_SECONDS,
        "reason": reason or payload.get("reason"),
    }


def coverage(observed: int, expected: int) -> dict:
    """An existing file or one matched row does not establish complete coverage."""
    status = (
        "not_needed"
        if not expected
        else "unavailable"
        if not observed
        else "partial"
        if observed < expected
        else "available"
    )
    return {"status": status, "observed": int(observed), "expected": int(expected)}


def data_quality(sources: dict) -> dict:
    """User-facing disclosure, separate from artifact age and model availability."""
    if not sources:
        return {
            "status": "unknown",
            "issues": [
                {
                    "source": "coverage",
                    "message": "Source coverage is not recorded for this update.",
                }
            ],
        }
    issues = []
    roster = sources.get("roster") or {}
    unresolved = roster.get("unresolved_players") or []
    if unresolved:
        issues.append(
            {
                "source": "roster",
                "message": f"Some roster players are omitted because their identity or active eligibility could not be verified ({len(unresolved)} affected).",
            }
        )
    if (sources.get("injuries") or {}).get("status") != "available":
        issues.append(
            {
                "source": "injuries",
                "message": "Injury-report freshness has not been verified for this update.",
            }
        )
    history = sources.get("history") or {}
    labels = {
        "snap_counts": "Snap counts",
        "ff_opportunity": "Expected-opportunity statistics",
        "qbr": "Quarterback ratings",
    }
    if history.get("completed_games", 0) and "coverage" not in history:
        issues.append(
            {
                "source": "history",
                "message": "Detailed coverage of recent game history is not recorded for this update.",
            }
        )
    for name, value in history.get("coverage", {}).items():
        if value["status"] not in ("available", "not_needed"):
            issues.append(
                {
                    "source": name,
                    "message": f"{labels.get(name, name)} cover {value['observed']} of {value['expected']} completed player-games; missing history uses the model's existing defaults.",
                }
            )
    practice = sources.get("practice") or {}
    if practice.get("unknown_players", 0):
        issues.append(
            {
                "source": "practice",
                "message": f"Practice participation is unknown for {practice['unknown_players']} players; projections use the training-average value.",
            }
        )
    weather = sources.get("weather") or {}
    missing_weather = sum(
        count
        for name, count in weather.get("coverage", {}).items()
        if name not in ("forecast", "covered_venue")
    )
    if missing_weather:
        issues.append(
            {
                "source": "weather",
                "message": f"Weather or roof status is uncertain for {missing_weather} games; missing weather uses the model's existing defaults.",
            }
        )
    metadata = sources.get("player_metadata") or {}
    if metadata.get("missing_players", 0):
        issues.append(
            {
                "source": "player_metadata",
                "message": f"Age or rookie information is unavailable for {metadata['missing_players']} players.",
            }
        )
    return {"status": "degraded" if issues else "complete", "issues": issues}
