"""Immutable practice observations paired with the forecast actually published.

Runs in the offline artifact builder, never the serving container. Each record
is an observation, not a practice session. A source without a report timestamp
is first known at collection time; neither polling nor a later revision backdates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.practice_reports import PracticeReport

SCHEMA_VERSION = 1
ARCHIVE_DIRECTORY = "practice_archive"


def build_cohort_context(frame: pd.DataFrame, season: int, week: int, *, reference=None) -> dict:
    """Freeze pregame cohort membership, never reconstruct it from game outcomes."""
    from src.shared.comparison_scoring import comparison_actuals
    from src.shared.comparison_truth import comparison_source_availability
    from src.shared.evaluation_cohorts import (
        load_reference,
        ranked_rows,
        reference_selection,
        regular_season_rows,
    )

    if reference is None:
        reference = load_reference()
    current = frame[frame["season"].eq(season) & frame["week"].eq(week)]
    context = {"players": [], "cohorts": {}}
    for position, players in current.groupby("position"):
        if position not in {"QB", "RB", "WR", "TE"}:
            continue
        prior = regular_season_rows(
            frame[frame["season"].eq(season - 1) & frame["position"].eq(position)]
        ).copy()
        # The general frame retains raw fumble components; preserve unknowns
        # when deriving the target, exactly as the scoring availability contract.
        components = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
        if set(components) <= set(prior):
            prior["fumbles_lost"] = prior[components].sum(axis=1, min_count=len(components))
        prior["importance"] = comparison_actuals(prior, position).where(
            comparison_source_availability(prior, position)
        )
        importance = prior.groupby("player_id", as_index=False)["importance"].mean().dropna()
        # Select on the full prior-season population, before dropping bye/Out
        # players from this week's slate. Neither arm's forecasts select its pool.
        importance["season"] = season - 1
        elite = set(ranked_rows(importance, "importance", ["season"], 24)["player_id"])
        elite_available = not importance.empty
        reference_mask, reference_metadata = reference_selection(position, players, reference, 24)
        reference_available = reference_metadata["status"] == "available"
        context["cohorts"][position] = {
            "elite_top24": {
                "status": "available" if elite_available else "unavailable",
                "definition": "prior_season_mean_shared_component_points",
                "selection_population": "full_previous_regular_season",
            },
            "weekly_reference_top24": reference_metadata,
        }
        for index, row in players.iterrows():
            returning = row.get("is_returning_from_absence")
            game_status = row.get("game_status")
            context["players"].append(
                {
                    "player_id": str(row["player_id"]),
                    "position": position,
                    "returning": bool(returning) if pd.notna(returning) else None,
                    "game_status": float(game_status) if pd.notna(game_status) else None,
                    "elite_top24": str(row["player_id"]) in elite if elite_available else None,
                    "weekly_reference_top24": bool(reference_mask.loc[index])
                    if reference_available
                    else None,
                }
            )
    return context


def archive_refresh(
    report: PracticeReport,
    forecast: dict,
    slate,
    *,
    directory: str | Path,
    bucket: str = "",
    prefix: str = "models",
    s3=None,
    available_at: str | None = None,
) -> Path:
    """Write content-addressed local/S3 evidence; retries cannot replace history.

    Call only after forecast publication (or a verified unchanged-input cache hit).
    Archive failure is visible to the builder while leaving the published forecast
    usable. No serving artifact or source timestamps are rewritten for archival.
    """
    if not forecast.get("available"):
        raise ValueError("Practice archive requires an available forecast")
    season, week = int(forecast["season"]), int(forecast["week"])
    if any((r["season"], r["week"]) != (season, week) for r in report.observations):
        raise ValueError("Practice observations and forecast refer to different weeks")
    games = [
        {
            "team": str(row["recent_team"]),
            "kickoff": row["kickoff"] if isinstance(row.get("kickoff"), str) else None,
        }
        for row in slate.to_dict("records")
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "observed_at": report.metadata["fetched_at"],
        "forecast_available_at": available_at or datetime.now(UTC).isoformat(),
        "practice": {"metadata": report.metadata, "observations": report.observations},
        "games": games,
        "forecast": forecast,
        "evidence_kind": "observed_live_snapshot",
    }
    body = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(body).hexdigest()
    relative = Path(ARCHIVE_DIRECTORY) / f"season={season}" / f"week={week:02d}" / f"{digest}.json"
    path = Path(directory) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file with an exclusive hard link; readers cannot see
    # partial JSON and another writer cannot replace an existing observation.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".practice-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise RuntimeError(f"Immutable practice archive disagrees: {path}") from None
    finally:
        os.unlink(temporary)
    if bucket:
        import boto3
        from botocore.exceptions import ClientError

        s3 = s3 if s3 is not None else boto3.client("s3")
        key = f"{prefix.strip('/')}/predictions_cache/{relative.as_posix()}"
        try:
            s3.put_object(
                Bucket=bucket, Key=key, Body=body, ContentType="application/json", IfNoneMatch="*"
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
                raise
            if s3.get_object(Bucket=bucket, Key=key)["Body"].read() != body:
                raise RuntimeError(f"Immutable practice archive disagrees: {key}") from error
    return path
