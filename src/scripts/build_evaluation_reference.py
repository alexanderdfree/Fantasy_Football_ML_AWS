"""Publish a pregame-only reference slate for offline and serving evaluation.

Uses the complete archived forecast pool, not players selected by realized stats.
No actuals, model predictions, or training features enter this artifact.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import pandas as pd

from src.config import TEST_SEASONS
from src.shared.evaluation_cohorts import (
    KEYS,
    REFERENCE_FILENAME,
    REFERENCE_VERSION,
    reference_path,
    regular_season_rows,
)
from src.shared.expert_eligibility import NFLCOM_OFFENSE_MIN_SEASON, filter_eligible_forecasts


def build_reference(
    seasons, *, nflcom_loader=None, rotowire_loader=None, espn_loader=None
) -> pd.DataFrame:
    """Build shared-component ranks: NFL/RotoWire offense, ESPN K, RotoWire DST."""
    from src.analysis.analysis_expert_comparison import _build_experts

    seasons = sorted({int(season) for season in seasons})
    sources = [
        source
        for source in _build_experts(nflcom_loader, rotowire_loader, None, espn_loader)
        if source.name in {"nflcom", "sleeper", "espn"}
    ]
    raw = {}
    for source in sources:
        minimum = NFLCOM_OFFENSE_MIN_SEASON if source.name == "nflcom" else 2018
        supported = [season for season in seasons if season >= minimum]
        raw[source.name] = source.load(supported) if supported else None
    parts = []
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        names = ("espn",) if pos == "K" else ("sleeper",) if pos == "DST" else ("nflcom", "sleeper")
        required = [
            source for source in sources if source.name in names and pos not in source.skipped
        ]
        if len(required) != len(names) or any(raw[source.name] is None for source in required):
            continue
        projected = []
        for source in required:
            frame = regular_season_rows(source.project(raw[source.name], pos, "ppr"))
            # The hvpkod NFL.com offense archive backfills box scores before 2024.
            # RotoWire's usable archive begins in 2018. Never silently substitute
            # another reference recipe when one required source is unavailable.
            frame = filter_eligible_forecasts(frame, source.name, pos)
            first_season = NFLCOM_OFFENSE_MIN_SEASON if source.name == "nflcom" else 2018
            frame = frame[frame["season"].isin(seasons) & frame["season"].ge(first_season)].dropna(
                subset=["expert_pred_total"]
            )
            frame["player_id"] = frame["player_id"].astype(str)
            if frame.duplicated(KEYS).any():
                raise ValueError(f"Duplicate {source.name} forecast player-weeks for {pos}")
            projected.append(frame.rename(columns={"expert_pred_total": source.name}))
        if not projected:
            continue
        combined = projected[0]
        for other in projected[1:]:
            combined = combined.merge(other, on=KEYS, how="inner", validate="one_to_one")
        combined["reference_pred"] = combined[list(names)].mean(axis=1)
        combined = combined.sort_values(
            ["season", "week", "reference_pred", "player_id"], ascending=[True, True, False, True]
        )
        combined["reference_rank"] = combined.groupby(["season", "week"]).cumcount() + 1
        combined["position"] = pos
        combined["reference_source"] = "+".join("rotowire" if n == "sleeper" else n for n in names)
        parts.append(
            combined[[*KEYS, "position", "reference_pred", "reference_rank", "reference_source"]]
        )
    result = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=[*KEYS, "position", "reference_pred", "reference_rank", "reference_source"]
        )
    )
    result["reference_version"] = REFERENCE_VERSION
    result["generated_at"] = datetime.now(UTC).isoformat()
    return result


def write_reference(
    seasons, *, upload=False, nflcom_loader=None, rotowire_loader=None, espn_loader=None
):
    """Replace requested seasons atomically and preserve other historical slates."""
    from src.data.cache_io import atomic_write_parquet
    from src.data.release import assert_source_fetch_allowed

    assert_source_fetch_allowed(reference_path())
    frame = build_reference(
        seasons,
        nflcom_loader=nflcom_loader,
        rotowire_loader=rotowire_loader,
        espn_loader=espn_loader,
    )
    if frame.empty:
        raise ValueError(
            "No valid archived pregame reference forecasts; existing artifact retained"
        )
    path = reference_path()
    if path.exists():
        old = pd.read_parquet(path)
        replaced = old["season"].isin(seasons) & old["reference_version"].eq(REFERENCE_VERSION)
        slate_keys = ["position", *KEYS]
        archived = pd.MultiIndex.from_frame(old.loc[replaced, slate_keys])
        refreshed = pd.MultiIndex.from_frame(frame[slate_keys])
        if len(archived.difference(refreshed)):
            raise ValueError(
                "Reference refresh loses archived forecast coverage; existing artifact retained"
            )
        old = old[~replaced]
        frame = pd.concat([old, frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(frame, str(path))
    if upload:
        import boto3

        bucket = os.environ.get("FF_MODEL_S3_BUCKET", "").strip()
        if not bucket:
            raise ValueError("FF_MODEL_S3_BUCKET is required to publish the evaluation reference")
        boto3.client("s3").upload_file(str(path), bucket, f"data/raw/{REFERENCE_FILENAME}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=TEST_SEASONS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    frame = write_reference(args.seasons, upload=args.upload)
    print(f"Wrote {len(frame)} pregame reference rows to {reference_path()}")


if __name__ == "__main__":
    main()
