"""Export the prepared, unscaled per-game frames that synthetic-history recipes consume.

Skill positions (QB/RB/WR/TE) start from the split parquets and run the shared
production preparation once, without training: the exported frame is the
train split after target computation, schedule and team box-score merges and
feature engineering (``PreparedDataset.train``). DST is built the way its
runner builds it, from the raw caches (team-level rows, no split parquets),
and additionally exports the opponent-offense per-game frame its second
attention stream reads plus the regular-season weekly player slice that frame
was aggregated from (the replay's identity control rebuilds the stream from
it). Nothing is fetched here; the raw caches must already be hydrated from a
verified data release (ADR-0026). Outputs are published into a new directory
with their hashes so a cohort manifest can tie back to the exact export. The
production preparation logs to stdout, so the CLI's JSON summary is the final
stdout line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.synthetic_history import (
    KEYS,
    _frame_hash,
    code_hashes,
    duplicate_game_keys,
    publish_artifact_dir,
    runtime_versions,
    validate_opponent_per_game,
)
from src.analysis.synthetic_history_schema import POSITION_HISTORY_SCHEMAS
from src.config import CACHE_DIR, SEASONS, SPLITS_DIR, TRAIN_SEASONS, VAL_SEASONS
from src.features.engineer import build_opp_offense_per_game_df
from src.prediction.bundle import file_digest
from src.shared.pipeline import _prepare_position_data, _read_split
from src.shared.registry import get_config
from src.training.context import raw_data_dir

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
EXPORT_POSITIONS = (*SKILL_POSITIONS, "DST")
# Shared preparation code whose edits change exported values; per-position
# modules are added per export.
SOURCE_LOADER_PATHS = (
    "shared/pipeline.py",
    "shared/feature_build.py",
    "shared/position_data.py",
    "shared/team_box_score.py",
    "features/engineer.py",
    "data/preprocessing.py",
)
DUPLICATE_KEY_SCOPE = (
    "whole prepared frame; a cohort manifest reports only the recipe's position, "
    "regular-season and donor-season pool"
)
# The columns the production opponent-offense aggregation reads from the
# all-position weekly frame, plus the identity the regular-season filter needs.
DST_WEEKLY_COLUMNS = (
    "player_id",
    "position",
    "recent_team",
    "season",
    "week",
    "season_type",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "interceptions",
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
)
DST_RAW_CACHES = ("weekly", "schedules", "team_stats", "dst_scoring_pbp_v1")


def _check_prepared(prepared, position: str):
    """The prepared dataset must match the registry and carry every consumed column."""
    schema = POSITION_HISTORY_SCHEMAS[position]
    if tuple(prepared.feature_columns) != schema.feature_columns:
        raise ValueError(
            f"prepared {position} feature columns differ from the registry; "
            "the preparation and the position whitelist disagree"
        )
    frame = prepared.train
    required = {
        *KEYS,
        *schema.identity_columns,
        *schema.validated_columns,
        *schema.feature_columns,
        *(["fantasy_points"] if schema.fantasy_points_policy == "assert_equal" else []),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prepared {position} frame lacks columns: {missing}")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"prepared {position} frame has missing player/season/week keys")
    return prepared


def export_skill_source(position: str, *, splits_dir: Path = Path(SPLITS_DIR)):
    """The prepared dataset of a skill position, built as production builds it.

    Returns the production ``PreparedDataset``; ``.train`` is the enriched,
    unscaled frame recipes consume, ``.feature_columns`` must match the registry
    and ``.data_id`` identifies the splits, configuration and side inputs.
    """
    if position not in SKILL_POSITIONS:
        raise ValueError(f"{position} is not a skill position; DST exports from its raw caches")
    train = _read_split(Path(splits_dir) / "train.parquet")
    val = _read_split(Path(splits_dir) / "val.parquet")
    prepared = _prepare_position_data(position, get_config(position), train, val)
    return _check_prepared(prepared, position)


def _weekly_cache_path() -> Path:
    return Path(raw_data_dir(CACHE_DIR)) / f"weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"


def dst_raw_cache_files() -> dict[str, Path]:
    root = Path(raw_data_dir(CACHE_DIR))
    return {name: root / f"{name}_{SEASONS[0]}_{SEASONS[-1]}.parquet" for name in DST_RAW_CACHES}


def export_dst_source():
    """DST: the runner's team-level build, prepared, plus the opponent stream inputs.

    Returns ``(prepared, per_game, weekly)``: the production ``PreparedDataset``
    (train seasons), the opponent-offense per-game frame built by the
    production aggregation from the regular-season weekly player slice, and
    that slice itself (the columns the aggregation reads).
    """
    from src.dst.data import build_data
    from src.dst.features import compute_features
    from src.dst.targets import compute_targets

    frame = compute_targets(build_data(allow_scoring_fetch=False))
    compute_features(frame)
    train = frame[frame["season"].isin(TRAIN_SEASONS)].copy()
    val = frame[frame["season"].isin(VAL_SEASONS)].copy()
    prepared = _check_prepared(_prepare_position_data("DST", get_config("DST"), train, val), "DST")
    weekly = pd.read_parquet(_weekly_cache_path())
    missing = sorted(set(DST_WEEKLY_COLUMNS) - set(weekly.columns))
    if missing:
        raise ValueError(f"weekly cache lacks the opponent-offense columns: {missing}")
    weekly = weekly.loc[weekly["season_type"].eq("REG"), list(DST_WEEKLY_COLUMNS)].reset_index(
        drop=True
    )
    per_game = validate_opponent_per_game(
        build_opp_offense_per_game_df(weekly), POSITION_HISTORY_SCHEMAS["DST"]
    )
    return prepared, per_game, weekly


def canonical_values_hash(frame: pd.DataFrame) -> str:
    """Row-order-independent fingerprint, so duplicate game rows cannot reorder it."""
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return _frame_hash(frame.iloc[np.argsort(row_hashes, kind="stable")])


def _manifest(position: str, prepared, splits: dict, extra_code_paths: tuple[str, ...]) -> dict:
    frame = prepared.train
    lower = position.lower()
    return {
        "position": position,
        "source": f"{lower}.parquet",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "seasons": sorted(int(s) for s in frame["season"].unique()),
        "duplicate_game_keys": duplicate_game_keys(frame),
        "duplicate_game_keys_scope": DUPLICATE_KEY_SCOPE,
        "train_seasons": list(TRAIN_SEASONS),
        "val_seasons": list(VAL_SEASONS),
        "values_sha256": canonical_values_hash(frame),
        "prepared_data_id": str(prepared.data_id),
        "splits": splits,
        "versions": runtime_versions(),
        "code_sha256": code_hashes(
            (
                "analysis/synthetic_history_sources.py",
                f"{lower}/config.py",
                f"{lower}/features.py",
                f"{lower}/data.py",
                f"{lower}/targets.py",
                *extra_code_paths,
                *SOURCE_LOADER_PATHS,
            )
        ),
    }


def write_sources(position: str, output: Path, *, splits_dir: Path = Path(SPLITS_DIR)) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if position == "DST":
        return _write_dst_sources(output)
    prepared = export_skill_source(position, splits_dir=splits_dir)
    frame = prepared.train
    name = f"{position.lower()}.parquet"

    def _write(directory: Path) -> None:
        frame.to_parquet(directory / name, index=False)

    splits = {
        split: file_digest(Path(splits_dir) / f"{split}.parquet") for split in ("train", "val")
    }
    manifest = _manifest(position, prepared, splits, ())
    return publish_artifact_dir(output, _write, manifest, manifest_name="sources.json")


def _write_dst_sources(output: Path) -> Path:
    prepared, per_game, weekly = export_dst_source()
    frame = prepared.train

    def _write(directory: Path) -> None:
        frame.to_parquet(directory / "dst.parquet", index=False)
        per_game.to_parquet(directory / "dst_opponent_per_game.parquet", index=False)
        weekly.to_parquet(directory / "dst_opponent_weekly.parquet", index=False)

    caches = {
        name: file_digest(path) if path.is_file() else None
        for name, path in dst_raw_cache_files().items()
    }
    manifest = _manifest("DST", prepared, {}, ("shared/aggregate_targets.py",))
    manifest.update(
        {
            "opponent_per_game": "dst_opponent_per_game.parquet",
            "opponent_per_game_rows": int(len(per_game)),
            "opponent_per_game_values_sha256": _frame_hash(per_game),
            "opponent_weekly": "dst_opponent_weekly.parquet",
            "opponent_weekly_rows": int(len(weekly)),
            "opponent_weekly_values_sha256": canonical_values_hash(weekly),
            "opponent_weekly_columns": list(DST_WEEKLY_COLUMNS),
            "raw_caches": caches,
            "raw_cache_dir": str(raw_data_dir(CACHE_DIR)),
        }
    )
    return publish_artifact_dir(output, _write, manifest, manifest_name="sources.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", required=True, type=str.upper, choices=EXPORT_POSITIONS)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(SPLITS_DIR),
        help="Split parquets for skill positions (DST reads the raw caches instead)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New source directory (never overwritten)"
    )
    args = parser.parse_args(argv)
    try:
        output = write_sources(args.position, args.output, splits_dir=args.splits_dir)
    except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"synthetic-history-sources: {exc}\n")
    manifest = json.loads((output / "sources.json").read_text())
    sys.stdout.flush()
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": manifest["rows"],
                "source": manifest["source"],
                "duplicate_game_keys": len(manifest["duplicate_game_keys"]),
                "opponent_per_game_rows": manifest.get("opponent_per_game_rows"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
