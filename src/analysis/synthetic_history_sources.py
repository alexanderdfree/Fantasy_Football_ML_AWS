"""Export the prepared, unscaled per-game frames that synthetic-history recipes consume.

Skill positions (QB/RB/WR/TE) start from the split parquets and run the shared
production preparation once, without training: the exported frame is the
train split after target computation, schedule and team box-score merges and
feature engineering (``PreparedDataset.train``). Nothing is fetched here; the
raw caches the preparation reads must already be hydrated from a verified data
release (ADR-0026). Outputs are published into a new directory with their
hashes so a cohort manifest can tie back to the exact export. The production
preparation logs to stdout, so the CLI's JSON summary is the final stdout line.
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
)
from src.analysis.synthetic_history_schema import POSITION_HISTORY_SCHEMAS
from src.config import SPLITS_DIR, TRAIN_SEASONS, VAL_SEASONS
from src.prediction.bundle import file_digest
from src.shared.pipeline import _prepare_position_data, _read_split
from src.shared.registry import get_config

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
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


def export_skill_source(position: str, *, splits_dir: Path = Path(SPLITS_DIR)):
    """The prepared dataset of a skill position, built as production builds it.

    Returns the production ``PreparedDataset``; ``.train`` is the enriched,
    unscaled frame recipes consume, ``.feature_columns`` must match the registry
    and ``.data_id`` identifies the splits, configuration and side inputs.
    """
    if position not in SKILL_POSITIONS:
        raise ValueError(f"{position} is not a skill position; K and DST export separately")
    train = _read_split(Path(splits_dir) / "train.parquet")
    val = _read_split(Path(splits_dir) / "val.parquet")
    prepared = _prepare_position_data(position, get_config(position), train, val)
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prepared {position} frame lacks columns: {missing}")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"prepared {position} frame has missing player/season/week keys")
    return prepared


def canonical_values_hash(frame: pd.DataFrame) -> str:
    """Row-order-independent fingerprint, so duplicate game rows cannot reorder it."""
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return _frame_hash(frame.iloc[np.argsort(row_hashes, kind="stable")])


def write_sources(position: str, output: Path, *, splits_dir: Path = Path(SPLITS_DIR)) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    prepared = export_skill_source(position, splits_dir=splits_dir)
    frame = prepared.train
    name = f"{position.lower()}.parquet"

    def _write(directory: Path) -> None:
        frame.to_parquet(directory / name, index=False)

    lower = position.lower()
    manifest = {
        "position": position,
        "source": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "seasons": sorted(int(s) for s in frame["season"].unique()),
        "duplicate_game_keys": duplicate_game_keys(frame),
        "duplicate_game_keys_scope": DUPLICATE_KEY_SCOPE,
        "train_seasons": list(TRAIN_SEASONS),
        "val_seasons": list(VAL_SEASONS),
        "values_sha256": canonical_values_hash(frame),
        "prepared_data_id": str(prepared.data_id),
        "splits": {
            split: file_digest(Path(splits_dir) / f"{split}.parquet") for split in ("train", "val")
        },
        "versions": runtime_versions(),
        "code_sha256": code_hashes(
            (
                "analysis/synthetic_history_sources.py",
                f"{lower}/config.py",
                f"{lower}/features.py",
                f"{lower}/data.py",
                f"{lower}/targets.py",
                *SOURCE_LOADER_PATHS,
            )
        ),
    }
    return publish_artifact_dir(output, _write, manifest, manifest_name="sources.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", required=True, type=str.upper, choices=SKILL_POSITIONS)
    parser.add_argument("--splits-dir", type=Path, default=Path(SPLITS_DIR))
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
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
