"""Export the prepared, unscaled per-game frames that synthetic-history recipes consume.

Skill positions (QB/RB/WR/TE) start from the split parquets and run the shared
production preparation once, without training: the exported frame is the
train split after target computation, schedule and team box-score merges and
feature engineering (``PreparedDataset.train``). Nothing is fetched here; the
raw caches the preparation reads must already be hydrated from a verified data
release (ADR-0026). Outputs are published into a new directory with their
hashes so a cohort manifest can tie back to the exact export.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import pandas as pd

from src.analysis.synthetic_history import (
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

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SOURCE_LOADER_PATHS = ("shared/pipeline.py", "shared/feature_build.py", "features/engineer.py")


def export_skill_source(position: str, *, splits_dir: Path = Path(SPLITS_DIR)) -> pd.DataFrame:
    """The enriched, unscaled train frame for a skill position, prepared as production does."""
    if position not in SKILL_POSITIONS:
        raise ValueError(f"{position} is not a skill position; K and DST export separately")
    config = import_module(f"src.{position.lower()}.run_pipeline").CONFIG
    train = _read_split(Path(splits_dir) / "train.parquet")
    val = _read_split(Path(splits_dir) / "val.parquet")
    frame = _prepare_position_data(position, config, train, val)[6]
    schema = POSITION_HISTORY_SCHEMAS[position]
    missing = sorted(set(schema.feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"prepared {position} frame lacks feature columns: {missing}")
    return frame


def write_sources(position: str, output: Path, *, splits_dir: Path = Path(SPLITS_DIR)) -> Path:
    frame = export_skill_source(position, splits_dir=splits_dir)
    name = f"{position.lower()}.parquet"

    def _write(directory: Path) -> None:
        frame.to_parquet(directory / name, index=False)

    manifest = {
        "position": position,
        "source": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "seasons": sorted(int(s) for s in frame["season"].unique()),
        "duplicate_game_keys": duplicate_game_keys(frame),
        "train_seasons": list(TRAIN_SEASONS),
        "val_seasons": list(VAL_SEASONS),
        "values_sha256": _frame_hash(
            frame.sort_values(["player_id", "season", "week"], kind="stable")
        ),
        "splits": {
            split: file_digest(Path(splits_dir) / f"{split}.parquet") for split in ("train", "val")
        },
        "versions": runtime_versions(),
        "code_sha256": code_hashes(
            (
                "analysis/synthetic_history_sources.py",
                f"{position.lower()}/config.py",
                *SOURCE_LOADER_PATHS,
            )
        ),
    }
    return publish_artifact_dir(output, _write, manifest, manifest_name="sources.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", required=True, choices=SKILL_POSITIONS)
    parser.add_argument("--splits-dir", type=Path, default=Path(SPLITS_DIR))
    parser.add_argument(
        "--output", type=Path, required=True, help="New source directory (never overwritten)"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        output = write_sources(args.position, args.output, splits_dir=args.splits_dir)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.exit(2, f"synthetic-history-sources: {exc}\n")
    manifest = json.loads((output / "sources.json").read_text())
    print(
        json.dumps({"output": str(output), "rows": manifest["rows"], "source": manifest["source"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
