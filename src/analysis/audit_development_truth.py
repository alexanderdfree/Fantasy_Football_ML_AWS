"""Reconstruct reporting truth from an already verified, local data release.

The caller owns manifest/hash verification. This module only reads its explicit
release root, runs production data/target functions, and never fits or fetches.
It retains unavailable comparison actuals rather than replacing them with the
training targets' zero fills. Native provider constant fills remain unchanged.
"""

from __future__ import annotations

import importlib
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.shared.comparison_scoring import ACTUAL_BASIS
from src.shared.comparison_truth import SOURCE_AVAILABLE, SOURCE_BASIS
from src.shared.evaluation_cohorts import KEYS, regular_season_rows
from src.training.context import RunContext, use_context

_SPLITS = tuple(f"splits/{split}.parquet" for split in ("train", "val", "test"))
_WEEKLY = "raw/weekly_2012_2025.parquet"
_SCHEDULES = "raw/schedules_2012_2025.parquet"
_KICKER = "raw/kicker_pbp_2015_2024.parquet"
_BACKFILL = "raw/kicker_backfill_pbp_v1_2025.parquet"
_TEAM_STATS = "raw/team_stats_2012_2025.parquet"
_DST_SCORING = "raw/dst_scoring_pbp_v1_2012_2025.parquet"


def required_files(position: str) -> tuple[str, ...]:
    """Return the exact release-relative dependency identities for this report."""
    position = position.upper()
    if position in ("QB", "RB", "WR", "TE"):
        return _SPLITS
    if position == "K":
        return (
            _WEEKLY,
            _SCHEDULES,
            _KICKER,
            _BACKFILL,
            "raw/snap_counts_2012_2025.parquet",
            "raw/rosters_2012_2025.parquet",
        )
    if position == "DST":
        return (_WEEKLY, _SCHEDULES, _TEAM_STATS, _DST_SCORING)
    raise ValueError(f"Unsupported truth position: {position}")


@contextmanager
def _local_sources_only():
    """Prevent native-loader fallback fetches, including swallowed failures.

    The D/ST loader's optional logo lookup is decorative; an empty local lookup
    preserves all scoring values. Scoped patches are for serial report execution.
    """
    from src.data import nfl_source, release
    from src.k import data as kicker_data

    attempts = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        attempts.append(True)
        raise release.DataReleaseError("Reporting truth cannot fetch or rebuild release data")

    with ExitStack() as stack:
        for name, value in vars(nfl_source).items():
            if (
                callable(value)
                and not name.startswith("_")
                and getattr(value, "__module__", "") == nfl_source.__name__
            ):
                stack.enter_context(patch.object(nfl_source, name, forbidden))
        stack.enter_context(
            patch.object(
                nfl_source,
                "teams",
                lambda: pd.DataFrame(columns=["team_abbr", "team_logo_espn"]),
            )
        )
        stack.enter_context(patch.object(release, "assert_source_fetch_allowed", forbidden))
        stack.enter_context(patch.object(kicker_data, "assert_source_fetch_allowed", forbidden))
        stack.enter_context(patch.object(kicker_data, "atomic_write_parquet", forbidden))
        yield
        if attempts:
            raise release.DataReleaseError("A native loader attempted to fetch/rebuild report data")


def _skill_frame(position: str, root: Path) -> pd.DataFrame:
    data = importlib.import_module(f"src.{position.lower()}.data")
    frame = data.filter_to_position(
        pd.concat([pd.read_parquet(root / path) for path in _SPLITS], ignore_index=True)
    )
    # Split values have already passed production preprocessing. Without the
    # stored pre-fill mask, finite zero-filled targets cannot certify observation.
    basis = f"{position}:{ACTUAL_BASIS}"
    if (
        SOURCE_AVAILABLE not in frame
        or SOURCE_BASIS not in frame
        or not frame[SOURCE_BASIS].eq(basis).all()
        or not frame[SOURCE_AVAILABLE].isin([True, False]).all()
    ):
        raise ValueError(f"{position} splits lack valid pre-imputation comparison certification")
    return frame


def _native_frame(position: str, root: Path) -> pd.DataFrame:
    if position == "K":
        from src.k import data

        # The native loader derives these cache names and participation seasons
        # from module configuration. A later season/schema must not fall through
        # to an unpinned local cache or source rebuild.
        if list(data.GLOBAL_SEASONS) != list(range(2012, 2026)) or list(data.SEASONS) != list(
            range(2015, 2026)
        ):
            raise ValueError("K provider seasons no longer match the pinned report release")
        if not data._cached_pbp_is_current(str(root / _KICKER)):
            raise ValueError("Pinned kicker weekly cache has a stale schema")
        # Validate through the native cache reader before load_data's backfill
        # wrapper, which tolerates some feature-only failures in production.
        backfill = data._load_backfill_pbp(2025)
        return data.load_data(
            weekly=pd.read_parquet(root / _WEEKLY),
            schedules=pd.read_parquet(root / _SCHEDULES),
            pbp=backfill,
            impute_context=False,
        )

    from src.data.dst_scoring import _valid_cache
    from src.dst.data import build_data

    scoring = pd.read_parquet(root / _DST_SCORING)
    if not _valid_cache(scoring, list(range(2012, 2026))):
        raise ValueError("Pinned D/ST scoring cache is incomplete or stale")
    return build_data(
        weekly=pd.read_parquet(root / _WEEKLY),
        schedules=pd.read_parquet(root / _SCHEDULES),
        team_stats=pd.read_parquet(root / _TEAM_STATS),
        scoring_events=scoring,
        allow_scoring_fetch=False,
        impute_context=False,
    )


def build_truth(position: str, release_root: str | Path, *, seasons=None) -> pd.DataFrame:
    """Return canonical regular-season keys, raw targets and certified actuals.

    Keeps the caller's declared scoring/prior seasons, or all released seasons
    when no scope is supplied. There is no test-row min-games
    filter in production; its min-games restriction applies only to training.
    K kick histories are unnecessary for observed truth and are never rebuilt.
    """
    position = position.upper()
    files = required_files(position)
    root = Path(release_root).resolve()
    for relative in files:
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise FileNotFoundError(f"Pinned truth dependency unavailable: {path}")
    context = RunContext(
        output_root=root,
        data_root=root,
        artifact_sink=None,
        report_sink=None,
    )
    with use_context(context), _local_sources_only():
        raw = (
            _native_frame(position, root)
            if position in ("K", "DST")
            else _skill_frame(position, root)
        )
        if seasons is not None:
            if not seasons or any(type(year) is not int for year in seasons):
                raise ValueError("Truth seasons must be an explicit nonempty integer collection")
            raw = raw[raw["season"].isin(seasons)].copy()
        targets = importlib.import_module(f"src.{position.lower()}.targets")
        frame = targets.compute_targets(raw)
    frame = regular_season_rows(frame)
    if frame.empty or frame[KEYS].isna().any().any():
        raise ValueError(f"{position} truth is empty or has missing canonical keys")
    frame["player_id"] = frame["player_id"].astype(str)
    for column in ("season", "week"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not values.eq(values.round()).all():
            raise ValueError(f"Invalid truth key: {column}")
        frame[column] = values.astype(int)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate {position} truth player-weeks")
    return frame.sort_values(KEYS).reset_index(drop=True)
