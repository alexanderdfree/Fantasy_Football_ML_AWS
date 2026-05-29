"""Boundary adapter: ``nflreadpy`` (Polars) → pandas.

This is the **only** module in the codebase that imports ``nflreadpy``. Every
consumer calls these wrappers, which convert Polars → pandas at the boundary, so
all downstream code stays pandas and tests monkeypatch *this* module rather than
the library.

Schema reconciliation (verified against nflreadpy 0.1.5, which serves the modern
nflverse release schemas):

- ``load_player_stats`` returns the modern ``stats_player`` schema; a handful of
  columns are renamed back to the legacy weekly names the pipeline expects. This is
  the same rename the loader's old 2025+ branch applied, now unified across all
  seasons.
- ``load_rosters`` keys players by ``gsis_id``; the pipeline expects ``player_id``.
- ``load_pbp`` returns ~370 columns; each consumer selects only what it needs. The
  Polars ``.select`` before conversion replaces nfl_data_py's removed ``columns=``
  and ``downcast=`` params and keeps the pandas conversion bounded.
- ``load_teams`` / ``load_schedules`` / ``load_snap_counts`` / ``load_injuries`` /
  ``load_depth_charts`` / ``load_ff_playerids`` already expose the columns the
  pipeline reads, so they pass through unchanged.
"""

from __future__ import annotations

import nflreadpy as _nflreadpy
import pandas as pd
import polars as pl

# Modern stats_player → legacy weekly column names. Matches the rename the loader's
# pre-migration 2025+ branch applied before this shim unified the weekly path.
_WEEKLY_RENAME = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}

# PBP column projections (replace nfl_data_py's removed ``columns=``/``downcast=``).
# ``load_pbp`` returns ~370 columns; selecting only these before converting to pandas
# keeps memory bounded. Missing columns are tolerated in ``pbp_data`` so a single
# season's schema break degrades that season instead of erroring the whole load —
# matching the per-year try/except the PBP consumers already wrap the call in.
PBP_REDZONE_COLS: tuple[str, ...] = (
    "season",
    "season_type",
    "week",
    "posteam",
    "rusher_player_id",
    "receiver_player_id",
    "pass_attempt",
    "play_type",
    "yardline_100",
)
PBP_KICKER_COLS: tuple[str, ...] = (
    "season",
    "season_type",
    "week",
    "posteam",
    "kicker_player_id",
    "kicker_player_name",
    "field_goal_attempt",
    "field_goal_result",
    "extra_point_attempt",
    "extra_point_result",
    "kick_distance",
    "fg_prob",
    "qtr",
    "score_differential",
    "play_id",
    "wind",
    "temp",
    "roof",
    "surface",
)


def _to_pandas(df: pl.DataFrame | pl.LazyFrame) -> pd.DataFrame:
    """Convert a Polars frame to numpy-backed pandas.

    Numpy-backed (the default ``to_pandas()``) rather than pyarrow-extension arrays:
    it preserves classic ``NaN`` semantics so the downstream
    ``.astype``/``.fillna``/``.map``/``.merge`` code behaves exactly as it did with
    nfl_data_py's pandas frames.
    """
    if isinstance(df, pl.LazyFrame):  # defensive: 0.1.5 is eager; future-proofing
        df = df.collect()
    return df.to_pandas()


def weekly_data(seasons: list[int]) -> pd.DataFrame:
    df = _to_pandas(_nflreadpy.load_player_stats(seasons, summary_level="week"))
    return df.rename(columns=_WEEKLY_RENAME)


def rosters(seasons: list[int]) -> pd.DataFrame:
    df = _to_pandas(_nflreadpy.load_rosters(seasons))
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df["player_id"] = df["gsis_id"]
    return df


def schedules(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_schedules(seasons))


def snap_counts(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_snap_counts(seasons))


def injuries(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_injuries(seasons))


def depth_charts(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_depth_charts(seasons))


def player_ids() -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_ff_playerids())


def teams() -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_teams())


def pbp_data(seasons: list[int], cols: tuple[str, ...]) -> pd.DataFrame:
    df = _nflreadpy.load_pbp(seasons)
    available = [c for c in cols if c in df.columns]
    return _to_pandas(df.select(available))
