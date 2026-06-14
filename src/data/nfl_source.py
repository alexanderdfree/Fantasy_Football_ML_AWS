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
- ``team_week_stats_release`` is the one wrapper that reads an nflverse release
  parquet *directly* (not via an ``nflreadpy`` loader): DST needs the full
  team-week defensive/ST schema the player-stats loaders don't expose. It lives
  here anyway so every nflverse fetch stays centralized + monkeypatchable. (#397)
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
    # Two-point-conversion plays sit at yardline_100==2 (inside the 20) and carry
    # a rusher/receiver id + pass_attempt/play_type=="run", so without this column
    # they inflate red-zone carries/targets even though 2pt conversions are scored
    # separately from the rushing/receiving TDs these features predict (#369 F28).
    "two_point_attempt",
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


def _native_int_seasons(seasons: list[int]) -> list[int]:
    """Coerce a season list to native Python ints for the nflreadpy boundary.

    nflreadpy 0.1.5's season-taking loaders (``load_rosters`` / ``load_pbp`` /
    ``load_snap_counts`` / ``load_injuries`` / ``load_rosters_weekly`` /
    ``load_player_stats`` / ...) validate each season with a strict
    ``not isinstance(season, int)`` check that REJECTS numpy integers — a
    DataFrame-derived list (``df["season"].astype(int).unique()`` yields
    ``numpy.int64``) trips it with a misleading "Season must be between <lo> and
    <hi>" even for an in-range year. Normalizing the season type here is this
    boundary adapter's job (mirrors the Polars→pandas reconciliation it already
    owns) and is numerically inert: ``int(2025)`` and ``int(np.int64(2025))`` are
    the same value, so the fetched data — and any model trained on it — is
    byte-identical. The serving expert join hit this exact trap (the NFL.com
    Season Leaders column served all-null); coercing at the boundary protects
    every caller, not just that one. See ``todo/fixed-archive.md``.
    """
    return [int(s) for s in seasons]


def weekly_data(seasons: list[int]) -> pd.DataFrame:
    df = _to_pandas(
        _nflreadpy.load_player_stats(_native_int_seasons(seasons), summary_level="week")
    )
    return df.rename(columns=_WEEKLY_RENAME)


def rosters(seasons: list[int]) -> pd.DataFrame:
    df = _to_pandas(_nflreadpy.load_rosters(_native_int_seasons(seasons)))
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df["player_id"] = df["gsis_id"]
    return df


def rosters_weekly(seasons: list[int]) -> pd.DataFrame:
    """Per-(player, season, week) roster snapshots with weekly ``status``.

    Distinct from :func:`rosters` (``load_rosters``), which is one row per
    player-season: the SEASONAL frame also carries ``week``/``status`` columns,
    so a column-presence guard cannot tell them apart — only this weekly frame
    actually has a row (and a status: ACT/RES/INA/...) for every rostered week.
    The inheritance vacancy out-set (#1106 findings A/INA) must use this one;
    building it from the seasonal frame silently registers ~no vacancies.
    """
    df = _to_pandas(_nflreadpy.load_rosters_weekly(_native_int_seasons(seasons)))
    if "player_id" not in df.columns and "gsis_id" in df.columns:
        df["player_id"] = df["gsis_id"]
    return df


def schedules(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_schedules(_native_int_seasons(seasons)))


def snap_counts(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_snap_counts(_native_int_seasons(seasons)))


def injuries(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_injuries(_native_int_seasons(seasons)))


def depth_charts(seasons: list[int]) -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_depth_charts(_native_int_seasons(seasons)))


def player_ids() -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_ff_playerids())


def teams() -> pd.DataFrame:
    return _to_pandas(_nflreadpy.load_teams())


def team_week_stats_release(season: int) -> pd.DataFrame:
    """One season of team-week stats from the nflverse ``stats_team`` release.

    The one wrapper that reads a release parquet *directly* rather than via an
    ``nflreadpy`` loader: DST's ``build_data`` depends on the full team-week
    defensive/ST schema (``def_tds``/``def_safeties``/``def_fumbles_forced``/
    ``fg_blocked``/``pat_blocked``/...) the player-stats loaders don't expose. It
    lives here so every nflverse fetch stays centralized and monkeypatchable in
    tests (the loader read this URL inline before). (#397)
    """
    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_team/stats_team_week_{season}.parquet"
    )
    return pd.read_parquet(url)


def pbp_data(seasons: list[int], cols: tuple[str, ...]) -> pd.DataFrame:
    df = _nflreadpy.load_pbp(_native_int_seasons(seasons))
    available = [c for c in cols if c in df.columns]
    return _to_pandas(df.select(available))


def ff_opportunity(seasons: list[int]) -> pd.DataFrame:
    """ff_opportunity expected-points per player-game (ffverse ``ep_weekly``
    model). gsis-keyed (``player_id``); ``*_exp`` columns are the modeled
    expected stats the opportunity features read."""
    return _to_pandas(_nflreadpy.load_ff_opportunity(_native_int_seasons(seasons)))


def contracts() -> pd.DataFrame:
    """Historical player contracts (OTC via nflverse). One row per contract,
    all seasons (no season filter at the source); carries ``gsis_id`` +
    ``year_signed`` / ``apy_cap_pct`` / ``guaranteed`` / ``years``."""
    return _to_pandas(_nflreadpy.load_contracts())


# nflreadpy 0.1.5 has no QBR loader, so QBR comes straight from the nflverse
# espnscrapeR-data CSV — the same source nfl_data_py.import_qbr read. ESPN-id
# keyed (bridged to gsis downstream via player_ids()).
_QBR_WEEKLY_URL = (
    "https://raw.githubusercontent.com/nflverse/espnscrapeR-data/master/data/qbr-nfl-weekly.csv"
)


def qbr_weekly(seasons: list[int]) -> pd.DataFrame:
    """Weekly ESPN QBR from the nflverse espnscrapeR-data CSV, filtered to the
    requested season range. Read with pandas directly (not nflreadpy — it has
    no QBR loader); already pandas, so no Polars conversion needed."""
    df = pd.read_csv(_QBR_WEEKLY_URL)
    if seasons and "season" in df.columns:
        df = df[df["season"].between(min(seasons), max(seasons))]
    return df
