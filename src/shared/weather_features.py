"""Weather, venue, and Vegas implied-odds features for the Weather NN model.

Merges schedule data onto player DataFrames and computes 12 derived features.
Used by the Weather NN — an exact copy of each position's NN except with
these additional features appended to the input.
"""

import threading

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, SEASONS
from src.data.nflcom_loader import schedule_team_code_normalization

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

WEATHER_FEATURES_ALL = [
    "implied_team_total",
    "implied_opp_total",
    "total_line",
    "is_dome",
    "is_grass",
    "temp_adjusted",
    "wind_adjusted",
    "is_divisional",
    "days_rest_improved",
    "rest_advantage",
    "implied_total_x_wind",
]

# Per-position drops (from docs/archive/design_weather_and_odds.md feature table)
WEATHER_DROPS_BY_POSITION = {
    "QB": {"is_grass"},
    "RB": {"is_dome", "temp_adjusted", "wind_adjusted", "implied_total_x_wind"},
    "WR": {"is_grass"},
    "TE": {"is_grass", "temp_adjusted", "wind_adjusted", "implied_total_x_wind"},
}

# Module-level cache for schedule data. Lock guards concurrent first-read
# races (parallel CV folds / threaded callers) — mirrors the precedent in
# ``feature_cache._lru_lock``.
_schedule_cache = None
_schedule_cache_lock = threading.Lock()

# Map franchise relocations back to their current abbreviations. Player data
# uses the current team codes throughout, but the historical schedule still
# carries the pre-relocation codes (OAK/SD/STL) for those seasons, so a
# direct (season, week, team) join misses every pre-move row without this.
#
# Derived from the canonical base map in ``src.data.nflcom_loader`` (single
# source of truth) via ``schedule_team_code_normalization()``, which encodes
# the one documented join-direction difference: nflverse schedule/weekly data
# canonicalize the Rams to ``"LA"`` (STL -> LA), whereas the NFL.com/roster
# universe uses ``"LAR"``. Resolves to ``{"OAK": "LV", "SD": "LAC", "STL": "LA"}``.
_TEAM_CODE_NORMALIZATION = schedule_team_code_normalization()


# ---------------------------------------------------------------------------
# Schedule loading and merge
# ---------------------------------------------------------------------------


def _load_schedules() -> pd.DataFrame:
    """Load and cache schedule data from the raw parquet.

    Double-checked locking: fast path (already-populated cache) bypasses the
    lock; cold path takes the lock once to serialise the parquet read so
    parallel CV folds don't all read the same file from disk.
    """
    global _schedule_cache
    if _schedule_cache is not None:
        return _schedule_cache

    with _schedule_cache_lock:
        if _schedule_cache is not None:
            return _schedule_cache
        path = f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
        schedules = pd.read_parquet(path)
        schedules = schedules[schedules["game_type"] == "REG"].copy()
        _schedule_cache = schedules
        return schedules


def build_implied_team_total_lookup(schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape a schedule frame into a ``(season, week, recent_team)`` keyed
    lookup of ``implied_team_total``.

    Single source of truth for the implied-total formula — ``spread_line`` is
    from the home perspective, so the home team's implied total is
    ``(total_line - spread_line) / 2`` and the away team's is
    ``(total_line + spread_line) / 2``. Both ``merge_schedule_features`` (full
    weather merge, downstream of ``_build_team_schedule_lookup``) and
    ``src.features.engineer._build_defense_matchup_features`` (the tests' direct
    ``build_features`` path, without the wider weather merge) consume this
    helper so the per-team value can't drift between code paths.

    Only requires ``home_team``, ``away_team``, ``spread_line``, ``total_line``
    in the input — kept minimal so callers using stripped-down schedule frames
    (the autouse fake schedules in ``tests/test_feature_leakage.py``) work
    without enrichment.
    """
    sched = schedules[
        ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    ].copy()
    sched["home_team"] = sched["home_team"].replace(_TEAM_CODE_NORMALIZATION)
    sched["away_team"] = sched["away_team"].replace(_TEAM_CODE_NORMALIZATION)

    # spread_line is from home perspective (negative = home favored).
    home_total = (sched["total_line"] - sched["spread_line"]) / 2
    away_total = (sched["total_line"] + sched["spread_line"]) / 2

    home = pd.DataFrame(
        {
            "season": sched["season"],
            "week": sched["week"],
            "recent_team": sched["home_team"],
            "implied_team_total": home_total,
        }
    )
    away = pd.DataFrame(
        {
            "season": sched["season"],
            "week": sched["week"],
            "recent_team": sched["away_team"],
            "implied_team_total": away_total,
        }
    )
    return pd.concat([home, away], ignore_index=True).drop_duplicates(
        subset=["season", "week", "recent_team"]
    )


def _build_team_schedule_lookup(schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape game-level schedule to team-level rows (home + away)."""
    cols = [
        "season",
        "week",
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "home_rest",
        "away_rest",
        "div_game",
    ]
    sched = schedules[cols + ["home_team", "away_team"]].copy()
    sched["home_team"] = sched["home_team"].replace(_TEAM_CODE_NORMALIZATION)
    sched["away_team"] = sched["away_team"].replace(_TEAM_CODE_NORMALIZATION)

    # Home team rows
    home = sched.copy()
    home["recent_team"] = home["home_team"]
    home["is_home_sched"] = 1
    home["team_rest"] = home["home_rest"]
    home["opp_rest"] = home["away_rest"]
    # spread_line is from home perspective (negative = home favored)
    home["implied_team_total"] = (home["total_line"] - home["spread_line"]) / 2

    # Away team rows
    away = sched.copy()
    away["recent_team"] = away["away_team"]
    away["is_home_sched"] = 0
    away["team_rest"] = away["away_rest"]
    away["opp_rest"] = away["home_rest"]
    away["implied_team_total"] = (away["total_line"] + away["spread_line"]) / 2

    lookup = pd.concat([home, away], ignore_index=True)
    keep = [
        "season",
        "week",
        "recent_team",
        "is_home_sched",
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "team_rest",
        "opp_rest",
        "div_game",
        "implied_team_total",
    ]
    return lookup[keep].drop_duplicates(subset=["season", "week", "recent_team"])


def merge_schedule_features(df: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    """Merge schedule data and compute 12 weather/venue/Vegas features in-place.

    Idempotent: skips if features are already present.

    Args:
        df: Player DataFrame with season, week, recent_team columns.
        label: Optional tag (e.g. "train"/"val"/"test") included in the
            unmatched-rows warning so callers that invoke the merge per-split
            produce distinguishable log lines.

    Returns:
        The same DataFrame with 12 new columns added.
    """
    if "_schedule_merged" in df.columns:
        return df
    # Drop any stale placeholders so the merge produces fresh values.
    # ``spread_line`` and ``div_game`` are NOT in ``WEATHER_FEATURES_ALL`` (they
    # remain as bare-named feature columns), but DST materializes them on
    # every row from the schedule. Without dropping them here they survive
    # into the ``df.merge(lookup, ...)`` below, pandas suffix-renames the
    # collision to ``spread_line_x`` / ``_y`` (and ``div_game_x`` / ``_y``),
    # the bare-name merge-back loop silently skips them, and the cleanup
    # loop drops them — so both columns end up zeroed by the catch-all
    # backfill in ``build_position_features``. Dropping them up front lets
    # the merge produce them cleanly as bare names so the merge-back below
    # populates them. See TODO.md "DST spread_line/div_game zeroed by
    # merge_schedule_features".
    for col in WEATHER_FEATURES_ALL + ["spread_line", "div_game"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    schedules = _load_schedules()
    lookup = _build_team_schedule_lookup(schedules)

    # Merge on (season, week, recent_team)
    n_before = len(df)
    df_merged = df.merge(lookup, on=["season", "week", "recent_team"], how="left")

    # Guard against row duplication from merge
    if len(df_merged) != n_before:
        df_merged = df_merged.drop_duplicates(subset=["player_id", "season", "week"], keep="first")

    # Copy merged columns back into original df (preserve index)
    merge_cols = [
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "team_rest",
        "opp_rest",
        "div_game",
        "is_home_sched",
        "implied_team_total",
    ]
    for col in merge_cols:
        if col in df_merged.columns:
            df[col] = df_merged[col].values

    # --- Vegas features ---
    df["implied_opp_total"] = df["total_line"] - df["implied_team_total"]
    # total_line already present from merge

    # --- Venue features ---
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int) if "roof" in df.columns else 0
    df["is_grass"] = (df["surface"] == "grass").astype(int) if "surface" in df.columns else 0
    df["temp_adjusted"] = np.where(
        df["is_dome"] == 1, 65.0, df["temp"].fillna(65.0) if "temp" in df.columns else 65.0
    )
    df["wind_adjusted"] = np.where(
        df["is_dome"] == 1, 0.0, df["wind"].fillna(0.0) if "wind" in df.columns else 0.0
    )
    df["is_divisional"] = df["div_game"].fillna(0).astype(int) if "div_game" in df.columns else 0

    # --- Rest features ---
    team_rest = df["team_rest"].fillna(7) if "team_rest" in df.columns else 7
    opp_rest = df["opp_rest"].fillna(7) if "opp_rest" in df.columns else 7
    df["days_rest_improved"] = pd.to_numeric(team_rest, errors="coerce").fillna(7).clip(4, 21)
    df["rest_advantage"] = df["days_rest_improved"] - pd.to_numeric(
        opp_rest, errors="coerce"
    ).fillna(7)

    # --- Interaction features ---
    # Preserve NaN from implied_team_total so unmatched games stay NaN rather
    # than silently becoming 0 (matches the docstring above and the NaN-kept
    # guarantee for implied_opp_total).
    wind_factor = (1 - df["wind_adjusted"] / 40).clip(0, 1)
    df["implied_total_x_wind"] = df["implied_team_total"] * wind_factor

    # NaN Vegas features indicate unmatched games — leave them as NaN
    # so downstream code can detect and handle them explicitly.
    n_missing = df["implied_team_total"].isna().sum()
    if n_missing > 0:
        tag = f" [{label}]" if label else ""
        print(f"  WARNING:{tag} {n_missing} rows have no schedule match (Vegas features are NaN)")

    # Populate is_home from the schedule merge (before cleanup deletes is_home_sched)
    if "is_home_sched" in df.columns:
        df["is_home"] = df["is_home_sched"].fillna(0).astype(int)

    # Clean up intermediate merge columns. ``spread_line`` and ``div_game``
    # are intentionally NOT dropped: DST carries them as bare-named feature
    # columns and the schedule merge above is what populates them. Stripping
    # them here would silently zero both features (DST trained for months
    # with ``spread_line == 0`` and ``div_game == 0`` because of this drop).
    for col in [
        "roof",
        "surface",
        "temp",
        "wind",
        "team_rest",
        "opp_rest",
        "is_home_sched",
    ]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    df["_schedule_merged"] = True
    return df


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------


def get_weather_feature_columns(position: str, base_cols: list[str]) -> list[str]:
    """Return base feature columns plus position-appropriate weather features.

    This enforces the Weather NN invariant: same base features as the regular
    NN, with weather/venue/Vegas features appended.

    Args:
        position: Position abbreviation (QB, RB, WR, TE).
        base_cols: Feature columns from the regular NN's get_feature_columns_fn.

    Returns:
        Extended feature list = base_cols + weather features (minus position drops).
    """
    drops = WEATHER_DROPS_BY_POSITION.get(position, set())
    weather_cols = [c for c in WEATHER_FEATURES_ALL if c not in drops]
    # Avoid duplicates if any weather feature is already in base_cols
    weather_cols = [c for c in weather_cols if c not in base_cols]
    return base_cols + weather_cols
