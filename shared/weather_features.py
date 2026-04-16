"""Weather, venue, and Vegas implied-odds features for the Weather NN model.

Merges schedule data onto player DataFrames and computes 12 derived features.
Used by the Weather NN — an exact copy of each position's NN except with
these additional features appended to the input.
"""

import os
import numpy as np
import pandas as pd
from src.config import CACHE_DIR, SEASONS

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

# Per-position drops (from docs/design_weather_and_odds.md feature table)
WEATHER_DROPS_BY_POSITION = {
    "QB": {"is_grass"},
    "RB": {"is_dome", "temp_adjusted", "wind_adjusted",
           "implied_total_x_wind"},
    "WR": {"is_grass"},
    "TE": {"is_grass", "temp_adjusted", "wind_adjusted", "implied_total_x_wind"},
}

# Module-level cache for schedule data
_schedule_cache = None


# ---------------------------------------------------------------------------
# Schedule loading and merge
# ---------------------------------------------------------------------------

def _load_schedules() -> pd.DataFrame:
    """Load and cache schedule data from the raw parquet."""
    global _schedule_cache
    if _schedule_cache is not None:
        return _schedule_cache

    path = f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    schedules = pd.read_parquet(path)
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    _schedule_cache = schedules
    return schedules


def _build_team_schedule_lookup(schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape game-level schedule to team-level rows (home + away)."""
    cols = ["season", "week", "spread_line", "total_line",
            "roof", "surface", "temp", "wind",
            "home_rest", "away_rest", "div_game"]
    sched = schedules[cols + ["home_team", "away_team"]].copy()

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
    keep = ["season", "week", "recent_team", "is_home_sched",
            "spread_line", "total_line", "roof", "surface", "temp", "wind",
            "team_rest", "opp_rest", "div_game", "implied_team_total"]
    return lookup[keep].drop_duplicates(subset=["season", "week", "recent_team"])


def merge_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge schedule data and compute 12 weather/venue/Vegas features in-place.

    Idempotent: skips if features are already present.

    Args:
        df: Player DataFrame with season, week, recent_team columns.

    Returns:
        The same DataFrame with 12 new columns added.
    """
    if "_schedule_merged" in df.columns:
        return df
    # Drop any stale placeholders so the merge produces fresh values
    for col in WEATHER_FEATURES_ALL + ["implied_total_x_dome"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    schedules = _load_schedules()
    lookup = _build_team_schedule_lookup(schedules)

    # Merge on (season, week, recent_team)
    n_before = len(df)
    df_merged = df.merge(lookup, on=["season", "week", "recent_team"], how="left")

    # Guard against row duplication from merge
    if len(df_merged) != n_before:
        df_merged = df_merged.drop_duplicates(
            subset=["player_id", "season", "week"], keep="first"
        )

    # Copy merged columns back into original df (preserve index)
    merge_cols = ["spread_line", "total_line", "roof", "surface", "temp", "wind",
                  "team_rest", "opp_rest", "div_game", "is_home_sched",
                  "implied_team_total"]
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
        df["is_dome"] == 1, 65.0,
        df["temp"].fillna(65.0) if "temp" in df.columns else 65.0
    )
    df["wind_adjusted"] = np.where(
        df["is_dome"] == 1, 0.0,
        df["wind"].fillna(0.0) if "wind" in df.columns else 0.0
    )
    df["is_divisional"] = df["div_game"].fillna(0).astype(int) if "div_game" in df.columns else 0

    # --- Rest features ---
    team_rest = df["team_rest"].fillna(7) if "team_rest" in df.columns else 7
    opp_rest = df["opp_rest"].fillna(7) if "opp_rest" in df.columns else 7
    df["days_rest_improved"] = pd.to_numeric(team_rest, errors="coerce").fillna(7).clip(4, 21)
    df["rest_advantage"] = df["days_rest_improved"] - pd.to_numeric(opp_rest, errors="coerce").fillna(7)

    # --- Interaction features ---
    wind_factor = (1 - df["wind_adjusted"] / 40).clip(0, 1)
    df["implied_total_x_wind"] = df["implied_team_total"].fillna(0) * wind_factor

    # NaN Vegas features indicate unmatched games — leave them as NaN
    # so downstream code can detect and handle them explicitly.
    n_missing = df["implied_team_total"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rows have no schedule match (Vegas features are NaN)")

    # Populate is_home from the schedule merge (before cleanup deletes is_home_sched)
    if "is_home_sched" in df.columns:
        df["is_home"] = df["is_home_sched"].fillna(0).astype(int)

    # Clean up intermediate merge columns
    for col in ["spread_line", "roof", "surface", "temp", "wind",
                "team_rest", "opp_rest", "div_game", "is_home_sched"]:
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
