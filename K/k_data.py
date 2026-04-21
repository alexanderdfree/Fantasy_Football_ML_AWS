import os

import nfl_data_py as nfl
import pandas as pd

from K.k_config import K_MIN_GAMES, K_SEASONS
from src.config import CACHE_DIR, SEASONS

# ---------------------------------------------------------------------------
# PBP-based kicker reconstruction (2015-2024)
# ---------------------------------------------------------------------------


def reconstruct_kicker_weekly_from_pbp(
    seasons: list[int],
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Reconstruct weekly kicker stats from play-by-play data.

    PBP has FG/XP play-level data from 1999+, while import_weekly_data()
    only has kicker columns starting in 2025. This function aggregates PBP
    plays into a weekly kicker-level dataframe matching the schema expected
    by k_targets.py, plus additional PBP-derived columns for features.
    """
    cache_path = f"{cache_dir}/kicker_pbp_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    all_weekly = []
    for yr in seasons:
        print(f"  Loading PBP for {yr}...")
        pbp = nfl.import_pbp_data([yr], downcast=True)
        # Keep only regular season
        pbp = pbp[pbp["season_type"] == "REG"]

        # --- Field goals ---
        fg = pbp[pbp["field_goal_attempt"] == 1].copy()
        fg["fg_made_flag"] = (fg["field_goal_result"] == "made").astype(int)
        fg["fg_missed_flag"] = (fg["field_goal_result"] != "made").astype(int)
        # Distance buckets matching weekly_data schema
        d = fg["kick_distance"]
        made = fg["fg_made_flag"].astype(bool)
        fg["fg_made_0_19"] = ((d < 20) & made).astype(int)
        fg["fg_made_20_29"] = ((d >= 20) & (d < 30) & made).astype(int)
        fg["fg_made_30_39"] = ((d >= 30) & (d < 40) & made).astype(int)
        fg["fg_made_40_49"] = ((d >= 40) & (d < 50) & made).astype(int)
        fg["fg_made_50_59"] = ((d >= 50) & (d < 60) & made).astype(int)
        fg["fg_made_60_"] = ((d >= 60) & made).astype(int)
        # Missed buckets
        missed = fg["fg_missed_flag"].astype(bool)
        fg["fg_missed_40_49"] = ((d >= 40) & (d < 50) & missed).astype(int)
        fg["fg_missed_50_59"] = ((d >= 50) & (d < 60) & missed).astype(int)
        fg["fg_missed_60_"] = ((d >= 60) & missed).astype(int)
        # PBP-only situational flags
        fg["is_clutch"] = (fg["score_differential"].abs() <= 7).astype(int)
        fg["clutch_made"] = (fg["is_clutch"] & fg["fg_made_flag"]).astype(int)
        fg["is_q4"] = (fg["qtr"] >= 4).astype(int)
        fg["q4_made"] = (fg["is_q4"] & fg["fg_made_flag"]).astype(int)
        fg["is_long"] = (d >= 40).astype(int)
        fg["long_made"] = (fg["is_long"] & fg["fg_made_flag"]).astype(int)
        # Sum of kick_distance restricted to made FGs — per-attempt contribution
        # to the `fg_yards_made` season-week aggregate consumed by k_targets.
        fg["_fg_yards_made_flag"] = fg["fg_made_flag"] * fg["kick_distance"]

        weekly_fg = (
            fg.groupby(["kicker_player_id", "kicker_player_name", "posteam", "season", "week"])
            .agg(
                fg_att=("fg_made_flag", "count"),
                fg_made=("fg_made_flag", "sum"),
                fg_missed=("fg_missed_flag", "sum"),
                fg_made_0_19=("fg_made_0_19", "sum"),
                fg_made_20_29=("fg_made_20_29", "sum"),
                fg_made_30_39=("fg_made_30_39", "sum"),
                fg_made_40_49=("fg_made_40_49", "sum"),
                fg_made_50_59=("fg_made_50_59", "sum"),
                fg_made_60_=("fg_made_60_", "sum"),
                fg_missed_40_49=("fg_missed_40_49", "sum"),
                fg_missed_50_59=("fg_missed_50_59", "sum"),
                fg_missed_60_=("fg_missed_60_", "sum"),
                fg_yards_made=("_fg_yards_made_flag", "sum"),
                # PBP-derived aggregates
                avg_fg_distance=("kick_distance", "mean"),
                max_fg_distance=("kick_distance", "max"),
                avg_fg_prob=("fg_prob", "mean"),
                clutch_fg_att=("is_clutch", "sum"),
                clutch_fg_made=("clutch_made", "sum"),
                q4_fg_att=("is_q4", "sum"),
                q4_fg_made=("q4_made", "sum"),
                long_fg_att=("is_long", "sum"),
                long_fg_made=("long_made", "sum"),
                # Weather (game-level, same for all plays in a game)
                game_wind=("wind", "first"),
                game_temp=("temp", "first"),
                roof=("roof", "first"),
                surface=("surface", "first"),
            )
            .reset_index()
        )

        # --- Extra points ---
        xp = pbp[pbp["extra_point_attempt"] == 1].copy()
        xp["xp_made"] = (xp["extra_point_result"] == "good").astype(int)
        xp["xp_missed"] = (xp["extra_point_result"] != "good").astype(int)

        weekly_xp = (
            xp.groupby(["kicker_player_id", "season", "week"])
            .agg(
                pat_att=("xp_made", "count"),
                pat_made=("xp_made", "sum"),
                pat_missed=("xp_missed", "sum"),
            )
            .reset_index()
        )

        # Merge FG + XP
        weekly_k = weekly_fg.merge(
            weekly_xp, on=["kicker_player_id", "season", "week"], how="outer"
        )
        all_weekly.append(weekly_k)

    result = pd.concat(all_weekly, ignore_index=True)

    # Fill NaN for kicker-weeks with only FGs or only XPs
    for col in result.columns:
        if col not in [
            "kicker_player_id",
            "kicker_player_name",
            "posteam",
            "season",
            "week",
            "roof",
            "surface",
        ]:
            result[col] = result[col].fillna(0)

    # Rename to match expected pipeline columns
    result.rename(
        columns={
            "kicker_player_id": "player_id",
            "kicker_player_name": "player_name",
            "posteam": "recent_team",
        },
        inplace=True,
    )
    result["position"] = "K"
    result["season_type"] = "REG"

    # Derive venue features
    result["is_dome"] = result["roof"].isin(["dome", "closed"]).astype(int)

    os.makedirs(cache_dir, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"  Cached PBP kicker data: {len(result)} rows -> {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_kicker_data() -> pd.DataFrame:
    """Load kicker data combining PBP reconstruction (2015-2024) + weekly (2025).

    Merges schedule info for Vegas lines and home/away.
    """
    pbp_seasons = [s for s in K_SEASONS if s <= 2024]
    weekly_seasons = [s for s in K_SEASONS if s >= 2025]

    parts = []

    # --- PBP-reconstructed data (2015-2024) ---
    if pbp_seasons:
        print("Reconstructing kicker weekly stats from PBP...")
        pbp_df = reconstruct_kicker_weekly_from_pbp(pbp_seasons)
        parts.append(pbp_df)

    # --- Existing weekly data for 2025+ ---
    if weekly_seasons:
        weekly = pd.read_parquet(f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet")
        k_weekly = weekly[
            (weekly["position"] == "K")
            & (weekly["season_type"] == "REG")
            & (weekly["season"].isin(weekly_seasons))
        ].copy()
        k_weekly = k_weekly[k_weekly["fg_att"].fillna(0) + k_weekly["pat_att"].fillna(0) > 0].copy()
        # Add PBP-derived columns with NaN (will be filled later)
        for col in [
            "avg_fg_distance",
            "max_fg_distance",
            "avg_fg_prob",
            "clutch_fg_att",
            "clutch_fg_made",
            "q4_fg_att",
            "q4_fg_made",
            "long_fg_att",
            "long_fg_made",
            "game_wind",
            "game_temp",
            "roof",
            "surface",
            "is_dome",
            "fg_yards_made",
        ]:
            if col not in k_weekly.columns:
                k_weekly[col] = float("nan")
        parts.append(k_weekly)

    k_df = pd.concat(parts, ignore_index=True)

    # --- Backfill PBP-derived columns for 2025 from PBP ---
    if weekly_seasons:
        _backfill_2025_pbp_columns(k_df, weekly_seasons)

    # Apply min-games filter
    games_per_season = k_df.groupby(["player_id", "season"])["week"].transform("count")
    k_df = k_df[games_per_season >= K_MIN_GAMES].copy()

    # --- Merge schedule info (is_home, Vegas lines) ---
    schedules = pd.read_parquet(f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")
    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()

    home = schedules_reg[["season", "week", "home_team", "spread_line", "total_line"]].copy()
    home.columns = ["season", "week", "recent_team", "spread_line", "total_line"]
    home["is_home"] = 1
    home["implied_team_total"] = (home["total_line"] - home["spread_line"]) / 2

    away = schedules_reg[["season", "week", "away_team", "spread_line", "total_line"]].copy()
    away.columns = ["season", "week", "recent_team", "spread_line", "total_line"]
    away["is_home"] = 0
    away["implied_team_total"] = (away["total_line"] + away["spread_line"]) / 2

    schedule_info = pd.concat([home, away], ignore_index=True)
    schedule_info.drop(columns=["spread_line"], inplace=True)

    k_df = k_df.merge(schedule_info, on=["recent_team", "season", "week"], how="left")

    # Fill missing Vegas lines with season medians
    for col in ["total_line", "implied_team_total"]:
        median_val = k_df[col].median()
        k_df[col] = k_df[col].fillna(median_val)
    if "is_home" not in k_df.columns or k_df["is_home"].isna().any():
        k_df["is_home"] = k_df["is_home"].fillna(0)

    print(
        f"  Kicker data: {len(k_df)} rows, {k_df['player_id'].nunique()} kickers, "
        f"seasons {int(k_df['season'].min())}-{int(k_df['season'].max())}"
    )

    return k_df


def _backfill_2025_pbp_columns(k_df: pd.DataFrame, seasons: list[int]) -> None:
    """Backfill PBP-derived columns for 2025 rows from PBP data."""
    mask = k_df["season"].isin(seasons)
    if not mask.any():
        return

    backfill_cols = [
        "avg_fg_distance",
        "max_fg_distance",
        "avg_fg_prob",
        "clutch_fg_att",
        "clutch_fg_made",
        "q4_fg_att",
        "q4_fg_made",
        "long_fg_att",
        "long_fg_made",
        "game_wind",
        "game_temp",
        "roof",
        "surface",
        "is_dome",
        "fg_yards_made",
    ]

    try:
        all_weekly = []
        for yr in seasons:
            pbp = nfl.import_pbp_data([yr], downcast=True)
            pbp = pbp[pbp["season_type"] == "REG"]

            fg = pbp[pbp["field_goal_attempt"] == 1].copy()
            d = fg["kick_distance"]
            fg["fg_made_flag"] = (fg["field_goal_result"] == "made").astype(int)
            fg["is_clutch"] = (fg["score_differential"].abs() <= 7).astype(int)
            fg["clutch_made"] = (fg["is_clutch"] & fg["fg_made_flag"]).astype(int)
            fg["is_q4"] = (fg["qtr"] >= 4).astype(int)
            fg["q4_made"] = (fg["is_q4"] & fg["fg_made_flag"]).astype(int)
            fg["is_long"] = (d >= 40).astype(int)
            fg["long_made"] = (fg["is_long"] & fg["fg_made_flag"]).astype(int)
            # Sum of kick_distance restricted to made FGs — mirrors the
            # historical reconstruction so `fg_yard_points` target is
            # available for 2025 rows as well.
            fg["_fg_yards_made_flag"] = fg["fg_made_flag"] * fg["kick_distance"]

            weekly_pbp = (
                fg.groupby(["kicker_player_id", "season", "week"])
                .agg(
                    avg_fg_distance=("kick_distance", "mean"),
                    max_fg_distance=("kick_distance", "max"),
                    avg_fg_prob=("fg_prob", "mean"),
                    clutch_fg_att=("is_clutch", "sum"),
                    clutch_fg_made=("clutch_made", "sum"),
                    q4_fg_att=("is_q4", "sum"),
                    q4_fg_made=("q4_made", "sum"),
                    long_fg_att=("is_long", "sum"),
                    long_fg_made=("long_made", "sum"),
                    game_wind=("wind", "first"),
                    game_temp=("temp", "first"),
                    roof=("roof", "first"),
                    surface=("surface", "first"),
                    fg_yards_made=("_fg_yards_made_flag", "sum"),
                )
                .reset_index()
            )
            weekly_pbp["is_dome"] = weekly_pbp["roof"].isin(["dome", "closed"]).astype(int)
            weekly_pbp.rename(columns={"kicker_player_id": "player_id"}, inplace=True)
            all_weekly.append(weekly_pbp)

        pbp_all = pd.concat(all_weekly, ignore_index=True)
        key = ["player_id", "season", "week"]
        # roof/surface are initialized as float NaN upstream; cast to object so
        # DataFrame.update can write the string values pulled from PBP.
        for str_col in ("roof", "surface"):
            if k_df[str_col].dtype != object:
                k_df[str_col] = k_df[str_col].astype(object)
        # DataFrame.update aligns on index and overwrites non-NaN values from the source.
        # Wrap in try/finally so a failure inside update() can't leave k_df stuck
        # with the composite index (which would break downstream groupby calls).
        k_df.set_index(key, inplace=True)
        try:
            k_df.update(pbp_all.set_index(key)[backfill_cols])
        finally:
            k_df.reset_index(inplace=True)
    except Exception as e:
        print(f"  WARNING: 2025 PBP backfill failed ({e}), PBP features will be NaN for 2025")


def filter_to_k(df: pd.DataFrame) -> pd.DataFrame:
    """Identity filter — kicker data is pre-filtered."""
    return df.copy()


def kicker_season_split(k_df: pd.DataFrame) -> tuple:
    """Split kicker data by season (cross-season, matching other positions).

    Train: 2015-2023, Val: 2024, Test: 2025
    """
    train = k_df[k_df["season"] <= 2023].copy()
    val = k_df[k_df["season"] == 2024].copy()
    test = k_df[k_df["season"] == 2025].copy()

    print("K cross-season split:")
    print(f"  Train: {int(train['season'].min())}-{int(train['season'].max())} ({len(train)} rows)")
    print(f"  Val:   2024 ({len(val)} rows)")
    print(f"  Test:  2025 ({len(test)} rows)")

    return train, val, test
