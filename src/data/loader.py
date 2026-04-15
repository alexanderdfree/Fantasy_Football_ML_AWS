import os
import pandas as pd
import nfl_data_py as nfl
from src.config import (
    SCORING, SCORING_STANDARD, SCORING_HALF_PPR, SCORING_PPR,
    PPR_FORMATS, CACHE_DIR, SEASONS,
)


def load_raw_data(seasons: list[int] = None, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load and merge NFL weekly data, rosters, snap counts, and schedules."""
    if seasons is None:
        seasons = SEASONS

    os.makedirs(cache_dir, exist_ok=True)
    weekly_path = f"{cache_dir}/weekly_{seasons[0]}_{seasons[-1]}.parquet"
    rosters_path = f"{cache_dir}/rosters_{seasons[0]}_{seasons[-1]}.parquet"
    schedules_path = f"{cache_dir}/schedules_{seasons[0]}_{seasons[-1]}.parquet"
    snap_path = f"{cache_dir}/snap_counts_{seasons[0]}_{seasons[-1]}.parquet"

    # 1. Weekly player stats
    if os.path.exists(weekly_path):
        weekly = pd.read_parquet(weekly_path)
    else:
        # nfl_data_py only has player_stats up to 2024; 2025+ uses nflverse stats_player
        old_seasons = [s for s in seasons if s <= 2024]
        new_seasons = [s for s in seasons if s >= 2025]
        parts = []
        if old_seasons:
            parts.append(nfl.import_weekly_data(old_seasons))
        for s in new_seasons:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{s}.parquet"
            df_new = pd.read_parquet(url)
            # Harmonise column names to match old format
            df_new = df_new.rename(columns={
                "team": "recent_team",
                "passing_interceptions": "interceptions",
                "sacks_suffered": "sacks",
                "sack_yards_lost": "sack_yards",
            })
            parts.append(df_new)
        weekly = pd.concat(parts, ignore_index=True)
        weekly.to_parquet(weekly_path)

    # 2. Roster data for reliable position labels
    if os.path.exists(rosters_path):
        rosters = pd.read_parquet(rosters_path)
    else:
        rosters = nfl.import_seasonal_rosters(seasons)
        # Coerce mixed-type columns that break parquet serialization
        # (e.g. jersey_number is str in older seasons, int in newer)
        for col in rosters.columns:
            if rosters[col].dtype == object:
                rosters[col] = rosters[col].astype(str)
        rosters.to_parquet(rosters_path)

    # 3. Schedule data for opponent info
    if os.path.exists(schedules_path):
        schedules = pd.read_parquet(schedules_path)
    else:
        schedules = nfl.import_schedules(seasons)
        schedules.to_parquet(schedules_path)

    # 4. Snap counts (only available from 2012+)
    if os.path.exists(snap_path):
        snap_counts = pd.read_parquet(snap_path)
    else:
        snap_seasons = [s for s in seasons if s >= 2012]
        snap_counts = nfl.import_snap_counts(snap_seasons) if snap_seasons else pd.DataFrame()
        snap_counts.to_parquet(snap_path)

    # --- Merge rosters for position override ---
    roster_pos = rosters[["player_id", "season", "position"]].drop_duplicates(
        subset=["player_id", "season"]
    )
    weekly = weekly.merge(
        roster_pos, on=["player_id", "season"], how="left", suffixes=("_weekly", "")
    )
    if "position_weekly" in weekly.columns:
        weekly["position"] = weekly["position"].fillna(weekly["position_weekly"])
        weekly.drop(columns=["position_weekly"], inplace=True)

    # --- Merge snap counts via ID bridge ---
    try:
        ids = nfl.import_ids()
        pfr_to_gsis = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()
        snap_counts = snap_counts.merge(
            pfr_to_gsis, left_on="pfr_player_id", right_on="pfr_id", how="left"
        )
        snap_merged = snap_counts[["gsis_id", "season", "week", "offense_pct"]].dropna(
            subset=["gsis_id"]
        )
        weekly = weekly.merge(
            snap_merged,
            left_on=["player_id", "season", "week"],
            right_on=["gsis_id", "season", "week"],
            how="left",
        )
        if "gsis_id" in weekly.columns:
            weekly.drop(columns=["gsis_id"], inplace=True)
        weekly.rename(columns={"offense_pct": "snap_pct"}, inplace=True)
    except Exception as e:
        print(f"WARNING: Snap count merge failed ({e}), snap_pct will be NaN")
        if "snap_pct" not in weekly.columns:
            weekly["snap_pct"] = float("nan")

    # 5. Injury reports
    injury_path = f"{cache_dir}/injuries_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(injury_path):
        injuries = pd.read_parquet(injury_path)
    else:
        injuries = nfl.import_injuries(seasons)
        injuries.to_parquet(injury_path)

    practice_map = {
        "Full Participation in Practice": 2,
        "Limited Participation in Practice": 1,
        "Did Not Participate In Practice": 0,
    }
    injuries["practice_status_num"] = injuries["practice_status"].map(practice_map)

    status_map = {"Questionable": 0.5, "Doubtful": 0.1, "Out": 0.0}
    injuries["game_status_num"] = injuries["report_status"].map(status_map).fillna(1.0)

    # Worst practice/game status per player-week (multiple injuries possible)
    inj_agg = injuries.groupby(["gsis_id", "season", "week"]).agg(
        practice_status=("practice_status_num", "min"),
        game_status=("game_status_num", "min"),
    ).reset_index()

    weekly = weekly.merge(
        inj_agg,
        left_on=["player_id", "season", "week"],
        right_on=["gsis_id", "season", "week"],
        how="left",
    )
    weekly.drop(columns=["gsis_id"], errors="ignore", inplace=True)
    weekly["practice_status"] = weekly["practice_status"].fillna(2.0)
    weekly["game_status"] = weekly["game_status"].fillna(1.0)

    # 6. Depth charts (Offense formation, most recent entry per player-week)
    depth_path = f"{cache_dir}/depth_charts_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(depth_path):
        depth = pd.read_parquet(depth_path)
    else:
        depth = nfl.import_depth_charts(seasons)
        depth.to_parquet(depth_path)

    depth_off = depth[depth["formation"] == "Offense"].copy()
    depth_off["depth_team"] = pd.to_numeric(depth_off["depth_team"], errors="coerce")
    # One row per player-week: take the most recent (last) entry per week
    depth_off = depth_off.sort_values(["gsis_id", "season", "week"])
    depth_agg = depth_off.groupby(["gsis_id", "season", "week"]).agg(
        depth_chart_rank=("depth_team", "last"),
    ).reset_index()

    weekly = weekly.merge(
        depth_agg,
        left_on=["player_id", "season", "week"],
        right_on=["gsis_id", "season", "week"],
        how="left",
    )
    weekly.drop(columns=["gsis_id"], errors="ignore", inplace=True)
    weekly["depth_chart_rank"] = weekly["depth_chart_rank"].fillna(3).clip(upper=3)

    # Store schedules for later use
    weekly.attrs["schedules"] = schedules

    return weekly


def compute_fantasy_points(df: pd.DataFrame, scoring: dict = None) -> pd.Series:
    """Compute fantasy points from raw stat columns for a given scoring dict."""
    if scoring is None:
        scoring = SCORING

    col_map = {
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "interceptions",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
    }

    fantasy_points = pd.Series(0.0, index=df.index)
    for key, weight in scoring.items():
        if key == "fumbles_lost":
            val = (df["sack_fumbles_lost"].fillna(0)
                   + df["rushing_fumbles_lost"].fillna(0)
                   + df["receiving_fumbles_lost"].fillna(0))
        else:
            val = df[col_map[key]].fillna(0)
        fantasy_points += val * weight
    return fantasy_points


def compute_all_scoring_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fantasy points for standard, half-PPR, and full PPR formats.

    Adds columns: fantasy_points_standard, fantasy_points_half_ppr, fantasy_points
    """
    df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)
    df["fantasy_points"] = compute_fantasy_points(df, SCORING_PPR)
    return df


def compute_fantasy_points_floor(df: pd.DataFrame, ppr_weight: float = 1.0) -> pd.Series:
    """Yardage + receptions only (no TDs, no turnovers)."""
    return (
        df["passing_yards"].fillna(0) * 0.04
        + df["rushing_yards"].fillna(0) * 0.1
        + df["receiving_yards"].fillna(0) * 0.1
        + df["receptions"].fillna(0) * ppr_weight
    )


def compute_all_floor_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fantasy points floor for all three scoring formats.

    Adds columns: fantasy_points_floor_standard, fantasy_points_floor_half_ppr,
    fantasy_points_floor
    """
    df["fantasy_points_floor_standard"] = compute_fantasy_points_floor(df, ppr_weight=0.0)
    df["fantasy_points_floor_half_ppr"] = compute_fantasy_points_floor(df, ppr_weight=0.5)
    df["fantasy_points_floor"] = compute_fantasy_points_floor(df, ppr_weight=1.0)
    return df
