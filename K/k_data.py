import pandas as pd
from K.k_config import K_MIN_GAMES
from src.config import CACHE_DIR, SEASONS


def load_kicker_data() -> pd.DataFrame:
    """Load raw kicker data from weekly parquet, merge with schedule info.

    Kicker FG/PAT data is only available for 2025 in the nflverse weekly stats.
    Earlier seasons only have offensive stats for kickers (trick plays).
    """
    weekly = pd.read_parquet(f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet")
    schedules = pd.read_parquet(f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")

    # Filter to K position with actual kicking data, regular season only
    k_df = weekly[
        (weekly["position"] == "K")
        & (weekly["season_type"] == "REG")
    ].copy()
    k_df = k_df[k_df["fg_att"].fillna(0) + k_df["pat_att"].fillna(0) > 0].copy()

    # Apply min-games filter
    games_per_season = k_df.groupby(["player_id", "season"])["week"].transform("count")
    k_df = k_df[games_per_season >= K_MIN_GAMES].copy()

    # Merge schedule info (is_home, Vegas lines)
    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()

    home = schedules_reg[["season", "week", "home_team", "spread_line", "total_line"]].copy()
    home.columns = ["season", "week", "recent_team", "spread_line", "total_line"]
    home["is_home"] = 1
    # Implied team total: (total - spread) / 2 for home team
    home["implied_team_total"] = (home["total_line"] - home["spread_line"]) / 2

    away = schedules_reg[["season", "week", "away_team", "spread_line", "total_line"]].copy()
    away.columns = ["season", "week", "recent_team", "spread_line", "total_line"]
    away["is_home"] = 0
    # Implied team total: (total + spread) / 2 for away team
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

    return k_df


def filter_to_k(df: pd.DataFrame) -> pd.DataFrame:
    """Identity filter — kicker data is pre-filtered."""
    return df.copy()


def kicker_week_split(
    k_df: pd.DataFrame,
    val_weeks: int = 3,
    test_weeks: int = 2,
) -> tuple:
    """Split kicker data by week within the season.

    Since kicker data is single-season (2025), we use within-season
    temporal splits instead of cross-season splits.
    """
    max_week = int(k_df["week"].max())
    test_start = max_week - test_weeks + 1
    val_start = test_start - val_weeks

    train = k_df[k_df["week"] < val_start].copy()
    val = k_df[(k_df["week"] >= val_start) & (k_df["week"] < test_start)].copy()
    test = k_df[k_df["week"] >= test_start].copy()

    print(f"K within-season split (max_week={max_week}):")
    print(f"  Train: weeks 1-{val_start - 1} ({len(train)} rows)")
    print(f"  Val:   weeks {val_start}-{test_start - 1} ({len(val)} rows)")
    print(f"  Test:  weeks {test_start}-{max_week} ({len(test)} rows)")

    return train, val, test
