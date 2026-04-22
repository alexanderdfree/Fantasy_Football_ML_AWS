import pandas as pd
from src.config import POSITIONS
from src.data.loader import compute_all_scoring_formats


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean raw NFL data for modeling."""
    df = raw_df.copy()

    # Filter to regular season
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    # Step 1: Filter to skill positions
    df = df[df["position"].isin(POSITIONS)].copy()

    # Step 2: Remove rows where player didn't play
    stat_cols = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "targets",
        "carries",
        "completions",
        "attempts",
    ]
    existing_stat_cols = [c for c in stat_cols if c in df.columns]
    all_zero = df[existing_stat_cols].fillna(0).sum(axis=1) == 0
    no_snaps = (
        df["snap_pct"].isna() if "snap_pct" in df.columns else pd.Series(True, index=df.index)
    )
    df = df[~(all_zero & no_snaps)].copy()

    # Step 3: Fill missing stat columns with 0
    fill_zero_cols = [
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "carries",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "targets",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "attempts",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "rushing_first_downs",
        "receiving_first_downs",
        "rushing_epa",
        "receiving_epa",
        "receiving_yards_after_catch",
        "receiving_air_yards",
        "passing_first_downs",
        "passing_yards_after_catch",
        "sack_yards",
        "special_teams_tds",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # Step 4: Fill missing snap_pct with position-week median
    if "snap_pct" in df.columns:
        medians = df.groupby(["position", "week"])["snap_pct"].transform("median")
        df["snap_pct"] = df["snap_pct"].fillna(medians)
        df["snap_pct"] = df["snap_pct"].fillna(0)

    # Step 5: Compute fantasy points for all scoring formats
    df = compute_all_scoring_formats(df)

    # Validate against nflverse pre-computed
    if "fantasy_points_ppr" in df.columns:
        discrepancy = (df["fantasy_points_ppr"] - df["fantasy_points"]).abs()
        n_mismatch = (discrepancy > 0.5).sum()
        if n_mismatch > 0:
            print(f"WARNING: {n_mismatch} rows differ from nflverse PPR points by > 0.5")

    return df
