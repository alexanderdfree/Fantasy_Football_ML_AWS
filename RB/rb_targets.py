import pandas as pd

from src.config import PPR_FORMATS


def compute_rb_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 prediction targets + fumble penalty for RB rows.

    Computes targets for all scoring formats (standard, half_ppr, ppr).
    The receiving_floor varies by format; rushing_floor and td_points are the same.

    Note: td_points intentionally excludes 2pt conversions to align with
    SCORING_PPR used by compute_fantasy_points().

    Args:
        df: DataFrame filtered to RB rows only, with raw stat columns available.

    Returns:
        df with added columns per format:
          - rushing_floor (same across formats)
          - receiving_floor_standard, receiving_floor_half_ppr, receiving_floor
          - td_points (same across formats)
          - fumble_penalty (same across formats)
          - fantasy_points_check
    """
    df = df.copy()

    df["rushing_floor"] = df["rushing_yards"].fillna(0) * 0.1

    # Receiving floor varies by PPR format (reception weight differs)
    rec_yards_pts = df["receiving_yards"].fillna(0) * 0.1
    receptions = df["receptions"].fillna(0)
    for fmt, weight in PPR_FORMATS.items():
        suffix = "" if fmt == "ppr" else f"_{fmt}"
        df[f"receiving_floor{suffix}"] = receptions * weight + rec_yards_pts

    df["td_points"] = df["rushing_tds"].fillna(0) * 6 + df["receiving_tds"].fillna(0) * 6

    df["fumble_penalty"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    ) * -2

    # Sanity check: sum should match fantasy_points minus passing component (full PPR)
    df["fantasy_points_check"] = (
        df["rushing_floor"] + df["receiving_floor"] + df["td_points"] + df["fumble_penalty"]
    )

    passing_component = (
        df["passing_yards"].fillna(0) * 0.04
        + df["passing_tds"].fillna(0) * 4
        + df["interceptions"].fillna(0) * -2
    )
    discrepancy = (df["fantasy_points"] - df["fantasy_points_check"] - passing_component).abs()
    if (discrepancy > 0.01).any():
        n_bad = (discrepancy > 0.01).sum()
        print(f"WARNING: {n_bad} rows have target decomposition discrepancy > 0.01 pts")

    # Cross-validate against nflverse pre-computed PPR points
    if "fantasy_points_ppr" in df.columns:
        nfl_discrepancy = (df["fantasy_points"] - df["fantasy_points_ppr"]).abs()
        n_nfl_mismatch = (nfl_discrepancy > 0.5).sum()
        if n_nfl_mismatch > 0:
            print(
                f"INFO: {n_nfl_mismatch} rows differ from nflverse fantasy_points_ppr by > 0.5 pts"
            )

    return df


def compute_fumble_adjustment(df: pd.DataFrame) -> pd.Series:
    """Compute per-player historical fumble rate for post-prediction adjustment.

    Uses the rolling mean of total fumbles over L8 window (shifted, no leakage).
    """
    total_fumbles = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )
    # Need a named column for groupby transform
    df = df.copy()
    df["_total_fumbles"] = total_fumbles
    fumble_rate = df.groupby(["player_id", "season"])["_total_fumbles"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    return (fumble_rate * -2).fillna(0)  # Convert to fantasy point penalty
