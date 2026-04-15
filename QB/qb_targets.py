import pandas as pd
import numpy as np


def compute_qb_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 prediction targets for QB rows.

    Targets:
      - passing_floor: passing_yards * 0.04
      - rushing_floor: rushing_yards * 0.1
      - td_points: passing_tds * 4 + rushing_tds * 6 + receiving_tds * 6

    Note: td_points intentionally excludes 2pt conversions to align with
    SCORING_PPR used by compute_fantasy_points().
    """
    df = df.copy()

    df["passing_floor"] = df["passing_yards"].fillna(0) * 0.04
    df["rushing_floor"] = df["rushing_yards"].fillna(0) * 0.1

    df["td_points"] = (
        df["passing_tds"].fillna(0) * 4
        + df["rushing_tds"].fillna(0) * 6
        + df["receiving_tds"].fillna(0) * 6
    )

    # Penalty components (not predicted, applied post-hoc)
    df["interception_penalty"] = df["interceptions"].fillna(0) * -2
    df["fumble_penalty"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    ) * -2

    # Minor receiving component (captured in residual)
    df["receiving_component"] = (
        df["receptions"].fillna(0) * 1.0  # PPR
        + df["receiving_yards"].fillna(0) * 0.1
    )

    # Sanity check: targets + penalties + receiving should ≈ fantasy_points
    fantasy_points_check = (
        df["passing_floor"] + df["rushing_floor"] + df["td_points"]
        + df["interception_penalty"] + df["fumble_penalty"]
        + df["receiving_component"]
    )
    discrepancy = (df["fantasy_points"] - fantasy_points_check).abs()
    if (discrepancy > 0.01).any():
        n_bad = (discrepancy > 0.01).sum()
        print(f"WARNING: {n_bad} QB rows have target decomposition discrepancy > 0.01 pts")

    return df


def compute_qb_adjustment(df: pd.DataFrame) -> pd.Series:
    """Compute per-player historical INT rate + fumble rate for post-prediction adjustment.

    Uses rolling L8 window (shifted) for both interceptions and fumbles.
    """
    df = df.copy()

    df["_int_pts"] = df["interceptions"].fillna(0) * -2
    int_adj = df.groupby(["player_id", "season"])["_int_pts"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    total_fumbles = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )
    df["_fumble_pts"] = total_fumbles * -2
    fumble_adj = df.groupby(["player_id", "season"])["_fumble_pts"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    # Also add the small receiving component as adjustment
    df["_recv_comp"] = df["receptions"].fillna(0) * 1.0 + df["receiving_yards"].fillna(0) * 0.1
    recv_adj = df.groupby(["player_id", "season"])["_recv_comp"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    return (int_adj + fumble_adj + recv_adj).fillna(0)
