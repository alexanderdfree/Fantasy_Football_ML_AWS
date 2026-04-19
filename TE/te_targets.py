import pandas as pd

from src.config import PPR_FORMATS


def compute_te_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 prediction targets for TE rows.

    Targets:
      - receiving_floor: receptions * PPR_weight + receiving_yards * 0.1
      - rushing_floor: rushing_yards * 0.1 (minor for TEs)
      - td_points: receiving_tds * 6 + rushing_tds * 6

    Note: td_points intentionally excludes 2pt conversions to align with
    SCORING_PPR used by compute_fantasy_points().
    """
    df = df.copy()

    rec_yards_pts = df["receiving_yards"].fillna(0) * 0.1
    receptions = df["receptions"].fillna(0)
    for fmt, weight in PPR_FORMATS.items():
        suffix = "" if fmt == "ppr" else f"_{fmt}"
        df[f"receiving_floor{suffix}"] = receptions * weight + rec_yards_pts

    df["rushing_floor"] = df["rushing_yards"].fillna(0) * 0.1

    df["td_points"] = df["receiving_tds"].fillna(0) * 6 + df["rushing_tds"].fillna(0) * 6

    df["fumble_penalty"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    ) * -2

    # Sanity check: sum should match fantasy_points minus passing component (full PPR)
    fantasy_points_check = (
        df["receiving_floor"] + df["rushing_floor"] + df["td_points"] + df["fumble_penalty"]
    )
    passing_component = (
        df["passing_yards"].fillna(0) * 0.04
        + df["passing_tds"].fillna(0) * 4
        + df["interceptions"].fillna(0) * -2
    )
    discrepancy = (df["fantasy_points"] - fantasy_points_check - passing_component).abs()
    if (discrepancy > 0.01).any():
        n_bad = (discrepancy > 0.01).sum()
        print(f"WARNING: {n_bad} TE rows have target decomposition discrepancy > 0.01 pts")

    return df


def compute_te_fumble_adjustment(df: pd.DataFrame) -> pd.Series:
    """Compute per-player historical fumble rate for post-prediction adjustment."""
    df = df.copy()
    total_fumbles = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )
    df["_total_fumbles"] = total_fumbles
    fumble_rate = df.groupby(["player_id", "season"])["_total_fumbles"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    return (fumble_rate * -2).fillna(0)
