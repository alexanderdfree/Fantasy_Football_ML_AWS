import pandas as pd


def compute_k_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 2 prediction targets + miss penalty for kicker rows.

    Target decomposition:
      fg_points  = FG scoring by distance (3/4/5 pts) — main variance driver
      pat_points = PAT makes (1 pt each) — correlated with team TD scoring

    Miss penalty (adjustment):
      fg_missed * (-1) + pat_missed * (-1)

    Total kicker fantasy points = fg_points + pat_points + miss_penalty
    """
    df = df.copy()

    # FG points by distance (standard fantasy scoring)
    df["fg_points"] = (
        (
            df["fg_made_0_19"].fillna(0)
            + df["fg_made_20_29"].fillna(0)
            + df["fg_made_30_39"].fillna(0)
        )
        * 3
        + df["fg_made_40_49"].fillna(0) * 4
        + (df["fg_made_50_59"].fillna(0) + df["fg_made_60_"].fillna(0)) * 5
    )

    # PAT points
    df["pat_points"] = df["pat_made"].fillna(0) * 1

    # Miss penalty (applied as post-prediction adjustment)
    df["miss_penalty"] = df["fg_missed"].fillna(0) * (-1) + df["pat_missed"].fillna(0) * (-1)

    # Override fantasy_points with correct kicker scoring
    df["fantasy_points"] = df["fg_points"] + df["pat_points"] + df["miss_penalty"]

    return df


def compute_k_miss_adjustment(df: pd.DataFrame) -> pd.Series:
    """Compute per-kicker historical miss rate for post-prediction adjustment.

    Uses the rolling mean of total misses over L8 window (shifted, no leakage).
    Returns the expected miss penalty (negative value).
    """
    total_misses = df["fg_missed"].fillna(0) + df["pat_missed"].fillna(0)
    df = df.copy()
    df["_total_misses"] = total_misses
    miss_rate = df.groupby(["player_id"])["_total_misses"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )

    return (miss_rate * -1).fillna(0)
