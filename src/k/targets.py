import pandas as pd


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 kicker prediction targets and fantasy_points total.

    Target decomposition (all non-negative raw counts / point values):
      fg_yard_points = fg_yards_made * 0.1   (sum of made-FG kick_distance)
      pat_points     = pat_made * 1          (PAT makes)
      fg_misses      = fg_missed             (raw count of missed FGs)
      xp_misses      = pat_missed            (raw count of missed PATs)

    Final fantasy total (sign vector [+1, +1, -1, -1]):
      fantasy_points = fg_yard_points + pat_points - fg_misses - xp_misses

    All 4 heads are trained non-negative; the sign flip happens only in the
    total aggregation here (and mirrored in the inference path).
    """
    df = df.copy()

    # Positive-contribution heads
    df["fg_yard_points"] = df["fg_yards_made"].fillna(0) * 0.1
    df["pat_points"] = df["pat_made"].fillna(0) * 1

    # Penalty heads — kept positive so the NN's non-negative clamp holds; the
    # sign is applied when aggregating to fantasy_points below.
    df["fg_misses"] = df["fg_missed"].fillna(0)
    df["xp_misses"] = df["pat_missed"].fillna(0)

    # Override fantasy_points with the signed K-specific formula.
    df["fantasy_points"] = (
        df["fg_yard_points"] + df["pat_points"] - df["fg_misses"] - df["xp_misses"]
    )

    return df
