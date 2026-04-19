import pandas as pd

# Standard D/ST points-allowed tiers
_PTS_ALLOWED_TIERS = [
    (0, 0, 10),
    (1, 6, 7),
    (7, 13, 4),
    (14, 20, 1),
    (21, 27, 0),
    (28, 34, -1),
    (35, 999, -4),
]


def _pts_allowed_to_bonus(pts: float) -> float:
    """Convert points allowed to fantasy bonus using standard tiers."""
    pts = int(pts)
    for lo, hi, bonus in _PTS_ALLOWED_TIERS:
        if lo <= pts <= hi:
            return bonus
    return -4  # 35+


def compute_dst_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 prediction targets for D/ST.

    Target decomposition (predictable components only):
      defensive_scoring = sacks * 1 + INTs * 2 + fumble_rec * 2
      td_points          = special_teams_tds * 6
      pts_allowed_bonus  = tiered scoring based on points allowed

    Excluded from targets (moved to adjustment):
      - def_tds: nflreadr only populates individual defensive player stats
        from 2025 onward (0% fill for 2012-2024). Including them creates a
        train/test distribution shift (~+0.31 pts/game bias in 2025).
      - def_safeties: same data gap; negligible magnitude (~0.03 pts/game).

    Total D/ST fantasy points = targets + adjustment (def_tds*6 + safeties*2)
    """
    df = df.copy()

    # 1. Defensive scoring (base production — excludes safeties)
    df["defensive_scoring"] = (
        df["def_sacks"].fillna(0) * 1
        + df["def_ints"].fillna(0) * 2
        + df["def_fumble_rec"].fillna(0) * 2
    )

    # 2. Touchdown points (special teams only — excludes defensive TDs)
    df["td_points"] = df["special_teams_tds"].fillna(0) * 6

    # 3. Points-allowed bonus (opponent-dependent, tiered)
    df["pts_allowed_bonus"] = df["points_allowed"].fillna(21).apply(_pts_allowed_to_bonus)

    # Unpredictable component (for adjustment at inference)
    df["_dst_adjustment"] = df["def_tds"].fillna(0) * 6 + df["def_safeties"].fillna(0) * 2

    # Total D/ST fantasy points (full, including adjustment)
    df["fantasy_points"] = (
        df["defensive_scoring"] + df["td_points"] + df["pts_allowed_bonus"] + df["_dst_adjustment"]
    )

    return df


def compute_dst_adjustment(df: pd.DataFrame) -> pd.Series:
    """Return unpredictable D/ST scoring: defensive TDs + safeties.

    These stats have no training history (nflreadr 2025-only), so they
    are excluded from model targets and captured here as irreducible noise.
    """
    return (
        df["_dst_adjustment"] if "_dst_adjustment" in df.columns else pd.Series(0.0, index=df.index)
    )
