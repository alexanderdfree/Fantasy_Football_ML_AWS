import pandas as pd
import numpy as np


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

    Target decomposition:
      defensive_scoring = sacks * 1 + INTs * 2 + fumble_rec * 2 + safeties * 2
      td_points          = (def_tds + special_teams_tds) * 6
      pts_allowed_bonus  = tiered scoring based on points allowed

    Total D/ST fantasy points = defensive_scoring + td_points + pts_allowed_bonus
    """
    df = df.copy()

    # 1. Defensive scoring (base production)
    df["defensive_scoring"] = (
        df["def_sacks"].fillna(0) * 1
        + df["def_ints"].fillna(0) * 2
        + df["def_fumble_rec"].fillna(0) * 2
        + df["def_safeties"].fillna(0) * 2
    )

    # 2. Touchdown points (high-variance big plays)
    df["td_points"] = (
        df["def_tds"].fillna(0) + df["special_teams_tds"].fillna(0)
    ) * 6

    # 3. Points-allowed bonus (opponent-dependent, tiered)
    df["pts_allowed_bonus"] = df["points_allowed"].fillna(21).apply(_pts_allowed_to_bonus)

    # Total D/ST fantasy points
    df["fantasy_points"] = (
        df["defensive_scoring"] + df["td_points"] + df["pts_allowed_bonus"]
    )

    return df


def compute_dst_adjustment(df: pd.DataFrame) -> pd.Series:
    """No adjustment needed — all D/ST scoring is captured in the three targets."""
    return pd.Series(0.0, index=df.index)
