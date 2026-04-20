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

# Yahoo-style yards-allowed tiers
_YDS_ALLOWED_TIERS = [
    (0, 99, 5),
    (100, 199, 3),
    (200, 299, 2),
    (300, 349, 0),
    (350, 399, -1),
    (400, 449, -3),
    (450, 99999, -5),
]


def _pts_allowed_to_bonus(pts: float) -> float:
    """Convert points allowed to fantasy bonus using standard tiers."""
    pts = int(pts)
    for lo, hi, bonus in _PTS_ALLOWED_TIERS:
        if lo <= pts <= hi:
            return bonus
    return -4  # 35+


def _yds_allowed_to_bonus(ya: float) -> float:
    """Convert yards allowed to fantasy bonus using Yahoo-style tiers."""
    ya = int(ya)
    for lo, hi, bonus in _YDS_ALLOWED_TIERS:
        if lo <= ya <= hi:
            return bonus
    return -5  # 450+


def compute_dst_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 prediction targets for D/ST.

    Decomposition:
      defensive_production = sacks*1 + INT*2 + fum_rec*2 + forced_fum*1 + safeties*2
      def_td_points        = def_tds * 6
      st_production        = special_teams_tds*6 + blocked_kicks*2
      points_allowed       = raw PA (0-55+), tier-mapped at inference
      yards_allowed        = raw YA (0-600+), tier-mapped at inference

    fantasy_points = defensive_production + def_td_points + st_production
                   + _pts_allowed_to_bonus(points_allowed)
                   + _yds_allowed_to_bonus(yards_allowed)
    """
    df = df.copy()

    # 1. Defensive production (sacks + turnovers + forced fumbles + safeties)
    df["defensive_production"] = (
        df["def_sacks"].fillna(0) * 1
        + df["def_ints"].fillna(0) * 2
        + df["def_fumble_rec"].fillna(0) * 2
        + df["def_fumbles_forced"].fillna(0) * 1
        + df["def_safeties"].fillna(0) * 2
    )

    # 2. Defensive touchdown points
    df["def_td_points"] = df["def_tds"].fillna(0) * 6

    # 3. Special teams production (ST TDs + blocked kicks)
    df["st_production"] = (
        df["special_teams_tds"].fillna(0) * 6 + df["def_blocked_kicks"].fillna(0) * 2
    )

    # 4/5. Raw PA/YA — regressed directly, tier mapping applied at inference.
    df["points_allowed"] = df["points_allowed"].fillna(21)
    df["yards_allowed"] = df["yards_allowed"].fillna(350)

    # Full fantasy points (tier-mapped) — used for eval and the total aux loss.
    df["fantasy_points"] = (
        df["defensive_production"]
        + df["def_td_points"]
        + df["st_production"]
        + df["points_allowed"].apply(_pts_allowed_to_bonus)
        + df["yards_allowed"].apply(_yds_allowed_to_bonus)
    )

    return df
