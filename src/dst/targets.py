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


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-fill the raw D/ST target columns and compute ``fantasy_points``.

    Required input columns (all 10 raw targets must be on ``df``; upstream
    ``build_data`` guarantees this — missing columns raise KeyError, which
    is the desired behavior because it surfaces data-pipeline bugs instead of
    silently imputing zeros):
      def_sacks, def_ints, def_fumble_rec, def_fumbles_forced, def_safeties,
      def_tds, def_blocked_kicks, special_teams_tds,
      points_allowed, yards_allowed

    ``fantasy_points`` is the scored total, used for baseline / eval / total
    aux-loss supervision. It's recomputed here (not predicted directly) so the
    column is always in sync with the raw stats.

    fantasy_points = def_sacks*1 + def_ints*2 + def_fumble_rec*2
                   + def_fumbles_forced*1 + def_safeties*2
                   + def_tds*6
                   + special_teams_tds*6 + def_blocked_kicks*2
                   + _pts_allowed_to_bonus(points_allowed)
                   + _yds_allowed_to_bonus(yards_allowed)
    """
    df = df.copy()

    # Zero-fill raw counts (missing rows typically mean 0 occurrences, not NaN)
    _count_cols = [
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_fumbles_forced",
        "def_safeties",
        "def_tds",
        "def_blocked_kicks",
        "special_teams_tds",
    ]
    for col in _count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # PA/YA — league-average defaults when missing
    df["points_allowed"] = df["points_allowed"].fillna(21)
    df["yards_allowed"] = df["yards_allowed"].fillna(350)

    # Fantasy points: linear portion + tier bonuses. Must match
    # ``src.shared.aggregate_targets.predictions_to_fantasy_points("DST", …)``.
    linear = (
        df["def_sacks"] * 1
        + df["def_ints"] * 2
        + df["def_fumble_rec"] * 2
        + df["def_fumbles_forced"] * 1
        + df["def_safeties"] * 2
        + df["def_tds"] * 6
        + df["special_teams_tds"] * 6
        + df["def_blocked_kicks"] * 2
    )
    df["fantasy_points"] = (
        linear
        + df["points_allowed"].apply(_pts_allowed_to_bonus)
        + df["yards_allowed"].apply(_yds_allowed_to_bonus)
    )

    return df
