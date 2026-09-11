"""Raw-stat display units, independent of model execution libraries."""

TARGET_UNITS = {
    "passing_yards": "yds",
    "rushing_yards": "yds",
    "receiving_yards": "yds",
    "passing_tds": "TDs",
    "rushing_tds": "TDs",
    "receiving_tds": "TDs",
    "receptions": "rec",
    "interceptions": "INT",
    "fumbles_lost": "fum",
    # DST raw-stat units
    "def_sacks": "sacks",
    "def_ints": "INT",
    "def_fumble_rec": "fum",
    "def_fumbles_forced": "FF",
    "def_safeties": "safety",
    "def_tds": "TDs",
    "def_blocked_kicks": "blk",
    "special_teams_tds": "TDs",
    "points_allowed": "pts",
    "yards_allowed": "yds",
    # Kicker raw-stat units (predictions-tab breakdown drill-down). See
    # ``K_TARGETS`` above / ``src/k/targets.py``: fg_yard_points and pat_points
    # are point values, fg_misses / xp_misses are raw miss counts.
    "fg_yard_points": "pts",
    "pat_points": "pts",
    "fg_misses": "misses",
    "xp_misses": "misses",
}
