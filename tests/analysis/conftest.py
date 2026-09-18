"""Shared synthetic-history fixtures: prepared-frame shaped, no real data."""

import pandas as pd
import pytest

from src.qb.config import POSITION_CONFIG as QB_CONFIG
from src.qb.features import get_feature_columns as qb_feature_columns

SHORT_WEEKS = (1, 2, 4, 5, 6, 7)
LONG_WEEKS = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


def qb_rows(weeks):
    """A prepared QB frame: every production feature column plus the raw history stats.

    Two players, three seasons (2025 is a held-out season), regular-season
    games with a deliberate week-3 gap. Opaque per-game signals are distinctive
    so resampling tests can prove they travel with their game, and the rolling
    column is a poison value that must reach the static context but never a
    generated history.
    """
    rows = []
    for player in ("p1", "p2"):
        for season in (2022, 2023, 2025):
            for week in weeks:
                row = dict.fromkeys(qb_feature_columns(), 0.0)
                row.update(dict.fromkeys(QB_CONFIG.attn_history_stats, 0.0))
                row.update(
                    player_id=player,
                    season=season,
                    week=week,
                    position="QB",
                    season_type="REG",
                    recent_team="KC",
                    opponent_team="BUF",
                    attempts=30,
                    completions=20,
                    passing_yards=180 + week * 10 + (20 if player == "p2" else 0),
                    passing_tds=2,
                    interceptions=1,
                    carries=4,
                    rushing_yards=20,
                    rushing_tds=0,
                    fumbles_lost=0,
                    snap_pct_raw=0.9,
                    # Deliberately distinctive correlated opaque fields.
                    qbr_total=week * 5,
                    pts_added=week,
                    pass_yards_gained_exp=150 + week,
                    team_rush_attempts=25,
                    team_rushing_yards=100,
                    team_points_scored=24,
                    opp_team_points_scored=21,
                    # Distinctive non-temporal statics for the forecast context.
                    depth_chart_rank=1.0,
                    prior_season_mean_passing_yards=250.0 + (10.0 if player == "p2" else 0.0),
                    season_starts_to_date=float(week - 1),
                    # Must never be copied into generated histories.
                    rolling_mean_passing_yards_L3=9999,
                )
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def qb_source():
    return qb_rows(SHORT_WEEKS)


@pytest.fixture
def qb_source_long():
    """Twelve games per season, long enough for the shipped eight-game recipes."""
    return qb_rows(LONG_WEEKS)
