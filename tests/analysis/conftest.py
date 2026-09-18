"""Shared synthetic-history fixtures: prepared-frame shaped, no real data.

Fake checkpoints live in ``tests.analysis.fake_bundles`` so this conftest
stays free of torch for the tests that never load a model.
"""

import pandas as pd
import pytest

from src.shared.registry import get_inference_spec

SHORT_WEEKS = (1, 2, 4, 5, 6, 7)
LONG_WEEKS = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

# Per-position base game lines; every schema relation and bound holds, and the
# eight-game windows land inside the shipped PPG bands.
BASE = {
    "QB": dict(
        attempts=30,
        completions=20,
        passing_tds=2,
        interceptions=1,
        carries=4,
        rushing_yards=20,
        rushing_tds=0,
        fumbles_lost=0,
        snap_pct_raw=0.9,
        pass_yards_gained_exp=150,
        team_rush_attempts=25,
        team_rushing_yards=100,
        team_points_scored=24,
        opp_team_points_scored=21,
    ),
    "RB": dict(
        carries=14,
        targets=4,
        receptions=3,
        rushing_tds=0,
        receiving_tds=0,
        fumbles_lost=0,
        receiving_yards=25,
        snap_pct_raw=0.6,
        rushing_first_downs=3,
        receiving_first_downs=1,
        redzone_carries=2,
        redzone_targets=1,
        inside10_carries=1,
        inside5_carries=1,
        game_carry_share=0.6,
        game_target_share=0.1,
        game_carry_hhi=0.5,
        game_target_hhi=0.3,
        redzone_target_share=0.2,
        team_pass_attempts=35,
        team_completions=23,
        team_passing_yards=250,
        team_rush_attempts=24,
        team_rushing_yards=110,
        team_points_scored=24,
        team_turnovers=1,
        opp_team_points_scored=21,
        rush_yards_gained_exp=60,
    ),
    "WR": dict(
        targets=8,
        receptions=5,
        receiving_tds=0,
        rushing_tds=0,
        fumbles_lost=0,
        carries=0,
        rushing_yards=0,
        snap_pct_raw=0.85,
        redzone_targets=1,
        redzone_target_share=0.25,
        game_target_share=0.25,
        game_target_hhi=0.3,
        game_opportunity_index=0.2,
        team_pass_attempts=35,
        team_passing_yards=250,
        team_rush_attempts=24,
        team_points_scored=24,
        opp_team_points_scored=21,
        rec_yards_gained_exp=55,
    ),
}
BASE["TE"] = {**BASE["WR"], "targets": 6, "receptions": 4, "snap_pct_raw": 0.7}
# Distinctive per-game production (by week and player) and one opaque signal per position.
YARDS = {
    "QB": ("passing_yards", 180, 10),
    "RB": ("rushing_yards", 50, 5),
    "WR": ("receiving_yards", 40, 5),
    "TE": ("receiving_yards", 30, 4),
}
OPAQUE = {
    "QB": ("qbr_total", 5),
    "RB": ("rush_touchdown_exp", 0.05),
    "WR": ("rec_touchdown_exp", 0.05),
    "TE": ("rec_touchdown_exp", 0.05),
}


# A whitelisted rolling feature per position: it must reach the context but never a history.
POISON = {
    "QB": "rolling_mean_passing_yards_L3",
    "RB": "rolling_mean_carries_L3",
    "WR": "rolling_mean_targets_L3",
    "TE": "rolling_mean_targets_L3",
}


def poison_column(position: str) -> str:
    assert POISON[position] in get_inference_spec(position)["get_feature_columns_fn"]()
    return POISON[position]


def position_rows(position: str, weeks=SHORT_WEEKS) -> pd.DataFrame:
    """A prepared frame: every production feature column plus the raw history stats.

    Two players, three seasons (2025 is a held-out season), regular-season
    games with a deliberate week-3 gap. Opaque per-game signals are distinctive
    so resampling tests can prove they travel with their game, and the rolling
    column is a poison value that must reach the static context but never a
    generated history.
    """
    spec = get_inference_spec(position)
    features = list(spec["get_feature_columns_fn"]())
    history = list(spec["attn_history_stats"])
    poison = poison_column(position)
    yards_column, base, step = YARDS[position]
    opaque_column, opaque_step = OPAQUE[position]
    rows = []
    for player in ("p1", "p2"):
        for season in (2022, 2023, 2025):
            for week in weeks:
                row = dict.fromkeys(features, 0.0)
                row.update(dict.fromkeys(history, 0.0))
                row.update(BASE[position])
                row.update(
                    player_id=player,
                    season=season,
                    week=week,
                    position=position,
                    season_type="REG",
                    recent_team="KC",
                    opponent_team="BUF",
                    depth_chart_rank=1.0,
                    season_starts_to_date=float(week - 1),
                )
                row[yards_column] = base + week * step + (20 if player == "p2" else 0)
                row[opaque_column] = week * opaque_step
                row[poison] = 9999
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def qb_source():
    return position_rows("QB")


@pytest.fixture
def qb_source_long():
    """Twelve games per season, long enough for the shipped eight-game recipes."""
    return position_rows("QB", LONG_WEEKS)
