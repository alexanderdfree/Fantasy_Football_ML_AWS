"""Shared fixtures for RB tests — thin wrappers over shared.tests.position_fixtures.

Generic factories (``make_sim_df``, ``make_ranking_df``, ``make_tensors``,
``make_splits``, ``make_position_df``) now come from
``shared/tests/position_fixtures.py``.  Only RB-specific helpers remain
here: the multi-week player-games frame, single-row target row, fumble
DataFrame, synthetic pipeline dataset for E2E/regression tests, and the
Ridge training data factory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.RB.rb_config import RB_TARGETS  # noqa: E402
from src.shared.tests.position_fixtures import (  # noqa: E402
    make_position_df as _make_position_df,
)
from src.shared.tests.position_fixtures import (
    make_sim_df as _make_sim_df,
)
from src.shared.tests.position_fixtures import (
    make_splits as _make_splits,
)
from src.shared.tests.position_fixtures import (
    make_tensors as _make_tensors,
)
from src.shared.tests.position_fixtures import (
    make_test_df as _make_ranking_df_shared,
)
from src.shared.tests.position_fixtures import (
    register_position_markers,
)

# RB scoring scale: ~20 fantasy points typical.
RB_SCORING_SCALE = 20


def pytest_configure(config):
    register_position_markers(config)


# ---------------------------------------------------------------------------
# Generic simulation / ranking / tensor / split fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def make_sim_df():
    def _make(n_weeks: int, n_players: int, seed: int = 42):
        return _make_sim_df(RB_SCORING_SCALE, n_weeks, n_players, seed, id_prefix="P")

    return _make


@pytest.fixture(scope="session")
def sim_df_default(make_sim_df):
    """Standard 4-week x 15-player RB-scale sim DataFrame."""
    return make_sim_df(n_weeks=4, n_players=15)


@pytest.fixture(scope="session")
def make_ranking_df():
    def _make(n_weeks: int, n_players: int, seed: int = 42):
        return _make_ranking_df_shared(RB_SCORING_SCALE, n_weeks, n_players, seed, id_prefix="P")

    return _make


@pytest.fixture(scope="session")
def make_tensors():
    def _make(n: int = 10, seed: int = 42):
        return _make_tensors(RB_TARGETS, n=n, seed=seed)

    return _make


@pytest.fixture(scope="session")
def make_splits():
    return _make_splits


@pytest.fixture(scope="session")
def make_position_df():
    def _make(positions, has_pos_cols: bool = True):
        return _make_position_df(positions, stat_col="rushing_yards", has_pos_cols=has_pos_cols)

    return _make


# ---------------------------------------------------------------------------
# RB-specific fixtures (not generalisable across positions)
# ---------------------------------------------------------------------------


def _build_player_games(
    player_id: str = "P1",
    season: int = 2023,
    n_weeks: int = 5,
    carries: int = 10,
    targets: int = 5,
    receptions: int = 3,
    rushing_yards: int = 50,
    receiving_yards: int = 30,
    rushing_epa: float = 2.0,
    rushing_first_downs: int = 2,
    receiving_first_downs: int = 1,
    receiving_yards_after_catch: int = 15,
    receiving_epa: float = 1.5,
    receiving_air_yards: int = 20,
    recent_team: str = "KC",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [player_id] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "carries": [carries] * n_weeks,
            "targets": [targets] * n_weeks,
            "receptions": [receptions] * n_weeks,
            "rushing_yards": [rushing_yards] * n_weeks,
            "receiving_yards": [receiving_yards] * n_weeks,
            "rushing_epa": [rushing_epa] * n_weeks,
            "rushing_first_downs": [rushing_first_downs] * n_weeks,
            "receiving_first_downs": [receiving_first_downs] * n_weeks,
            "receiving_yards_after_catch": [receiving_yards_after_catch] * n_weeks,
            "receiving_epa": [receiving_epa] * n_weeks,
            "receiving_air_yards": [receiving_air_yards] * n_weeks,
            "recent_team": [recent_team] * n_weeks,
        }
    )


@pytest.fixture(scope="session")
def make_player_games():
    """Factory for multi-week single-player DataFrames (RB feature inputs)."""
    return _build_player_games


def _build_rb_row(**overrides) -> pd.DataFrame:
    """Single-row RB DataFrame with sensible defaults. fantasy_points auto-computed."""
    defaults = {
        "rushing_yards": 60,
        "receiving_yards": 30,
        "receptions": 3,
        "rushing_tds": 1,
        "receiving_tds": 0,
        "rushing_2pt_conversions": 0,
        "receiving_2pt_conversions": 0,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "interceptions": 0,
        "fantasy_points": 0.0,
    }
    defaults.update(overrides)
    if "fantasy_points" not in overrides:
        fp = (
            defaults["rushing_yards"] * 0.1
            + defaults["receptions"] * 1.0
            + defaults["receiving_yards"] * 0.1
            + (defaults["rushing_tds"] + defaults["receiving_tds"]) * 6
            + (
                defaults["sack_fumbles_lost"]
                + defaults["rushing_fumbles_lost"]
                + defaults["receiving_fumbles_lost"]
            )
            * -2
            + defaults["passing_yards"] * 0.04
            + defaults["passing_tds"] * 4
            + defaults["interceptions"] * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])


@pytest.fixture(scope="session")
def make_rb_row():
    """Factory for single-row RB target inputs."""
    return _build_rb_row


@pytest.fixture(scope="session")
def simple_ridge_data():
    """20 samples x 5 features with RB-scale raw-stat targets.

    Used by RidgeMultiTarget tests. Scales match realistic RB raw stats so
    per-target alpha grids exercise numerics near the post-migration units.
    """
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "rushing_tds": (np.random.rand(n) * 2).astype(np.float32),
        "receiving_tds": (np.random.rand(n) * 1).astype(np.float32),
        "rushing_yards": (np.random.rand(n) * 100).astype(np.float32),
        "receiving_yards": (np.random.rand(n) * 60).astype(np.float32),
        "receptions": (np.random.rand(n) * 6).astype(np.float32),
        "fumbles_lost": (np.random.rand(n) * 1).astype(np.float32),
    }
    return X, y_dict


# ---------------------------------------------------------------------------
# Synthetic RB dataset for E2E pipeline / regression tests
# ---------------------------------------------------------------------------

# Team pool from schedules_2012_2025 — these are guaranteed to exist and merge.
_SYNTH_TEAMS = ["KC", "BUF", "SF", "BAL", "PHI", "DAL", "CIN", "NYJ"]


def _build_synthetic_rb_dataset(
    n_players: int = 50,
    seasons: tuple = (2022, 2023),
    n_weeks: int = 17,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic RB-style DataFrame the pipeline can train on.

    Designed for the E2E smoke: small enough to run in ~5-15s, large enough to
    satisfy MIN_GAMES_PER_SEASON=6 after filtering.

    Fields mirror the raw `weekly` schema the pipeline expects. Values are
    drawn from plausible RB distributions so compute_rb_targets, feature
    engineering, and the NN training loop all see non-degenerate signal.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_players + 1):
        player_id = f"RB{pid:03d}"
        # Each player stays on one team across seasons for simplicity.
        team_idx = pid % len(_SYNTH_TEAMS)
        home_team = _SYNTH_TEAMS[team_idx]
        # Player "skill" makes predictions non-trivial (not pure noise).
        skill = rng.normal(0.0, 1.0)
        for season in seasons:
            for week in range(1, n_weeks + 1):
                # Opponent from the pool (deterministic rotation).
                opp_idx = (team_idx + week) % len(_SYNTH_TEAMS)
                opp_team = _SYNTH_TEAMS[opp_idx]
                if opp_team == home_team:
                    opp_team = _SYNTH_TEAMS[(opp_idx + 1) % len(_SYNTH_TEAMS)]

                carries = max(0, int(rng.normal(12 + 3 * skill, 4)))
                targets = max(0, int(rng.normal(3 + skill, 2)))
                receptions = min(targets, max(0, int(rng.normal(targets * 0.7, 1))))
                rushing_yards = max(0, int(carries * rng.normal(4.2, 0.8)))
                receiving_yards = max(0, int(receptions * rng.normal(7.5, 2.0)))
                rushing_tds = rng.poisson(0.4 + 0.1 * skill)
                receiving_tds = rng.poisson(0.1)

                fp = (
                    rushing_yards * 0.1
                    + receiving_yards * 0.1
                    + receptions * 1.0
                    + (rushing_tds + receiving_tds) * 6
                )

                rows.append(
                    {
                        "player_id": player_id,
                        "player_name": player_id,
                        "season": season,
                        "week": week,
                        "recent_team": home_team,
                        "opponent_team": opp_team,
                        "position": "RB",
                        "carries": carries,
                        "targets": targets,
                        "receptions": receptions,
                        "rushing_yards": rushing_yards,
                        "receiving_yards": receiving_yards,
                        "rushing_tds": rushing_tds,
                        "receiving_tds": receiving_tds,
                        "rushing_2pt_conversions": 0,
                        "receiving_2pt_conversions": 0,
                        "sack_fumbles_lost": 0,
                        "rushing_fumbles_lost": int(rng.random() < 0.03),
                        "receiving_fumbles_lost": 0,
                        "passing_yards": 0,
                        "passing_tds": 0,
                        "interceptions": 0,
                        "rushing_epa": rng.normal(0.0, 0.5) * carries,
                        "rushing_first_downs": int(carries * 0.25),
                        "receiving_first_downs": int(receptions * 0.4),
                        "receiving_yards_after_catch": int(receiving_yards * 0.5),
                        "receiving_epa": rng.normal(0.0, 0.3) * receptions,
                        "receiving_air_yards": max(0, int(receiving_yards * 0.8)),
                        "fantasy_points": fp,
                        "fantasy_points_ppr": fp,
                        "snap_pct": min(1.0, max(0.1, rng.normal(0.6 + 0.1 * skill, 0.1))),
                        "pos_QB": 0,
                        "pos_RB": 1,
                        "pos_WR": 0,
                        "pos_TE": 0,
                        # Features the pipeline may reference; default to 0.
                        "target_share_L3": 0.0,
                        "target_share_L5": 0.0,
                        "carry_share_L3": 0.0,
                        "carry_share_L5": 0.0,
                        "air_yards_share": 0.0,
                        "is_home": week % 2,
                        "is_returning_from_absence": 0,
                        "days_rest": 7,
                        "practice_status": 0,
                        "game_status": 0,
                        "depth_chart_rank": 1,
                        "trend_fantasy_points": 0.0,
                        "trend_targets": 0.0,
                        "trend_carries": 0.0,
                        "trend_snap_pct": 0.0,
                        "opp_fantasy_pts_allowed_to_pos": 10.0,
                        "opp_rush_pts_allowed_to_pos": 5.0,
                        "opp_recv_pts_allowed_to_pos": 5.0,
                        "opp_def_rank_vs_pos": 16,
                        "opp_def_sacks_L5": 2.0,
                        "opp_def_pass_yds_allowed_L5": 250.0,
                        "opp_def_pass_td_allowed_L5": 1.5,
                        "opp_def_ints_L5": 0.8,
                        "opp_def_rush_yds_allowed_L5": 120.0,
                        "opp_def_pts_allowed_L5": 22.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_rb_dataset():
    """Full-season synthetic RB DataFrame (50 players x 2 seasons x 17 weeks)."""
    return _build_synthetic_rb_dataset()


@pytest.fixture(scope="session")
def synthetic_rb_splits(synthetic_rb_dataset):
    """Split synthetic RB data into (train, val, test).

    Train: full first season (17 weeks).
    Val:   second season, weeks 1..13.
    Test:  second season, weeks 14..17.

    The pipeline's Ridge CV uses ``cv_split_column="week"`` (set in the E2E
    config) — within one training season this gives ~17 distinct week labels
    so expanding-window folds still work. Keeping training to one season
    sticks to the coordinator's "2 seasons x 17 weeks" recipe.
    """
    df = synthetic_rb_dataset
    seasons = sorted(df["season"].unique())
    train = df[df["season"] == seasons[0]].copy()
    held = df[df["season"] == seasons[-1]].copy()
    max_week = held["week"].max()
    val = held[held["week"] <= max_week - 4].copy()
    test = held[held["week"] > max_week - 4].copy()
    return train, val, test
