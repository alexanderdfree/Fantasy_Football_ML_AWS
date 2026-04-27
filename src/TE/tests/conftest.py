"""Shared fixtures for TE tests — thin wrappers over src.shared.tests.position_fixtures.

Generic factories are imported from ``src.shared.tests.position_fixtures``;
this conftest binds them to the TE scoring scale (~15) and targets, and
keeps the TE-specific tiny-splits fixture used by E2E and regression tests.
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
    make_test_df as _make_test_df,
)
from src.shared.tests.position_fixtures import (
    register_position_markers,
)
from src.TE.te_config import TE_TARGETS  # noqa: E402

# TE fantasy points typically span 0-15 (lower than WRs).
TE_SCORING_SCALE = 15


def pytest_configure(config):
    register_position_markers(
        config,
        extra=[("slow", "excluded from default run")],
    )


# ---------------------------------------------------------------------------
# Generic TE fixtures — bind shared factories to TE scale / prefix / targets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def te_sim_df_factory():
    """Factory for synthetic weekly-simulation DataFrames with TE scoring."""

    def _factory(n_weeks: int = 4, n_players: int = 15, seed: int = 42):
        return _make_sim_df(
            TE_SCORING_SCALE,
            n_weeks,
            n_players,
            seed,
            id_prefix="TE",
            rng_kind="default",
        )

    return _factory


@pytest.fixture()
def te_sim_df(te_sim_df_factory):
    """Default-size weekly-simulation DataFrame (4 weeks x 15 players)."""
    return te_sim_df_factory()


@pytest.fixture(scope="session")
def te_test_df_factory():
    """Factory for synthetic ranking-metrics DataFrames with TE scoring scale."""

    def _factory(n_weeks: int = 3, n_players: int = 15, seed: int = 42):
        return _make_test_df(
            TE_SCORING_SCALE,
            n_weeks,
            n_players,
            seed,
            id_prefix="TE",
            rng_kind="default",
        )

    return _factory


@pytest.fixture()
def te_test_df(te_test_df_factory):
    """Default-size ranking-metrics DataFrame (3 weeks x 15 players)."""
    return te_test_df_factory()


@pytest.fixture(scope="session")
def te_tensor_factory():
    """Factory for random TE (preds, targets) tensor dicts used by loss tests."""

    def _factory(n: int = 10, seed: int = 42):
        return _make_tensors(TE_TARGETS, n=n, seed=seed)

    return _factory


@pytest.fixture()
def te_tensors(te_tensor_factory):
    """Default-size TE tensor dicts (n=10)."""
    return te_tensor_factory()


@pytest.fixture(scope="session")
def te_splits_factory():
    """Factory for (train, val, test) DataFrames used by fill_te_nans tests."""
    return _make_splits


@pytest.fixture(scope="session")
def te_position_df_factory():
    """Factory for position-encoded DataFrames used by filter_to_te tests."""

    def _factory(positions, has_pos_cols: bool = True):
        return _make_position_df(positions, stat_col="receiving_yards", has_pos_cols=has_pos_cols)

    return _factory


# ---------------------------------------------------------------------------
# TE-specific: tiny synthetic dataset for E2E + regression tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def te_tiny_splits():
    """Build tiny synthetic TE-only train/val/test splits for E2E/regression tests.

    50 players x 2 seasons x 17 weeks = 1700 rows per split, deterministic (seed=42).
    All columns required by the full pipeline (raw stats, opponent, schedule-friendly
    team, position encoding, etc.) are populated.
    """
    return _build_tiny_te_splits(seed=42)


def _build_tiny_te_splits(seed: int = 42):
    """Construct fully-featured tiny TE splits.

    Shared by ``te_tiny_splits`` and callable directly for tests that need
    two fresh independent builds (e.g. reproducibility assertions).
    """
    from src.features.engineer import build_features

    rng = np.random.default_rng(seed)
    n_players = 50
    seasons = [2022, 2023]
    weeks = list(range(1, 18))
    teams = ["KC", "BUF", "SF", "DAL", "GB", "BAL", "CIN", "MIA"]

    rows = []
    for pid in range(1, n_players + 1):
        player_id = f"TE{pid:03d}"
        # Give each player a consistent team across seasons to keep schedule merges stable.
        team = teams[(pid - 1) % len(teams)]
        skill = rng.random()  # per-player "talent" for realistic correlation
        for season in seasons:
            for week in weeks:
                opp = teams[(pid + week - 1) % len(teams)]
                if opp == team:
                    opp = teams[(pid + week) % len(teams)]
                base_targets = rng.poisson(4 + 4 * skill)
                base_rec = int(min(base_targets, rng.poisson(3 + 3 * skill)))
                receiving_yards = max(0.0, rng.normal(55 * skill + 20, 15))
                receiving_tds = int(rng.binomial(1, 0.05 + 0.15 * skill))
                rushing_yards = max(0.0, rng.normal(1, 3))
                rushing_tds = 0
                receiving_epa = rng.normal(0.5, 1.5)
                receiving_first_downs = int(base_rec * 0.6)
                receiving_air_yards = receiving_yards * 1.2
                rec_yac = max(0.0, receiving_yards * 0.5)
                snap_pct = float(np.clip(rng.normal(0.6 + 0.3 * skill, 0.1), 0.0, 1.0))
                fumbles_lost_recv = int(rng.binomial(1, 0.01))
                fp = (
                    receiving_yards * 0.1
                    + base_rec * 1.0
                    + receiving_tds * 6
                    + rushing_yards * 0.1
                    - fumbles_lost_recv * 2
                )
                rows.append(
                    {
                        "player_id": player_id,
                        "season": season,
                        "week": week,
                        "position": "TE",
                        "recent_team": team,
                        "opponent_team": opp,
                        "fantasy_points": fp,
                        "targets": base_targets,
                        "receptions": base_rec,
                        "carries": 0,
                        "rushing_yards": rushing_yards,
                        "rushing_tds": rushing_tds,
                        "receiving_yards": receiving_yards,
                        "receiving_tds": receiving_tds,
                        "receiving_air_yards": receiving_air_yards,
                        "receiving_yards_after_catch": rec_yac,
                        "receiving_epa": receiving_epa,
                        "receiving_first_downs": receiving_first_downs,
                        "passing_yards": 0.0,
                        "passing_tds": 0,
                        "attempts": 0,
                        "completions": 0,
                        "interceptions": 0,
                        "sacks": 0,
                        "sack_fumbles_lost": 0,
                        "rushing_fumbles_lost": 0,
                        "receiving_fumbles_lost": fumbles_lost_recv,
                        "fumbles_lost": fumbles_lost_recv,
                        "snap_pct": snap_pct,
                        "is_returning_from_absence": 0,
                        "days_rest": 7,
                        "practice_status": 0,
                        "game_status": 0,
                        "depth_chart_rank": 1,
                    }
                )

    df = pd.DataFrame(rows)
    df = build_features(df)

    # Season-based split: 2022 train, 2023 first half val, 2023 second half test.
    train_df = df[df["season"] == 2022].copy()
    val_df = df[(df["season"] == 2023) & (df["week"] <= 9)].copy()
    test_df = df[(df["season"] == 2023) & (df["week"] > 9)].copy()
    return train_df, val_df, test_df
