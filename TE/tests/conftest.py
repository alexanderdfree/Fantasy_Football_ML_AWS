"""Shared fixtures and pytest configuration for TE tests.

Consolidates helpers previously duplicated across TE test files:
  - `_make_sim_df`  -> `te_sim_df_factory`, `te_sim_df`
  - `_make_test_df` -> `te_test_df_factory`, `te_test_df`
  - `_make_tensors` -> `te_tensor_factory`, `te_tensors`
  - `_make_splits`  -> `te_splits_factory`
  - `_make_df`      -> `te_position_df_factory`

TE-specific scoring scale (15 pts range) is preserved via the factory
defaults. Registers `unit`, `integration`, `e2e`, and `regression` markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is importable when tests are run from arbitrary cwd.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register TE test markers. Mirrors the project-wide pytest marker plan."""
    config.addinivalue_line("markers", "unit: fast unit tests (< 1s each)")
    config.addinivalue_line(
        "markers", "integration: multi-component tests (< 10s each)"
    )
    config.addinivalue_line("markers", "e2e: full-pipeline tests (< 60s each)")
    config.addinivalue_line(
        "markers", "regression: model quality thresholds"
    )
    config.addinivalue_line("markers", "slow: excluded from default run")


# ---------------------------------------------------------------------------
# TE target constant (re-exported from config so fixtures stay in sync)
# ---------------------------------------------------------------------------

from TE.te_config import TE_TARGETS  # noqa: E402


# ---------------------------------------------------------------------------
# Weekly-simulation fixtures (formerly `_make_sim_df`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def te_sim_df_factory():
    """Factory for synthetic weekly-simulation DataFrames with TE scoring.

    TE scoring scale is ~15 points (much lower than WR) — preserved here.
    """

    def _factory(n_weeks: int = 4, n_players: int = 15, seed: int = 42):
        rows = []
        rng = np.random.default_rng(seed)
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                fp = rng.random() * 15  # TEs score lower than WRs
                rows.append({
                    "week": week,
                    "player_id": f"TE{pid}",
                    "fantasy_points": fp,
                    "pred_ridge": fp + rng.standard_normal() * 2,
                    "pred_nn": fp + rng.standard_normal() * 3,
                })
        return pd.DataFrame(rows)

    return _factory


@pytest.fixture()
def te_sim_df(te_sim_df_factory):
    """Default-size weekly-simulation DataFrame (4 weeks x 15 players)."""
    return te_sim_df_factory()


# ---------------------------------------------------------------------------
# Ranking-metrics fixtures (formerly `_make_test_df`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def te_test_df_factory():
    """Factory for synthetic ranking-metrics DataFrames with TE scoring scale."""

    def _factory(n_weeks: int = 3, n_players: int = 15, seed: int = 42):
        rows = []
        rng = np.random.default_rng(seed)
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                rows.append({
                    "week": week,
                    "player_id": f"TE{pid}",
                    "pred_total": rng.random() * 15,
                    "fantasy_points": rng.random() * 15,
                })
        return pd.DataFrame(rows)

    return _factory


@pytest.fixture()
def te_test_df(te_test_df_factory):
    """Default-size ranking-metrics DataFrame (3 weeks x 15 players)."""
    return te_test_df_factory()


# ---------------------------------------------------------------------------
# Loss-function tensor fixtures (formerly `_make_tensors`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def te_tensor_factory():
    """Factory for random TE (preds, targets) tensor dicts used by loss tests."""

    def _factory(n: int = 10, seed: int = 42):
        # Fresh generator per call keeps fixtures reproducible regardless of order.
        gen = torch.Generator().manual_seed(seed)
        preds = {t: torch.randn(n, generator=gen) for t in TE_TARGETS}
        preds["total"] = torch.randn(n, generator=gen)
        targets = {t: torch.randn(n, generator=gen) for t in TE_TARGETS}
        targets["total"] = torch.randn(n, generator=gen)
        return preds, targets

    return _factory


@pytest.fixture()
def te_tensors(te_tensor_factory):
    """Default-size TE tensor dicts (n=10)."""
    return te_tensor_factory()


# ---------------------------------------------------------------------------
# NaN-filling splits fixture (formerly `_make_splits`)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def te_splits_factory():
    """Factory for (train, val, test) DataFrames used by fill_te_nans tests."""

    def _factory(train_vals, val_vals, test_vals, col: str = "feat1"):
        train = pd.DataFrame({col: train_vals})
        val = pd.DataFrame({col: val_vals})
        test = pd.DataFrame({col: test_vals})
        return train, val, test

    return _factory


# ---------------------------------------------------------------------------
# Position-filter DataFrame fixture (formerly `_make_df` in test_te_data.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def te_position_df_factory():
    """Factory for position-encoded DataFrames used by filter_to_te tests."""

    def _factory(positions: list[str], has_pos_cols: bool = True):
        data = {"position": positions, "receiving_yards": range(len(positions))}
        if has_pos_cols:
            data.update({
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            })
        return pd.DataFrame(data)

    return _factory


# ---------------------------------------------------------------------------
# Tiny E2E dataset fixture (used by test_te_pipeline_e2e and test_te_regression)
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

    Shared by `te_tiny_splits` and callable directly for tests that need
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
                rows.append({
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                    "position": "TE",
                    "recent_team": team,
                    "opponent_team": opp,
                    "fantasy_points": fp,
                    "fantasy_points_floor": fp * 0.8,
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
                })

    df = pd.DataFrame(rows)
    df = build_features(df)

    # Season-based split: 2022 train, 2023 first half val, 2023 second half test.
    train_df = df[df["season"] == 2022].copy()
    val_df = df[(df["season"] == 2023) & (df["week"] <= 9)].copy()
    test_df = df[(df["season"] == 2023) & (df["week"] > 9)].copy()
    return train_df, val_df, test_df
