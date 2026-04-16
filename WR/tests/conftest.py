"""Test fixtures and marker registration for WR/ tests.

Fixtures here replace the per-file ``_make_*`` helpers that used to be
duplicated across every WR test module. They all produce deterministic
output (seed=42) at WR-appropriate scoring scales.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register WR test markers so pytest -m <marker> works without warnings."""
    config.addinivalue_line("markers", "unit: fast isolated tests (< 1s each)")
    config.addinivalue_line(
        "markers", "integration: multi-component tests (< 10s each)"
    )
    config.addinivalue_line(
        "markers", "e2e: full-pipeline smokes (< 60s each)"
    )
    config.addinivalue_line(
        "markers", "regression: ML quality thresholds on held-out metrics"
    )
    config.addinivalue_line(
        "markers", "slow: excluded from the default local run"
    )


# ---------------------------------------------------------------------------
# Shared WR constants (imported from production config to avoid drift)
# ---------------------------------------------------------------------------

from WR.wr_config import WR_TARGETS, WR_LOSS_WEIGHTS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (factory-style so each test can override shape without mutating
# session-cached state)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wr_sim_df_factory():
    """Factory producing a WR-scale simulation DataFrame for backtest tests.

    WR PPR fantasy points ~= 0-25 range. Ridge noise std=2, NN noise std=3.
    """
    def _make(n_weeks: int = 4, n_players: int = 15, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                fp = rng.random() * 20  # WR scale: 0-20
                rows.append({
                    "week": week,
                    "player_id": f"WR{pid}",
                    "fantasy_points": fp,
                    "pred_ridge": fp + rng.standard_normal() * 2,
                    "pred_nn": fp + rng.standard_normal() * 3,
                })
        return pd.DataFrame(rows)
    return _make


@pytest.fixture
def wr_sim_df(wr_sim_df_factory):
    """Default WR simulation DataFrame (4 weeks × 15 players, seed=42)."""
    return wr_sim_df_factory()


@pytest.fixture(scope="session")
def wr_test_df_factory():
    """Factory producing WR ranking test DataFrames.

    Used by evaluation tests (compute_ranking_metrics, etc.). WR scoring
    scale: predictions and truth in [0, 20].
    """
    def _make(n_weeks: int = 3, n_players: int = 15, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for week in range(1, n_weeks + 1):
            for pid in range(1, n_players + 1):
                rows.append({
                    "week": week,
                    "player_id": f"WR{pid}",
                    "pred_total": rng.random() * 20,
                    "fantasy_points": rng.random() * 20,
                })
        return pd.DataFrame(rows)
    return _make


@pytest.fixture
def wr_test_df(wr_test_df_factory):
    """Default WR test DataFrame (3 weeks × 15 players, seed=42)."""
    return wr_test_df_factory()


@pytest.fixture(scope="session")
def wr_nn_tensors_factory():
    """Factory producing (preds, targets) tensor dicts for MultiTargetLoss tests.

    WR has 3 targets plus ``total``; tensors are zero-mean Gaussian.
    """
    def _make(n: int = 10, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        preds = {t: torch.randn(n, generator=g) for t in WR_TARGETS}
        preds["total"] = torch.randn(n, generator=g)
        targets = {t: torch.randn(n, generator=g) for t in WR_TARGETS}
        targets["total"] = torch.randn(n, generator=g)
        return preds, targets
    return _make


@pytest.fixture
def wr_nn_tensors(wr_nn_tensors_factory):
    """Default WR (preds, targets) pair (n=10, seed=42)."""
    return wr_nn_tensors_factory()


@pytest.fixture(scope="session")
def wr_nan_splits_factory():
    """Factory producing (train, val, test) DataFrames for fill_wr_nans tests."""
    def _make(train_vals, val_vals, test_vals, col: str = "feat1"):
        train = pd.DataFrame({col: train_vals})
        val = pd.DataFrame({col: val_vals})
        test = pd.DataFrame({col: test_vals})
        return train, val, test
    return _make


@pytest.fixture(scope="session")
def wr_player_games_factory():
    """Factory for multi-week WR game DataFrames used by feature-compute tests."""
    def _make(
        player_id: str = "W1",
        season: int = 2023,
        n_weeks: int = 5,
        receptions: int = 5,
        targets: int = 8,
        receiving_yards: int = 70,
        receiving_air_yards: int = 100,
        receiving_yards_after_catch: int = 30,
        receiving_epa: float = 2.0,
        receiving_first_downs: int = 3,
        recent_team: str = "KC",
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "player_id": [player_id] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "receptions": [receptions] * n_weeks,
            "targets": [targets] * n_weeks,
            "receiving_yards": [receiving_yards] * n_weeks,
            "receiving_air_yards": [receiving_air_yards] * n_weeks,
            "receiving_yards_after_catch": [receiving_yards_after_catch] * n_weeks,
            "receiving_epa": [receiving_epa] * n_weeks,
            "receiving_first_downs": [receiving_first_downs] * n_weeks,
            "recent_team": [recent_team] * n_weeks,
        })
    return _make


@pytest.fixture(scope="session")
def wr_position_df_factory():
    """Factory for DataFrames used by filter_to_wr tests (position + pos_* cols)."""
    def _make(positions, has_pos_cols: bool = True) -> pd.DataFrame:
        data = {"position": positions, "receiving_yards": range(len(positions))}
        if has_pos_cols:
            data.update({
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            })
        return pd.DataFrame(data)
    return _make
