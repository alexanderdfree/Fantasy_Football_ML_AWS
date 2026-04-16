"""Shared pytest fixtures and marker registration for QB/tests/.

Consolidates helpers previously duplicated across test files:
- make_sim_df: simulation DataFrame for backtest tests (QB scoring scale, ~25 pts)
- make_test_df: ranking DataFrame for evaluation tests
- make_tensors: pytorch tensor dicts for loss/training tests
- make_splits: train/val/test tuple for NaN-fill tests
- make_df: position-encoded DataFrame for filter_to_qb tests

All fixtures seed numpy with seed=42 for determinism.
Markers are registered in pytest_configure so the unit works standalone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure project root is importable when tests run from QB/tests/
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from QB.qb_config import QB_TARGETS  # noqa: E402 -- needs sys.path tweak above


def pytest_configure(config):
    """Register markers locally so pytest doesn't warn about unregistered ones."""
    markers = [
        "unit: fast isolated test (<=1s), no external I/O, no training loops",
        "integration: hits run_pipeline or shared training loop",
        "e2e: full-pipeline smoke test (synthetic data)",
        "regression: numerical-performance assertion (MAE thresholds)",
    ]
    for m in markers:
        config.addinivalue_line("markers", m)


# ---------------------------------------------------------------------------
# make_sim_df — simulation DataFrame for backtest tests
# ---------------------------------------------------------------------------

def _build_sim_df(n_weeks=4, n_players=15, seed=42):
    """Create a test DataFrame for run_weekly_simulation."""
    rows = []
    np.random.seed(seed)
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            fp = np.random.rand() * 25  # QBs score higher than RBs
            rows.append({
                "week": week,
                "player_id": f"QB{pid}",
                "fantasy_points": fp,
                "pred_ridge": fp + np.random.randn() * 2,
                "pred_nn": fp + np.random.randn() * 3,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def make_sim_df():
    """Factory fixture: call make_sim_df(n_weeks=..., n_players=...)."""
    return _build_sim_df


# ---------------------------------------------------------------------------
# make_test_df — ranking DataFrame for evaluation tests
# ---------------------------------------------------------------------------

def _build_test_df(n_weeks=3, n_players=15, seed=42):
    rows = []
    np.random.seed(seed)
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            rows.append({
                "week": week,
                "player_id": f"QB{pid}",
                "pred_total": np.random.rand() * 25,
                "fantasy_points": np.random.rand() * 25,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def make_test_df():
    return _build_test_df


# ---------------------------------------------------------------------------
# make_tensors — tensor dicts for MultiTargetLoss tests
# ---------------------------------------------------------------------------

def _build_tensors(n=10, seed=42):
    torch.manual_seed(seed)
    preds = {t: torch.randn(n) for t in QB_TARGETS}
    preds["total"] = torch.randn(n)
    targets = {t: torch.randn(n) for t in QB_TARGETS}
    targets["total"] = torch.randn(n)
    return preds, targets


@pytest.fixture
def make_tensors():
    return _build_tensors


# ---------------------------------------------------------------------------
# make_splits — (train, val, test) tuples for fill_qb_nans tests
# ---------------------------------------------------------------------------

def _build_splits(train_vals, val_vals, test_vals, col="feat1"):
    train = pd.DataFrame({col: train_vals})
    val = pd.DataFrame({col: val_vals})
    test = pd.DataFrame({col: test_vals})
    return train, val, test


@pytest.fixture
def make_splits():
    return _build_splits


# ---------------------------------------------------------------------------
# make_df — position-encoded DataFrame for filter_to_qb tests
# ---------------------------------------------------------------------------

def _build_df(positions, has_pos_cols=True):
    data = {"position": positions, "passing_yards": range(len(positions))}
    if has_pos_cols:
        data.update({
            "pos_QB": [1 if p == "QB" else 0 for p in positions],
            "pos_RB": [1 if p == "RB" else 0 for p in positions],
            "pos_WR": [1 if p == "WR" else 0 for p in positions],
            "pos_TE": [1 if p == "TE" else 0 for p in positions],
        })
    return pd.DataFrame(data)


@pytest.fixture
def make_df():
    return _build_df
