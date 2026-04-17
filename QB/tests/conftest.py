"""Shared fixtures for QB tests — thin wrappers over shared.tests.position_fixtures.

The generic pieces (``make_sim_df``, ``make_test_df``, ``make_tensors``,
``make_splits``, ``make_df``) live in ``shared/tests/position_fixtures.py``.
This conftest only binds the QB-specific scale (~25 fantasy points) and
target list to those factories.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from QB.qb_config import QB_TARGETS  # noqa: E402
from shared.tests.position_fixtures import (  # noqa: E402
    make_position_df as _make_position_df,
    make_sim_df as _make_sim_df,
    make_splits as _make_splits,
    make_tensors as _make_tensors,
    make_test_df as _make_test_df,
    register_position_markers,
)

# QBs score higher than any other skill position (~25-pt scale).
QB_SCORING_SCALE = 25


def pytest_configure(config):
    register_position_markers(config)


@pytest.fixture
def make_sim_df():
    def _make(n_weeks=4, n_players=15, seed=42):
        return _make_sim_df(QB_SCORING_SCALE, n_weeks, n_players, seed, id_prefix="QB")
    return _make


@pytest.fixture
def make_test_df():
    def _make(n_weeks=3, n_players=15, seed=42):
        return _make_test_df(QB_SCORING_SCALE, n_weeks, n_players, seed, id_prefix="QB")
    return _make


@pytest.fixture
def make_tensors():
    def _make(n=10, seed=42):
        return _make_tensors(QB_TARGETS, n=n, seed=seed)
    return _make


@pytest.fixture
def make_splits():
    return _make_splits


@pytest.fixture
def make_df():
    def _make(positions, has_pos_cols=True):
        return _make_position_df(positions, stat_col="passing_yards", has_pos_cols=has_pos_cols)
    return _make
