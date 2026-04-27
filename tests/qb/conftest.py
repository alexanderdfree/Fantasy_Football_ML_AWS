"""Shared fixtures for QB tests — thin wrappers over tests.shared.position_fixtures.

The generic pieces (``make_sim_df``, ``make_test_df``, ``make_tensors``,
``make_splits``, ``make_df``) live in ``tests/shared/position_fixtures.py``.
This conftest only binds the QB-specific scale (~25 fantasy points) and
target list to those factories.
"""

import pytest

from src.QB.qb_config import QB_TARGETS
from tests.shared.position_fixtures import (
    make_position_df as _make_position_df,
)
from tests.shared.position_fixtures import (
    make_sim_df as _make_sim_df,
)
from tests.shared.position_fixtures import (
    make_splits as _make_splits,
)
from tests.shared.position_fixtures import (
    make_tensors as _make_tensors,
)
from tests.shared.position_fixtures import (
    make_test_df as _make_test_df,
)
from tests.shared.position_fixtures import (
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
