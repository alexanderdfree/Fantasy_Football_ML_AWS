"""Shared fixtures for WR tests — thin wrappers over src.shared.tests.position_fixtures.

Generic factories are imported from ``src.shared.tests.position_fixtures``;
this conftest binds them to the WR scoring scale (~20) and targets, and
keeps the WR-specific ``wr_player_games_factory`` feature-input builder.
"""

import sys
from pathlib import Path

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
from src.WR.wr_config import WR_TARGETS  # noqa: E402

# WR fantasy points typically land in the 0-20 PPR range.
WR_SCORING_SCALE = 20


def pytest_configure(config):
    register_position_markers(
        config,
        extra=[("slow", "excluded from the default local run")],
    )


# ---------------------------------------------------------------------------
# Generic WR fixtures (WR scale, WR prefix, default_rng for backwards-compat)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def wr_sim_df_factory():
    """Factory producing a WR-scale simulation DataFrame for backtest tests."""

    def _make(n_weeks: int = 4, n_players: int = 15, seed: int = 42):
        return _make_sim_df(
            WR_SCORING_SCALE,
            n_weeks,
            n_players,
            seed,
            id_prefix="WR",
            rng_kind="default",
        )

    return _make


@pytest.fixture
def wr_sim_df(wr_sim_df_factory):
    """Default WR simulation DataFrame (4 weeks x 15 players, seed=42)."""
    return wr_sim_df_factory()


@pytest.fixture(scope="session")
def wr_test_df_factory():
    """Factory producing WR ranking test DataFrames."""

    def _make(n_weeks: int = 3, n_players: int = 15, seed: int = 42):
        return _make_test_df(
            WR_SCORING_SCALE,
            n_weeks,
            n_players,
            seed,
            id_prefix="WR",
            rng_kind="default",
        )

    return _make


@pytest.fixture
def wr_test_df(wr_test_df_factory):
    """Default WR test DataFrame (3 weeks x 15 players, seed=42)."""
    return wr_test_df_factory()


@pytest.fixture(scope="session")
def wr_nn_tensors_factory():
    """Factory producing (preds, targets) tensor dicts for MultiTargetLoss tests."""

    def _make(n: int = 10, seed: int = 42):
        return _make_tensors(WR_TARGETS, n=n, seed=seed)

    return _make


@pytest.fixture
def wr_nn_tensors(wr_nn_tensors_factory):
    """Default WR (preds, targets) pair (n=10, seed=42)."""
    return wr_nn_tensors_factory()


@pytest.fixture(scope="session")
def wr_nan_splits_factory():
    """Factory producing (train, val, test) DataFrames for fill_wr_nans tests."""
    return _make_splits


@pytest.fixture(scope="session")
def wr_position_df_factory():
    """Factory for DataFrames used by filter_to_wr tests (position + pos_* cols)."""

    def _make(positions, has_pos_cols: bool = True):
        return _make_position_df(positions, stat_col="receiving_yards", has_pos_cols=has_pos_cols)

    return _make


# ---------------------------------------------------------------------------
# WR-specific fixtures (not generic across positions)
# ---------------------------------------------------------------------------


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
        return pd.DataFrame(
            {
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
            }
        )

    return _make
