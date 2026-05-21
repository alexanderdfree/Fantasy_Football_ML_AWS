"""Shared fixtures for WR tests — installs the standard position fixtures
bound to WR's scoring scale and target list. Generic factories live in
``tests/shared/position_fixtures.py``.

WR matches the QB/RB/K/DST pattern: the generic ``register_standard_fixtures``
helper installs ``make_sim_df`` / ``make_test_df`` / ``make_tensors`` /
``make_splits`` / ``make_position_df`` (plus default shortcuts ``sim_df`` /
``test_df``). Only the WR-specific ``wr_player_games_factory`` stays here.
"""

import pandas as pd
import pytest

from src.wr.config import POSITION_CONFIG
from tests.shared.position_fixtures import (
    register_position_markers,
    register_standard_fixtures,
)

# WR fantasy points typically land in the 0-20 PPR range.
SCORING_SCALE = 20


def pytest_configure(config):
    register_position_markers(
        config,
        extra=[("slow", "excluded from the default local run")],
    )


register_standard_fixtures(
    globals(),
    scoring_scale=SCORING_SCALE,
    id_prefix="WR",
    targets=POSITION_CONFIG.targets,
    stat_col="receiving_yards",
    rng_kind="default",
    install_default_shortcuts=True,
)


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
