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


def _build_wr_row(**overrides) -> pd.DataFrame:
    """Single-row WR DataFrame with sensible defaults; fantasy_points auto-computed."""
    defaults = {
        "receiving_yards": 80,
        "rushing_yards": 0,
        "receptions": 6,
        "targets": 8,
        "receiving_tds": 1,
        "rushing_tds": 0,
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
            defaults["receiving_yards"] * 0.1
            + defaults["receptions"] * 1.0  # PPR
            + defaults["rushing_yards"] * 0.1
            + (defaults["receiving_tds"] + defaults["rushing_tds"]) * 6
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
def make_wr_row():
    """Factory for single-row WR target inputs."""
    return _build_wr_row


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
