"""Shared fixtures for QB tests — installs the standard position fixtures
bound to QB's scoring scale and target list. Generic factories live in
``tests/shared/position_fixtures.py``.

QB historically refers to the position-df factory as ``make_df`` rather than
``make_position_df``, so we pass the name through ``position_df_fixture_name``.
"""

import pandas as pd
import pytest

from src.qb.config import POSITION_CONFIG
from tests.shared.position_fixtures import (
    register_position_markers,
    register_standard_fixtures,
)

# QBs score higher than any other skill position (~25-pt scale).
SCORING_SCALE = 25


def pytest_configure(config):
    register_position_markers(config)


register_standard_fixtures(
    globals(),
    scoring_scale=SCORING_SCALE,
    id_prefix="QB",
    targets=POSITION_CONFIG.targets,
    stat_col="passing_yards",
    position_df_fixture_name="make_df",
)


def _build_qb_row(**overrides) -> pd.DataFrame:
    """Single-row QB DataFrame with sensible defaults; fantasy_points auto-computed."""
    defaults = {
        "passing_yards": 250,
        "rushing_yards": 20,
        "receiving_yards": 0,
        "receptions": 0,
        "passing_tds": 2,
        "rushing_tds": 0,
        "receiving_tds": 0,
        "interceptions": 1,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "fantasy_points": 0.0,
    }
    defaults.update(overrides)
    if "fantasy_points" not in overrides:
        fp = (
            defaults["passing_yards"] * 0.04
            + defaults["rushing_yards"] * 0.1
            + defaults["receiving_yards"] * 0.1
            + defaults["receptions"] * 1.0
            + defaults["passing_tds"] * 4
            + (defaults["rushing_tds"] + defaults["receiving_tds"]) * 6
            + defaults["interceptions"] * -2
            + (
                defaults["sack_fumbles_lost"]
                + defaults["rushing_fumbles_lost"]
                + defaults["receiving_fumbles_lost"]
            )
            * -2
        )
        defaults["fantasy_points"] = fp
    return pd.DataFrame([defaults])
