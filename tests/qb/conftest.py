"""Shared fixtures for QB tests — installs the standard position fixtures
bound to QB's scoring scale and target list. Generic factories live in
``tests/shared/position_fixtures.py``.

QB historically refers to the position-df factory as ``make_df`` rather than
``make_position_df``, so we pass the name through ``position_df_fixture_name``.
"""

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
