"""TE data helpers — thin wrappers around shared position helpers."""

import pandas as pd

from src.shared.position_data import compute_team_position_totals
from src.shared.position_data import filter_to_position as _filter_to_position

_TEAM_TE_AGGREGATIONS = {
    "team_te_targets": ("targets", "sum"),
}


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to TE rows only."""
    return _filter_to_position(df, "TE")


def compute_team_te_totals(full_te_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level TE totals (targets) for share features."""
    return compute_team_position_totals(full_te_df, _TEAM_TE_AGGREGATIONS)
