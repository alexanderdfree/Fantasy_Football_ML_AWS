"""WR data helpers — thin wrappers around shared position helpers."""

import pandas as pd

from src.shared.position_data import compute_team_position_totals
from src.shared.position_data import filter_to_position as _filter_to_position

_TEAM_WR_AGGREGATIONS = {
    "team_wr_targets": ("targets", "sum"),
}


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to WR rows only."""
    return _filter_to_position(df, "WR")


def compute_team_wr_totals(full_wr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level WR totals (targets) for share features."""
    return compute_team_position_totals(full_wr_df, _TEAM_WR_AGGREGATIONS)
