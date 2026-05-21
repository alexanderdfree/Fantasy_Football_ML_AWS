"""RB data helpers — thin wrappers around shared position helpers."""

import pandas as pd

from src.shared.position_data import compute_team_position_totals
from src.shared.position_data import filter_to_position as _filter_to_position

_TEAM_RB_AGGREGATIONS = {
    "team_rb_carries": ("carries", "sum"),
    "team_rb_targets": ("targets", "sum"),
}


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to RB rows only.

    Must be called AFTER build_features() and AFTER temporal_split()
    so all team-level and opponent-level features are correctly computed
    from the full-position dataset.
    """
    return _filter_to_position(df, "RB")


def compute_team_rb_totals(full_rb_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level RB totals (carries + targets) for share features.

    Args:
        full_rb_df: All RB rows (before min-games filter), from the general pipeline.
    """
    return compute_team_position_totals(full_rb_df, _TEAM_RB_AGGREGATIONS)
