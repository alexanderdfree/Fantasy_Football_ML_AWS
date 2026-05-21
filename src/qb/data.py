"""QB data helpers — thin wrapper around shared position helpers."""

import pandas as pd

from src.shared.position_data import filter_to_position as _filter_to_position


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to QB rows only."""
    return _filter_to_position(df, "QB")
