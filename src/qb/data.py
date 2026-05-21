"""QB data helpers — thin wrapper around shared position helpers.

Unlike ``src/rb/data.py`` / ``src/wr/data.py`` / ``src/te/data.py``, QB
intentionally exposes no ``compute_team_*_totals`` helper: there is only one
quarterback on the field at a time, so team-level "share of carries" / "share
of targets" features (the RB/WR/TE motivation) are degenerate at QB. If you
are copy-pasting from a sibling position to add a team-totals aggregator
here, stop — the omission is by design, not an oversight.
"""

import pandas as pd

from src.shared.position_data import filter_to_position as _filter_to_position


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to QB rows only."""
    return _filter_to_position(df, "QB")
