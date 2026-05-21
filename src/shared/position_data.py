"""Shared data helpers for QB/RB/WR/TE.

Each of those positions has a near-identical ``data.py``: a
``filter_to_position`` (filter rows + drop ``pos_*`` one-hot columns) and,
for RB/WR/TE, a ``compute_team_<pos>_totals`` aggregator over
``(recent_team, season, week)``. This module factors the duplicated logic
into a generic implementation so per-position ``data.py`` files only have
to declare the position code and the aggregation column list.

K and DST data.py have genuinely position-specific workflows (PBP
reconstruction for K, team-level pre-build for DST) and intentionally do
not consume this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

# Position encoding columns dropped by ``filter_to_position`` after filtering.
# Kept centralized so adding a new skill position only requires updating
# this list in one place.
_POS_ONE_HOT_COLS = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]


def drop_position_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``pos_QB`` / ``pos_RB`` / ``pos_WR`` / ``pos_TE`` columns in-place.

    Returns the same DataFrame for chaining. Safe to call when the columns
    are absent (no-op).
    """
    df.drop(columns=[c for c in _POS_ONE_HOT_COLS if c in df.columns], inplace=True)
    return df


def filter_to_position(df: pd.DataFrame, pos_code: str) -> pd.DataFrame:
    """Filter a featured DataFrame to a single position and strip ``pos_*`` columns.

    Must be called AFTER ``build_features()`` and AFTER ``temporal_split()``
    so team-level / opponent-level features are computed from the full
    multi-position frame.
    """
    pos_df = df[df["position"] == pos_code].copy()
    drop_position_encodings(pos_df)
    return pos_df


def compute_team_position_totals(
    full_pos_df: pd.DataFrame,
    aggregations: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Aggregate per-position team totals grouped by ``(recent_team, season, week)``.

    ``aggregations`` maps the output column name to a ``(source_column,
    agg_func)`` pair, mirroring pandas' ``DataFrame.agg(**kwargs)`` form.
    E.g. ``{"team_rb_carries": ("carries", "sum")}``.
    """
    return (
        full_pos_df.groupby(["recent_team", "season", "week"])
        .agg(**dict(aggregations))
        .reset_index()
    )
