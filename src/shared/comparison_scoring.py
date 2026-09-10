"""Scoring contract for statistics projected by both models and experts.

These are output components, not training input features. Every column in a
position's comparison uses this same set and the same observed-stat total.
"""

import numpy as np
import pandas as pd

from src.shared.aggregate_targets import (
    DST_TARGETS,
    K_TARGETS,
    POSITION_TARGET_MAP,
    predictions_to_fantasy_points,
)

ACTUAL_BASIS = "shared_projected_components_v1"
# NFL.com publishes bucket-scored kicker totals, without the made-yardage and
# miss projections required by our K heads. Those totals cannot enter this
# comparison. ESPN supplies all four K components.
EXCLUDED_SOURCES = {
    "K": {"nflcom": "NFL.com does not supply matching field-goal yardage and miss projections."}
}


def scoring_components(position: str) -> tuple[str, ...]:
    """The fixed component intersection for a position's comparable sources."""
    if position == "K":
        return K_TARGETS
    if position == "DST":
        return DST_TARGETS
    return tuple(POSITION_TARGET_MAP[position])


def score_actual_components(frame, position, scoring="ppr", *, prefix="") -> pd.Series:
    """Score observed components; an absent/unknown component is not zero.

    Serving stores ``actual_<target>``; pipeline frames store ``<target>``. Neither
    path may fall back to full fantasy actuals, which include unprojected stats.
    """
    components = scoring_components(position)
    if any(f"{prefix}{name}" not in frame for name in components):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = {
        name: pd.to_numeric(frame[f"{prefix}{name}"], errors="coerce").to_numpy(dtype=float)
        for name in components
    }
    valid = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    # Replace unknowns only during arithmetic (DST digitizes PA/YA); the result
    # is masked back to unknown so missing actuals cannot produce a real score.
    total = predictions_to_fantasy_points(
        position, {name: np.where(valid, value, 0) for name, value in values.items()}, scoring
    )
    return pd.Series(np.where(valid, total, np.nan), index=frame.index)
