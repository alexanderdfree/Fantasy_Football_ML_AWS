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

ACTUAL_BASIS = "shared_projected_components_v2"
EXCLUDED_COMPONENTS = {
    "DST": {"points_allowed": "Scoreboard, ESPN and RotoWire points-allowed definitions differ."}
}
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
        return tuple(target for target in DST_TARGETS if target != "points_allowed")
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
    scored = {name: np.where(valid, value, 0) for name, value in values.items()}
    if position == "DST":
        # 21 points is the zero-bonus PA tier. This removes the non-shared
        # component from both sides without altering normal fantasy scoring.
        scored["points_allowed"] = np.full(len(frame), 21.0)
    total = predictions_to_fantasy_points(position, scored, scoring)
    return pd.Series(np.where(valid, total, np.nan), index=frame.index)


def comparison_actuals(frame, position, scoring="ppr", *, prefix="") -> pd.Series:
    """Use certified pre-imputation truth and preserve its missing-value mask."""
    metadata = (frame.attrs.get("actual_projected_total_metadata_by_position") or {}).get(position)
    metadata = metadata or frame.attrs.get("actual_projected_total_metadata") or {}
    verified = (
        "actual_projected_total" in frame
        and isinstance(metadata, dict)
        and metadata.get("basis") == "configured_target_aggregation_v1"
        and set(metadata.get("targets") or ()) == set(scoring_components(position))
    )
    if verified:
        observed = pd.to_numeric(frame["actual_projected_total"], errors="coerce")
        valid = np.isfinite(observed)
        if metadata.get("scoring_format") == scoring:
            return observed.where(valid)
    values = score_actual_components(frame, position, scoring, prefix=prefix)
    return values.where(valid) if verified else values


def comparison_model_totals(
    frame: pd.DataFrame, position: str, scoring="ppr", *, rescore=False
) -> pd.DataFrame:
    """Rebuild DST comparison totals from raw heads, never rounded native totals.

    Pipeline totals retain ordinary fantasy scoring. Other positions already
    share that total's component set; DST comparison omits points allowed.
    """
    out = frame.copy()
    if position == "DST" or rescore:
        for column in frame:
            if column.startswith("pred_") and column.endswith("_total"):
                out[column] = score_actual_components(frame, position, scoring, prefix=column[:-5])
    return out
