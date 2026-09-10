"""Keep comparison truth separate from full-fantasy diagnostic labels."""

import numpy as np
import pandas as pd

from src.shared.comparison_scoring import score_actual_components, scoring_components

PROJECTED_ACTUAL = "actual_projected_total"
PROJECTED_METADATA = "actual_projected_total_metadata"
PROJECTED_METADATA_BY_POSITION = "actual_projected_total_metadata_by_position"


def comparison_actuals(frame: pd.DataFrame, position: str, scoring: str = "ppr") -> pd.Series:
    """Reuse verified pipeline truth only for its exact component/format contract."""
    metadata = (frame.attrs.get(PROJECTED_METADATA_BY_POSITION) or {}).get(position)
    metadata = metadata or frame.attrs.get(PROJECTED_METADATA) or {}
    targets = list(scoring_components(position))
    verified = (
        PROJECTED_ACTUAL in frame
        and isinstance(metadata, dict)
        and metadata.get("basis") == "configured_target_aggregation_v1"
        and set(metadata.get("targets") or ()) == set(targets)
    )
    if verified and metadata.get("scoring_format") == scoring:
        values = pd.to_numeric(frame[PROJECTED_ACTUAL], errors="coerce")
        return values.where(np.isfinite(values))
    values = score_actual_components(frame, position, scoring)
    if verified:
        # The producer checked source stats before target fillna. Re-scoring
        # prepared columns in another format must not revive unavailable rows.
        values = values.where(np.isfinite(pd.to_numeric(frame[PROJECTED_ACTUAL], errors="coerce")))
    return values


def attach_comparison_actuals(
    frame: pd.DataFrame, position: str, scoring: str = "ppr"
) -> pd.DataFrame:
    """Return a reporting copy; preserve the caller's full fantasy-point column."""
    out = frame.copy()
    out[PROJECTED_ACTUAL] = comparison_actuals(frame, position, scoring)
    out.attrs[PROJECTED_METADATA] = {
        "basis": "configured_target_aggregation_v1",
        "targets": list(scoring_components(position)),
        "scoring_format": scoring,
    }
    if PROJECTED_METADATA_BY_POSITION in out.attrs:
        out.attrs[PROJECTED_METADATA_BY_POSITION] = {
            **out.attrs[PROJECTED_METADATA_BY_POSITION],
            position: out.attrs[PROJECTED_METADATA],
        }
    return out
