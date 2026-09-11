"""Keep comparison observation availability separate from training fills."""

import numpy as np
import pandas as pd

SOURCE_AVAILABLE = "actual_projected_source_available"
SOURCE_BASIS = "actual_projected_source_basis"
SOURCE_METADATA = "actual_projected_source_components_by_position"
ACTUAL_METADATA = "actual_projected_total_metadata"
ACTUAL_BASIS = "configured_target_aggregation_v1"

_DERIVED_SOURCES = {
    "fumbles_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "fg_yard_points": ("fg_yards_made",),
    "pat_points": ("pat_made",),
    "fg_misses": ("fg_missed",),
    "xp_misses": ("pat_missed",),
}


def _source_basis(position):
    from src.shared.comparison_scoring import ACTUAL_BASIS as comparison_basis

    return f"{position}:{comparison_basis}"


def _actual_metadata(position):
    from src.shared.comparison_scoring import scoring_components

    return {
        "basis": ACTUAL_BASIS,
        "targets": list(scoring_components(position)),
        "scoring_format": "ppr",
    }


def comparison_source_availability(frame: pd.DataFrame, position: str) -> pd.Series:
    """Capture finite raw inputs before a canonical target builder fills them."""
    from src.shared.comparison_scoring import scoring_components

    components = scoring_components(position)
    required = {source for name in components for source in _DERIVED_SOURCES.get(name, (name,))}
    if not required.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    available = pd.Series(True, index=frame.index)
    for column in sorted(required):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        available &= np.isfinite(values)
    if SOURCE_AVAILABLE in frame and SOURCE_BASIS in frame:
        # Earlier preprocessing may already have replaced unknown raw values.
        # Once unknown, those observations must remain unavailable for reporting.
        certified = frame[SOURCE_BASIS].eq(_source_basis(position)).fillna(False)
        available &= ~certified | frame[SOURCE_AVAILABLE].eq(True).fillna(False)
    return available


def preserve_comparison_source_availability(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain a non-feature mask on mixed-position rows before preprocessing fills."""
    from src.shared.comparison_scoring import scoring_components

    available = pd.Series(False, index=frame.index)
    basis = pd.Series("", index=frame.index)
    metadata = dict(frame.attrs.get(SOURCE_METADATA, {}))
    for position in ("QB", "RB", "WR", "TE", "K", "DST"):
        selected = frame["position"].eq(position)
        if selected.any():
            available.loc[selected] = comparison_source_availability(frame.loc[selected], position)
            basis.loc[selected] = _source_basis(position)
            metadata[position] = list(scoring_components(position))
    frame[SOURCE_AVAILABLE] = available
    frame[SOURCE_BASIS] = basis
    frame.attrs[SOURCE_METADATA] = metadata
    return frame


def attach_comparison_actuals(
    frame: pd.DataFrame, position: str, source_available: pd.Series
) -> pd.DataFrame:
    """Attach certified reporting totals after targets, without editing training values."""
    from src.shared.comparison_scoring import score_actual_components, scoring_components

    if not frame.index.equals(source_available.index):
        raise ValueError("Comparison availability must stay aligned through target construction")
    components = list(scoring_components(position))
    available = source_available.eq(True).fillna(False)
    frame[SOURCE_AVAILABLE] = available
    frame[SOURCE_BASIS] = _source_basis(position)
    frame["actual_projected_total"] = score_actual_components(frame, position).where(available)
    frame.attrs[SOURCE_METADATA] = {
        **frame.attrs.get(SOURCE_METADATA, {}),
        position: components,
    }
    frame.attrs[ACTUAL_METADATA] = _actual_metadata(position)
    return frame


def restore_comparison_actuals(frame: pd.DataFrame) -> None:
    """Recreate reporting metadata after feature merges that discard pandas attrs."""
    from src.shared.comparison_scoring import score_actual_components

    if SOURCE_AVAILABLE not in frame or SOURCE_BASIS not in frame:
        return
    metadata = {}
    for position in ("QB", "RB", "WR", "TE", "K", "DST"):
        selected = frame[SOURCE_BASIS].eq(_source_basis(position)).fillna(False)
        if not selected.any():
            continue
        rows = frame.loc[selected]
        total = score_actual_components(rows, position).where(rows[SOURCE_AVAILABLE].eq(True))
        frame.loc[selected, "actual_projected_total"] = total
        metadata[position] = _actual_metadata(position)
    if len(metadata) == 1:
        frame.attrs[ACTUAL_METADATA] = next(iter(metadata.values()))
    elif metadata:
        frame.attrs.pop(ACTUAL_METADATA, None)
        frame.attrs["actual_projected_total_metadata_by_position"] = metadata
