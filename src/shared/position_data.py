"""Shared data helpers for QB/RB/WR/TE.

Each of those positions has a near-identical ``data.py``: a
``filter_to_position`` (filter rows + drop ``pos_*`` one-hot columns) and,
for RB/WR/TE, a ``compute_team_<pos>_totals`` aggregator over
``(recent_team, season, week)``. This module factors the duplicated logic
into a generic implementation so per-position ``data.py`` files only have
to declare the position code and the aggregation column list.

K and DST data.py have genuinely position-specific workflows (PBP
reconstruction for K, team-level pre-build for DST) and intentionally do
not consume the filtering helpers above.

The offline-diagnostics helpers at the bottom (``load_position_frames``,
``prepare_native_ablation``) load each position's train/val/test frames the
way its ``run()`` does — the shared parquet splits for QB/RB/WR/TE and the
native K/DST builders — so ablations and analyses can hand explicit frames
to ``run_pipeline`` without re-implementing a position's loader.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.config import SPLITS_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS

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


# --------------------------------------------------------------------------- #
# Position-native train/validation/test frames for offline diagnostics
# --------------------------------------------------------------------------- #


def _native_frame(pos: str) -> pd.DataFrame:
    if pos == "K":
        from src.k.data import load_data
        from src.k.features import compute_features
        from src.k.targets import compute_targets

        frame = compute_targets(load_data())
    elif pos == "DST":
        from src.dst.data import build_data
        from src.dst.features import compute_features
        from src.dst.targets import compute_targets

        frame = compute_targets(build_data())
    else:
        raise ValueError(f"No native data loader for position: {pos}")
    compute_features(frame)
    return frame


def _native_splits(pos: str, frame: pd.DataFrame):
    if pos == "K":
        from src.k.data import season_split

        return season_split(frame)
    return tuple(
        frame[frame["season"].isin(seasons)].copy()
        for seasons in (TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)
    )


def load_position_frames(pos: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test frames exactly as each position's run() does."""
    if pos in ("QB", "RB", "WR", "TE"):
        return (
            pd.read_parquet(f"{SPLITS_DIR}/train.parquet"),
            pd.read_parquet(f"{SPLITS_DIR}/val.parquet"),
            pd.read_parquet(f"{SPLITS_DIR}/test.parquet"),
        )
    if pos in ("K", "DST"):
        return _native_splits(pos, _native_frame(pos))
    raise ValueError(f"Unknown position: {pos}")


def prepare_native_ablation(pos: str):
    """Return native frames and the same runtime config used by K/DST run().

    Capture K's kick history before training-row filtering, as its canonical
    runner does. KEEP and CUT can then pass distinct frames to run_pipeline
    without losing the nested attention history or reloading uncut splits.
    """
    from src.shared.registry import get_config

    frame = _native_frame(pos)
    cfg = dict(get_config(pos))
    if pos == "K":
        from src.k.data import load_kicks
        from src.k.run_pipeline import _build_kick_history_closure

        cfg["attn_history_builder_fn"] = _build_kick_history_closure(cfg, load_kicks(frame))
    return _native_splits(pos, frame), cfg
