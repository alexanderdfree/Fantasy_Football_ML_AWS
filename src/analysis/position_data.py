"""Position-native train/validation/test frames for offline diagnostics."""

import pandas as pd

from src.config import SPLITS_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS


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
