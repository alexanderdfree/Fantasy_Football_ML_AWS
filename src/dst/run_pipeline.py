"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2012-2023 / 2024 / 2025).

DST is one of two positions (with K) that loads its own data inside ``run()``;
the shared factory provides the CONFIG dict but the team-level data assembly
stays here because it doesn't fit the shared splits' player-level shape.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.dst.config import POSITION_CONFIG
from src.dst.data import build_data
from src.dst.features import compute_features
from src.dst.targets import compute_targets
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config

CONFIG = build_pipeline_config("DST", POSITION_CONFIG)


def run(seed=42, config=None):
    """Run the DST pipeline. ``config`` lets callers (e.g.
    ``src/tuning/tune_nn.py``) pass an overridden cfg dict per trial; mirrors
    the RB/QB/WR/TE shape. Defaults to the module-level ``CONFIG`` when
    omitted, preserving the existing caller contract.
    """
    # --- Build team-level D/ST data ---
    print("Building D/ST team-level data...")
    dst_df = build_data()
    print(f"  Built {len(dst_df)} team-week rows, {dst_df['team'].nunique()} teams")
    print(f"  Seasons: {sorted(dst_df['season'].unique())}")

    # Compute targets on full data (needed for feature computation)
    dst_df = compute_targets(dst_df)

    # Compute ALL features on full data before splitting
    print("Computing D/ST features on full dataset...")
    compute_features(dst_df)

    # --- Standard temporal split ---
    train_df = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val_df = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test_df = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    print(f"  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Explicit None check (not ``config or CONFIG``) so an empty dict ``{}``
    # falls through to the caller's intent rather than silently reverting to
    # CONFIG — matches K's pattern in src/k/run_pipeline.py.
    return run_pipeline(
        "DST",
        config if config is not None else CONFIG,
        train_df,
        val_df,
        test_df,
        seed,
    )


def run_cv(seed=42, config=None):
    """Expanding-window CV for DST. Self-loads team-level data (like ``run()``),
    then hands the full train+val frame plus a held-out 2025 test frame to the
    shared CV pipeline. Defined as a module-level function (not a factory
    closure) so the runpy-monkeypatch test pattern keeps working.
    """
    dst_df = build_data()
    dst_df = compute_targets(dst_df)
    compute_features(dst_df)
    full_df = dst_df[dst_df["season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()
    test_df = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return run_cv_pipeline("DST", config if config is not None else CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DST pipeline")
    parser.add_argument("--cv", action="store_true", help="Use expanding-window cross-validation")
    args = parser.parse_args()
    (run_cv if args.cv else run)()
