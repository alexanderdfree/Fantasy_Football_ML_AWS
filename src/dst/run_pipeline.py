"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2013-2023 / 2024 / 2025);
2012 is loaded for prior-season/rolling context only, not trained on.

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
from src.shared.run_pipeline_factory import cli_main
from src.training.context import runner_context
from src.training.contracts import DatasetSplits

CONFIG = build_pipeline_config("DST", POSITION_CONFIG)


def provide_dataset(cfg, *, cross_validation=False) -> DatasetSplits:
    """Supply team-level frames without flattening the DST data contract."""
    print("Building D/ST team-level data...")
    dst_df = compute_targets(build_data())
    compute_features(dst_df)
    train_seasons = TRAIN_SEASONS + VAL_SEASONS if cross_validation else TRAIN_SEASONS
    train_df = dst_df[dst_df["season"].isin(train_seasons)].copy()
    val_df = None if cross_validation else dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test_df = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return DatasetSplits(train_df, val_df, test_df, {})


@runner_context
def run(seed=42, config=None, *, context=None):
    cfg = config if config is not None else CONFIG
    dataset = provide_dataset(cfg)
    return run_pipeline(
        "DST", cfg, *dataset.frames, seed, **({"context": context} if context is not None else {})
    )


@runner_context
def run_cv(seed=42, config=None, *, context=None):
    cfg = config if config is not None else CONFIG
    dataset = provide_dataset(cfg, cross_validation=True)
    return run_cv_pipeline(
        "DST",
        cfg,
        dataset.train,
        dataset.test,
        seed,
        **({"context": context} if context is not None else {}),
    )


if __name__ == "__main__":
    cli_main(
        position_name="DST",
        default_config=CONFIG,
        run_fn=run,
        run_cv_fn=run_cv,
    )
