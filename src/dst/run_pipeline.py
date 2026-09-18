"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2013-2023 / 2024 / 2025);
2012 is loaded for prior-season/rolling context only, not trained on.

DST is one of two positions (with K) that loads its own data inside ``run()``;
the shared factory provides the CONFIG dict but the team-level data assembly
stays here because it doesn't fit the shared splits' player-level shape.
"""

import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.dst.config import POSITION_CONFIG
from src.dst.data import build_data, impute_context_from_train
from src.dst.features import compute_features
from src.dst.targets import compute_targets
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.shared.run_pipeline_factory import cli_main
from src.training.context import runner_context
from src.training.contracts import DatasetSplits

CONFIG = build_pipeline_config("DST", POSITION_CONFIG)


def _fill_fold_context(train_df, val_df, test_df, feature_cols, *, fill_nans_fn):
    """Fit context at the existing post-split feature-fill boundary."""
    frames = [
        impute_context_from_train(frame, fit_on=train_df) for frame in (train_df, val_df, test_df)
    ]
    return fill_nans_fn(*frames, feature_cols)


def with_fold_imputation(config):
    """Copy a native config so every preparation uses its own train frame."""
    cfg = dict(config)
    cfg["fill_nans_fn"] = functools.partial(_fill_fold_context, fill_nans_fn=cfg["fill_nans_fn"])
    return cfg


def provide_dataset(cfg, *, cross_validation=False) -> DatasetSplits:
    """Supply team-level frames without flattening the DST data contract.

    Cross-validation builds with ``impute_context=False`` so every shared CV
    fold, rolling origin and the final refit fits the context fills on its
    own training frame (``with_fold_imputation``); ``run()`` keeps the
    loader's default training-season fill.
    """
    print("Building D/ST team-level data...")
    dst_df = compute_targets(build_data(impute_context=False) if cross_validation else build_data())
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
    cfg = with_fold_imputation(config if config is not None else CONFIG)
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
