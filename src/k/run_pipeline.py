"""End-to-end K (Kicker) position model pipeline.

Uses PBP-reconstructed kicker data from 2015-2025 (post-PAT rule change).
Cross-season splits: train 2015-2023, val 2024, test 2025.

K is one of two positions (with DST) that loads its own data inside ``run()``
rather than receiving DataFrames from the shared splits. The
``attn_history_builder_fn`` closure also has to be built at runtime because
it captures the ``kicks_df`` produced by ``load_kicks(k_df)``. The factory
provides the rest of the CONFIG dict; only the runtime-dependent
``attn_history_builder_fn`` is injected here.
"""

import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.k.config import POSITION_CONFIG
from src.k.data import load_data, load_kicks, season_split
from src.k.features import build_nested_kick_history, compute_features
from src.k.targets import compute_targets
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.shared.position_pipeline import build_pipeline_config
from src.shared.run_pipeline_factory import cli_main
from src.training.context import runner_context
from src.training.contracts import DatasetSplits

# K's CONFIG omits the runtime-injected attn_history_builder_fn; run() fills
# it in after kicks_df is loaded.
CONFIG = build_pipeline_config("K", POSITION_CONFIG)


def provide_dataset(cfg, *, cross_validation=False) -> DatasetSplits:
    """Keep kicker reconstruction and nested per-kick state at the provider boundary."""
    print("Loading kicker data...")
    k_df = compute_targets(load_data())
    compute_features(k_df)
    kicks_df = load_kicks(k_df)
    if cross_validation:
        train_df = k_df[k_df["season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()
        val_df = None
        test_df = k_df[k_df["season"].isin(TEST_SEASONS)].copy()
    else:
        train_df, val_df, test_df = season_split(k_df)
    return DatasetSplits(
        train_df,
        val_df,
        test_df,
        {
            "attn_history_builder_fn": _build_kick_history_closure(cfg, kicks_df),
            "attn_kick_stats": list(cfg.get("attn_kick_stats", POSITION_CONFIG.attn_kick_stats)),
        },
    )


@runner_context
def run(seed=42, config=None, *, context=None):
    cfg = dict(config if config is not None else CONFIG)
    dataset = provide_dataset(cfg)
    cfg.update(dataset.bindings)
    return run_pipeline(
        "K", cfg, *dataset.frames, seed, **({"context": context} if context is not None else {})
    )


def _build_kick_history_closure(cfg, kicks_df):
    """Bind the per-kick nested-history builder over ``kicks_df`` (shared by
    ``run`` and ``run_cv``). Reads the attention-window shape from ``cfg`` so a
    tuner override of ``attn_max_games`` etc. takes effect."""
    return functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=list(cfg.get("attn_kick_stats", POSITION_CONFIG.attn_kick_stats)),
        max_games=cfg.get("attn_max_games", POSITION_CONFIG.attn_max_games),
        max_kicks_per_game=cfg.get(
            "attn_max_kicks_per_game", POSITION_CONFIG.attn_max_kicks_per_game
        ),
    )


@runner_context
def run_cv(seed=42, config=None, *, context=None):
    cfg = dict(config if config is not None else CONFIG)
    dataset = provide_dataset(cfg, cross_validation=True)
    cfg.update(dataset.bindings)
    return run_cv_pipeline(
        "K",
        cfg,
        dataset.train,
        dataset.test,
        seed,
        **({"context": context} if context is not None else {}),
    )


if __name__ == "__main__":
    cli_main(
        position_name="K",
        default_config=CONFIG,
        run_fn=run,
        run_cv_fn=run_cv,
    )
