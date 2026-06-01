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

# K's CONFIG omits the runtime-injected attn_history_builder_fn; run() fills
# it in after kicks_df is loaded.
CONFIG = build_pipeline_config("K", POSITION_CONFIG)


def run(seed=42, config=None):
    """Run the K pipeline. ``config`` lets callers (e.g. ``src/tuning/tune_nn.py``)
    pass an overridden cfg dict per trial; we still inject the runtime-only
    ``attn_history_builder_fn`` closure on top because it captures ``kicks_df``
    loaded inside this function and can't be pre-baked into the static
    ``CONFIG`` dict.
    """
    # --- Load and prepare kicker data ---
    print("Loading kicker data...")
    k_df = load_data()
    print(f"  Loaded {len(k_df)} kicker rows, {k_df['player_id'].nunique()} kickers")

    # Compute targets on full data (needed for feature computation)
    k_df = compute_targets(k_df)

    # Compute ALL features on full data before splitting
    # (rolling features need complete within-season history)
    print("Computing kicker features on full dataset...")
    compute_features(k_df)

    # Per-kick records for the attention NN's inner pool.
    print("Loading per-kick records...")
    kicks_df = load_kicks(k_df)
    print(f"  Loaded {len(kicks_df)} kick records")

    # --- Cross-season split ---
    train_df, val_df, test_df = season_split(k_df)

    # Shallow-copy the caller's config (or CONFIG default) so we can inject the
    # runtime builder without mutating the source dict — the tuner reuses the
    # same base_cfg across trials and would crash on the second trial if we
    # mutated it in place.
    cfg = dict(config if config is not None else CONFIG)

    # Closure over kicks_df so the shared pipeline can build nested history
    # arrays for each split without knowing kicker specifics. The window shape
    # (attn_max_games etc.) is read from ``cfg`` inside the helper so a tuner
    # override via ``run(config=...)`` takes effect — these keys aren't plumbed
    # into the cfg dict by ``build_pipeline_config`` (K's nested attention is the
    # sole consumer), so the closure is the only place a tuner override can land.
    # Shared with ``run_cv`` via ``_build_kick_history_closure``.
    cfg["attn_history_builder_fn"] = _build_kick_history_closure(cfg, kicks_df)

    return run_pipeline("K", cfg, train_df, val_df, test_df, seed)


def _build_kick_history_closure(cfg, kicks_df):
    """Bind the per-kick nested-history builder over ``kicks_df`` (shared by
    ``run`` and ``run_cv``). Reads the attention-window shape from ``cfg`` so a
    tuner override of ``attn_max_games`` etc. takes effect."""
    return functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=cfg.get("attn_kick_stats", POSITION_CONFIG.attn_kick_stats),
        max_games=cfg.get("attn_max_games", POSITION_CONFIG.attn_max_games),
        max_kicks_per_game=cfg.get(
            "attn_max_kicks_per_game", POSITION_CONFIG.attn_max_kicks_per_game
        ),
    )


def run_cv(seed=42, config=None):
    """Expanding-window CV for K. Self-loads kicker data + the per-kick closure
    (like ``run()``), then hands the full train+val frame plus a held-out 2025
    test frame to the shared CV pipeline.

    ``full_df`` is built from the *unfiltered* ``k_df`` (not ``season_split``,
    which applies K's ``min_games`` to train) so ``_prepare_position_data``
    applies the shared ``MIN_GAMES_PER_SEASON`` filter uniformly per fold,
    matching the QB/RB/WR CV path. Module-level def (not a factory closure) so
    the runpy-monkeypatch test pattern keeps working.
    """
    k_df = load_data()
    k_df = compute_targets(k_df)
    compute_features(k_df)
    kicks_df = load_kicks(k_df)

    cfg = dict(config if config is not None else CONFIG)
    cfg["attn_history_builder_fn"] = _build_kick_history_closure(cfg, kicks_df)

    full_df = k_df[k_df["season"].isin(TRAIN_SEASONS + VAL_SEASONS)].copy()
    test_df = k_df[k_df["season"].isin(TEST_SEASONS)].copy()
    return run_cv_pipeline("K", cfg, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K pipeline")
    parser.add_argument("--cv", action="store_true", help="Use expanding-window cross-validation")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Set FF_DEVICE for this run; omitted leaves the environment unchanged.",
    )
    args = parser.parse_args()
    if args.device is not None:
        os.environ["FF_DEVICE"] = args.device
    (run_cv if args.cv else run)()
