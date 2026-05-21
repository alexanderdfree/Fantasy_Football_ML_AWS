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

from src.k.config import POSITION_CONFIG
from src.k.data import load_data, load_kicks, season_split
from src.k.features import build_nested_kick_history, compute_features
from src.k.targets import compute_targets
from src.shared.pipeline import run_pipeline
from src.shared.position_pipeline import build_pipeline_config

# K's CONFIG omits the runtime-injected attn_history_builder_fn; run() fills
# it in after kicks_df is loaded.
CONFIG = build_pipeline_config("K", POSITION_CONFIG)


def run(seed=42):
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

    # Closure over kicks_df so the shared pipeline can build nested history
    # arrays for each split without knowing kicker specifics.
    kick_history_builder = functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=POSITION_CONFIG.attn_kick_stats,
        max_games=POSITION_CONFIG.attn_max_games,
        max_kicks_per_game=POSITION_CONFIG.attn_max_kicks_per_game,
    )

    cfg = dict(CONFIG)
    cfg["attn_history_builder_fn"] = kick_history_builder

    return run_pipeline("K", cfg, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run()
