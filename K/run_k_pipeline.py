"""End-to-end K (Kicker) position model pipeline.

Unlike QB/RB/WR/TE, kicker data is only available for 2025 in nflverse.
This pipeline uses within-season temporal splits and builds all features
from kicker-specific stats (FG/PAT breakdowns, Vegas lines).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from K.k_config import (
    K_TARGETS, K_RIDGE_ALPHAS, K_SPECIFIC_FEATURES,
    K_NN_BACKBONE_LAYERS, K_NN_HEAD_HIDDEN, K_NN_DROPOUT,
    K_NN_LR, K_NN_WEIGHT_DECAY, K_NN_EPOCHS, K_NN_BATCH_SIZE,
    K_NN_PATIENCE,
    K_LOSS_WEIGHTS, K_LOSS_W_TOTAL, K_HUBER_DELTAS,
    K_SCHEDULER_TYPE, K_ONECYCLE_MAX_LR, K_ONECYCLE_PCT_START,
    K_VAL_WEEKS, K_TEST_WEEKS,
)
from K.k_data import load_kicker_data, filter_to_k, kicker_week_split
from K.k_targets import compute_k_targets, compute_k_miss_adjustment
from K.k_features import (
    compute_k_features, add_k_specific_features,
    get_k_feature_columns, fill_k_nans,
)
from shared.pipeline import run_pipeline


def run_k_pipeline(seed=42):
    # --- Load and prepare kicker data ---
    print("Loading kicker data...")
    k_df = load_kicker_data()
    print(f"  Loaded {len(k_df)} kicker rows, {k_df['player_id'].nunique()} kickers")

    # Compute targets on full data (needed for feature computation)
    k_df = compute_k_targets(k_df)

    # Compute ALL features on full data before splitting
    # (rolling features need complete within-season history)
    print("Computing kicker features on full dataset...")
    compute_k_features(k_df)

    # --- Within-season split ---
    train_df, val_df, test_df = kicker_week_split(
        k_df, val_weeks=K_VAL_WEEKS, test_weeks=K_TEST_WEEKS
    )

    # --- Run shared pipeline ---
    K_CONFIG = {
        "targets": K_TARGETS,
        "ridge_alphas": K_RIDGE_ALPHAS,
        "specific_features": K_SPECIFIC_FEATURES,
        "filter_fn": filter_to_k,
        "compute_targets_fn": compute_k_targets,
        "add_features_fn": add_k_specific_features,
        "fill_nans_fn": fill_k_nans,
        "get_feature_columns_fn": get_k_feature_columns,
        "compute_adjustment_fn": compute_k_miss_adjustment,
        "nn_backbone_layers": K_NN_BACKBONE_LAYERS,
        "nn_head_hidden": K_NN_HEAD_HIDDEN,
        "nn_dropout": K_NN_DROPOUT,
        "nn_head_hidden_overrides": None,
        "nn_lr": K_NN_LR,
        "nn_weight_decay": K_NN_WEIGHT_DECAY,
        "nn_epochs": K_NN_EPOCHS,
        "nn_batch_size": K_NN_BATCH_SIZE,
        "nn_patience": K_NN_PATIENCE,
        "loss_weights": K_LOSS_WEIGHTS,
        "loss_w_total": K_LOSS_W_TOTAL,
        "huber_deltas": K_HUBER_DELTAS,
        "scheduler_type": K_SCHEDULER_TYPE,
        "onecycle_max_lr": K_ONECYCLE_MAX_LR,
        "onecycle_pct_start": K_ONECYCLE_PCT_START,
    }

    return run_pipeline("K", K_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_k_pipeline()
