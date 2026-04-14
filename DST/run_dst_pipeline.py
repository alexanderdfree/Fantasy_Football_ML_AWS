"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2018-2023 / 2024 / 2025).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DST.dst_config import (
    DST_TARGETS, DST_RIDGE_ALPHAS, DST_SPECIFIC_FEATURES,
    DST_NN_BACKBONE_LAYERS, DST_NN_HEAD_HIDDEN, DST_NN_HEAD_HIDDEN_OVERRIDES,
    DST_NN_DROPOUT,
    DST_NN_LR, DST_NN_WEIGHT_DECAY, DST_NN_EPOCHS, DST_NN_BATCH_SIZE,
    DST_NN_PATIENCE,
    DST_LOSS_WEIGHTS, DST_LOSS_W_TOTAL, DST_HUBER_DELTAS,
    DST_SCHEDULER_TYPE, DST_COSINE_T0, DST_COSINE_T_MULT, DST_COSINE_ETA_MIN,
)
from DST.dst_data import build_dst_data, filter_to_dst
from DST.dst_targets import compute_dst_targets, compute_dst_adjustment
from DST.dst_features import (
    compute_dst_features, add_dst_specific_features,
    get_dst_feature_columns, fill_dst_nans,
)
from shared.pipeline import run_pipeline
from src.config import TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS


def run_dst_pipeline(seed=42):
    # --- Build team-level D/ST data ---
    print("Building D/ST team-level data...")
    dst_df = build_dst_data()
    print(f"  Built {len(dst_df)} team-week rows, {dst_df['team'].nunique()} teams")
    print(f"  Seasons: {sorted(dst_df['season'].unique())}")

    # Compute targets on full data (needed for feature computation)
    dst_df = compute_dst_targets(dst_df)

    # Compute ALL features on full data before splitting
    print("Computing D/ST features on full dataset...")
    compute_dst_features(dst_df)

    # --- Standard temporal split ---
    train_df = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val_df = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test_df = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    print(f"  Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # --- Run shared pipeline ---
    DST_CONFIG = {
        "targets": DST_TARGETS,
        "ridge_alphas": DST_RIDGE_ALPHAS,
        "specific_features": DST_SPECIFIC_FEATURES,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": compute_dst_adjustment,
        "nn_backbone_layers": DST_NN_BACKBONE_LAYERS,
        "nn_head_hidden": DST_NN_HEAD_HIDDEN,
        "nn_dropout": DST_NN_DROPOUT,
        "nn_head_hidden_overrides": DST_NN_HEAD_HIDDEN_OVERRIDES,
        "nn_lr": DST_NN_LR,
        "nn_weight_decay": DST_NN_WEIGHT_DECAY,
        "nn_epochs": DST_NN_EPOCHS,
        "nn_batch_size": DST_NN_BATCH_SIZE,
        "nn_patience": DST_NN_PATIENCE,
        "loss_weights": DST_LOSS_WEIGHTS,
        "loss_w_total": DST_LOSS_W_TOTAL,
        "huber_deltas": DST_HUBER_DELTAS,
        "scheduler_type": DST_SCHEDULER_TYPE,
        "cosine_t0": DST_COSINE_T0,
        "cosine_t_mult": DST_COSINE_T_MULT,
        "cosine_eta_min": DST_COSINE_ETA_MIN,
    }

    return run_pipeline("DST", DST_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_dst_pipeline()
