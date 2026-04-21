"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2018-2023 / 2024 / 2025).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DST.dst_config import (
    DST_ATTN_BATCH_SIZE,
    DST_ATTN_D_MODEL,
    DST_ATTN_DROPOUT,
    DST_ATTN_ENCODER_HIDDEN_DIM,
    DST_ATTN_GATED_FUSION,
    DST_ATTN_GATED_TD,
    DST_ATTN_HISTORY_STATS,
    DST_ATTN_LR,
    DST_ATTN_MAX_SEQ_LEN,
    DST_ATTN_N_HEADS,
    DST_ATTN_PATIENCE,
    DST_ATTN_POSITIONAL_ENCODING,
    DST_ATTN_PROJECT_KV,
    DST_ATTN_WEIGHT_DECAY,
    DST_COSINE_ETA_MIN,
    DST_COSINE_T0,
    DST_COSINE_T_MULT,
    DST_HUBER_DELTAS,
    DST_LGBM_COLSAMPLE_BYTREE,
    DST_LGBM_LEARNING_RATE,
    DST_LGBM_MIN_CHILD_SAMPLES,
    DST_LGBM_N_ESTIMATORS,
    DST_LGBM_NUM_LEAVES,
    DST_LGBM_REG_ALPHA,
    DST_LGBM_REG_LAMBDA,
    DST_LGBM_SUBSAMPLE,
    DST_LOSS_W_TOTAL,
    DST_LOSS_WEIGHTS,
    DST_NN_BACKBONE_LAYERS,
    DST_NN_BATCH_SIZE,
    DST_NN_DROPOUT,
    DST_NN_EPOCHS,
    DST_NN_HEAD_HIDDEN,
    DST_NN_HEAD_HIDDEN_OVERRIDES,
    DST_NN_LR,
    DST_NN_NON_NEGATIVE_TARGETS,
    DST_NN_PATIENCE,
    DST_NN_WEIGHT_DECAY,
    DST_RIDGE_ALPHA_GRIDS,
    DST_RIDGE_PCA_COMPONENTS,
    DST_SCHEDULER_TYPE,
    DST_SPECIFIC_FEATURES,
    DST_TARGETS,
    DST_TRAIN_ATTENTION_NN,
    DST_TRAIN_LIGHTGBM,
)
from DST.dst_data import build_dst_data, filter_to_dst
from DST.dst_features import (
    add_dst_specific_features,
    compute_dst_features,
    fill_dst_nans,
    get_dst_feature_columns,
)
from DST.dst_targets import compute_dst_targets
from shared.aggregate_targets import aggregate_fn_for
from shared.pipeline import run_pipeline
from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS


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
        "ridge_alpha_grids": DST_RIDGE_ALPHA_GRIDS,
        "ridge_pca_components": DST_RIDGE_PCA_COMPONENTS,
        "specific_features": DST_SPECIFIC_FEATURES,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": None,
        # DST is in _FANTASY_POINTS_AUX_POSITIONS — wiring aggregate_fn here
        # lets the NN supervise ``total`` on ``fantasy_points`` directly
        # instead of the raw-sum (~380/game, dominated by yards_allowed).
        "aggregate_fn": aggregate_fn_for("DST"),
        "nn_backbone_layers": DST_NN_BACKBONE_LAYERS,
        "nn_head_hidden": DST_NN_HEAD_HIDDEN,
        "nn_dropout": DST_NN_DROPOUT,
        "nn_head_hidden_overrides": DST_NN_HEAD_HIDDEN_OVERRIDES,
        "nn_non_negative_targets": DST_NN_NON_NEGATIVE_TARGETS,
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
        # Attention NN
        "train_attention_nn": DST_TRAIN_ATTENTION_NN,
        "attn_d_model": DST_ATTN_D_MODEL,
        "attn_n_heads": DST_ATTN_N_HEADS,
        "attn_max_seq_len": DST_ATTN_MAX_SEQ_LEN,
        "attn_history_stats": DST_ATTN_HISTORY_STATS,
        "attn_encoder_hidden_dim": DST_ATTN_ENCODER_HIDDEN_DIM,
        "attn_project_kv": DST_ATTN_PROJECT_KV,
        "attn_positional_encoding": DST_ATTN_POSITIONAL_ENCODING,
        "attn_gated_fusion": DST_ATTN_GATED_FUSION,
        "attn_dropout": DST_ATTN_DROPOUT,
        "attn_lr": DST_ATTN_LR,
        "attn_weight_decay": DST_ATTN_WEIGHT_DECAY,
        "attn_batch_size": DST_ATTN_BATCH_SIZE,
        "attn_patience": DST_ATTN_PATIENCE,
        "attn_gated_td": DST_ATTN_GATED_TD,
        # LightGBM
        "train_lightgbm": DST_TRAIN_LIGHTGBM,
        "lgbm_n_estimators": DST_LGBM_N_ESTIMATORS,
        "lgbm_learning_rate": DST_LGBM_LEARNING_RATE,
        "lgbm_num_leaves": DST_LGBM_NUM_LEAVES,
        "lgbm_subsample": DST_LGBM_SUBSAMPLE,
        "lgbm_colsample_bytree": DST_LGBM_COLSAMPLE_BYTREE,
        "lgbm_reg_lambda": DST_LGBM_REG_LAMBDA,
        "lgbm_reg_alpha": DST_LGBM_REG_ALPHA,
        "lgbm_min_child_samples": DST_LGBM_MIN_CHILD_SAMPLES,
    }

    return run_pipeline("DST", DST_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_dst_pipeline()
