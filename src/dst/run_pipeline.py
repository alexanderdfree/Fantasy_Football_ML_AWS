"""End-to-end DST (Defense/Special Teams) model pipeline.

D/ST operates at the team level (not player level). Data is constructed
from schedule scores, opponent offensive stats, and individual defensive
player stats. Uses standard temporal splits (2018-2023 / 2024 / 2025).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.dst.config import (
    ATTN_BATCH_SIZE,
    ATTN_D_MODEL,
    ATTN_DROPOUT,
    ATTN_ENCODER_HIDDEN_DIM,
    ATTN_GATED,
    ATTN_GATED_FUSION,
    ATTN_HISTORY_STATS,
    ATTN_LR,
    ATTN_MAX_SEQ_LEN,
    ATTN_N_HEADS,
    ATTN_PATIENCE,
    ATTN_POSITIONAL_ENCODING,
    ATTN_PROJECT_KV,
    ATTN_STATIC_FEATURES,
    ATTN_WEIGHT_DECAY,
    COSINE_ETA_MIN,
    COSINE_T0,
    COSINE_T_MULT,
    ENET_L1_RATIOS,
    HEAD_LOSSES,
    HUBER_DELTAS,
    LGBM_COLSAMPLE_BYTREE,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_MIN_SPLIT_GAIN,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_OBJECTIVE,
    LGBM_REG_ALPHA,
    LGBM_REG_LAMBDA,
    LGBM_SUBSAMPLE,
    LOSS_WEIGHTS,
    NN_BACKBONE_LAYERS,
    NN_BATCH_SIZE,
    NN_DROPOUT,
    NN_EPOCHS,
    NN_HEAD_HIDDEN,
    NN_HEAD_HIDDEN_OVERRIDES,
    NN_LR,
    NN_NON_NEGATIVE_TARGETS,
    NN_PATIENCE,
    NN_WEIGHT_DECAY,
    POISSON_TARGETS,
    RIDGE_ALPHA_GRIDS,
    RIDGE_PCA_COMPONENTS,
    SCHEDULER_TYPE,
    SPECIFIC_FEATURES,
    TARGETS,
    TRAIN_ATTENTION_NN,
    TRAIN_ELASTICNET,
    TRAIN_LIGHTGBM,
)
from src.dst.data import build_data, filter_to_position
from src.dst.features import (
    add_specific_features,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.dst.targets import compute_targets
from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_pipeline

CONFIG = {
    "targets": TARGETS,
    "ridge_alpha_grids": RIDGE_ALPHA_GRIDS,
    "ridge_pca_components": RIDGE_PCA_COMPONENTS,
    "specific_features": SPECIFIC_FEATURES,
    "filter_fn": filter_to_position,
    "compute_targets_fn": compute_targets,
    "add_features_fn": add_specific_features,
    "fill_nans_fn": fill_nans,
    "get_feature_columns_fn": get_feature_columns,
    "compute_adjustment_fn": None,
    # Serving/reporting aggregator; training is on raw-stat heads only.
    "aggregate_fn": aggregate_fn_for("DST"),
    "nn_backbone_layers": NN_BACKBONE_LAYERS,
    "nn_head_hidden": NN_HEAD_HIDDEN,
    "nn_dropout": NN_DROPOUT,
    "nn_head_hidden_overrides": NN_HEAD_HIDDEN_OVERRIDES,
    "nn_non_negative_targets": NN_NON_NEGATIVE_TARGETS,
    "nn_lr": NN_LR,
    "nn_weight_decay": NN_WEIGHT_DECAY,
    "nn_epochs": NN_EPOCHS,
    "nn_batch_size": NN_BATCH_SIZE,
    "nn_patience": NN_PATIENCE,
    "loss_weights": LOSS_WEIGHTS,
    "huber_deltas": HUBER_DELTAS,
    "poisson_targets": POISSON_TARGETS,
    "scheduler_type": SCHEDULER_TYPE,
    "cosine_t0": COSINE_T0,
    "cosine_t_mult": COSINE_T_MULT,
    "cosine_eta_min": COSINE_ETA_MIN,
    # Attention NN
    "train_attention_nn": TRAIN_ATTENTION_NN,
    "attn_d_model": ATTN_D_MODEL,
    "attn_n_heads": ATTN_N_HEADS,
    "attn_max_seq_len": ATTN_MAX_SEQ_LEN,
    "attn_history_stats": ATTN_HISTORY_STATS,
    "attn_static_features": ATTN_STATIC_FEATURES,
    "attn_encoder_hidden_dim": ATTN_ENCODER_HIDDEN_DIM,
    "attn_project_kv": ATTN_PROJECT_KV,
    "attn_positional_encoding": ATTN_POSITIONAL_ENCODING,
    "attn_gated_fusion": ATTN_GATED_FUSION,
    "attn_dropout": ATTN_DROPOUT,
    "attn_lr": ATTN_LR,
    "attn_weight_decay": ATTN_WEIGHT_DECAY,
    "attn_batch_size": ATTN_BATCH_SIZE,
    "attn_patience": ATTN_PATIENCE,
    "attn_gated": ATTN_GATED,
    "head_losses": HEAD_LOSSES,
    # LightGBM
    "train_elasticnet": TRAIN_ELASTICNET,
    "enet_l1_ratios": ENET_L1_RATIOS,
    "train_lightgbm": TRAIN_LIGHTGBM,
    "lgbm_n_estimators": LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": LGBM_LEARNING_RATE,
    "lgbm_num_leaves": LGBM_NUM_LEAVES,
    "lgbm_max_depth": LGBM_MAX_DEPTH,
    "lgbm_subsample": LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": LGBM_REG_ALPHA,
    "lgbm_min_child_samples": LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": LGBM_OBJECTIVE,
}


def run(seed=42):
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

    return run_pipeline("DST", CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run()
