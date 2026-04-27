"""End-to-end K (Kicker) position model pipeline.

Uses PBP-reconstructed kicker data from 2015-2025 (post-PAT rule change).
Cross-season splits: train 2015-2023, val 2024, test 2025.
"""

import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.k.config import (
    K_ATTN_BATCH_SIZE,
    K_ATTN_D_MODEL,
    K_ATTN_DROPOUT,
    K_ATTN_ENCODER_HIDDEN_DIM,
    K_ATTN_KICK_DIM,
    K_ATTN_KICK_STATS,
    K_ATTN_LR,
    K_ATTN_MAX_GAMES,
    K_ATTN_MAX_KICKS_PER_GAME,
    K_ATTN_N_HEADS,
    K_ATTN_PATIENCE,
    K_ATTN_POSITIONAL_ENCODING,
    K_ATTN_PROJECT_KV,
    K_ATTN_STATIC_FEATURES,
    K_ATTN_WEIGHT_DECAY,
    K_CV_SPLIT_COLUMN,
    K_ENET_L1_RATIOS,
    K_HUBER_DELTAS,
    K_LGBM_COLSAMPLE_BYTREE,
    K_LGBM_LEARNING_RATE,
    K_LGBM_MAX_DEPTH,
    K_LGBM_MIN_CHILD_SAMPLES,
    K_LGBM_MIN_SPLIT_GAIN,
    K_LGBM_N_ESTIMATORS,
    K_LGBM_NUM_LEAVES,
    K_LGBM_OBJECTIVE,
    K_LGBM_REG_ALPHA,
    K_LGBM_REG_LAMBDA,
    K_LGBM_SUBSAMPLE,
    K_LOSS_WEIGHTS,
    K_NN_BACKBONE_LAYERS,
    K_NN_BATCH_SIZE,
    K_NN_DROPOUT,
    K_NN_EPOCHS,
    K_NN_HEAD_HIDDEN,
    K_NN_LR,
    K_NN_NON_NEGATIVE_TARGETS,
    K_NN_PATIENCE,
    K_NN_WEIGHT_DECAY,
    K_ONECYCLE_MAX_LR,
    K_ONECYCLE_PCT_START,
    K_RIDGE_ALPHA_GRIDS,
    K_RIDGE_CV_FOLDS,
    K_RIDGE_REFINE_POINTS,
    K_SCHEDULER_TYPE,
    K_SPECIFIC_FEATURES,
    K_TARGETS,
    K_TRAIN_ATTENTION_NN,
    K_TRAIN_ELASTICNET,
    K_TRAIN_LIGHTGBM,
)
from src.k.data import filter_to_k, kicker_season_split, load_kicker_data, load_kicker_kicks
from src.k.features import (
    add_k_specific_features,
    build_nested_kick_history,
    compute_k_features,
    fill_k_nans,
    get_k_feature_columns,
)
from src.k.targets import compute_k_targets
from src.shared.pipeline import run_pipeline

# attn_history_builder_fn is injected at runtime (closes over kicks_df).
K_CONFIG = {
    "targets": K_TARGETS,
    "ridge_alpha_grids": K_RIDGE_ALPHA_GRIDS,
    "ridge_cv_folds": K_RIDGE_CV_FOLDS,
    "cv_split_column": K_CV_SPLIT_COLUMN,
    "ridge_refine_points": K_RIDGE_REFINE_POINTS,
    "specific_features": K_SPECIFIC_FEATURES,
    "filter_fn": filter_to_k,
    "compute_targets_fn": compute_k_targets,
    "add_features_fn": add_k_specific_features,
    "fill_nans_fn": fill_k_nans,
    "get_feature_columns_fn": get_k_feature_columns,
    "compute_adjustment_fn": None,
    "nn_backbone_layers": K_NN_BACKBONE_LAYERS,
    "nn_head_hidden": K_NN_HEAD_HIDDEN,
    "nn_dropout": K_NN_DROPOUT,
    "nn_head_hidden_overrides": None,
    "nn_non_negative_targets": K_NN_NON_NEGATIVE_TARGETS,
    "nn_lr": K_NN_LR,
    "nn_weight_decay": K_NN_WEIGHT_DECAY,
    "nn_epochs": K_NN_EPOCHS,
    "nn_batch_size": K_NN_BATCH_SIZE,
    "nn_patience": K_NN_PATIENCE,
    "loss_weights": K_LOSS_WEIGHTS,
    "huber_deltas": K_HUBER_DELTAS,
    "scheduler_type": K_SCHEDULER_TYPE,
    "onecycle_max_lr": K_ONECYCLE_MAX_LR,
    "onecycle_pct_start": K_ONECYCLE_PCT_START,
    # Attention NN (nested: inner pool per game, outer per-target attention).
    "train_attention_nn": K_TRAIN_ATTENTION_NN,
    "attn_history_structure": "nested",
    "attn_static_from_df": True,
    "attn_static_features": K_ATTN_STATIC_FEATURES,
    "attn_d_model": K_ATTN_D_MODEL,
    "attn_n_heads": K_ATTN_N_HEADS,
    "attn_kick_dim": K_ATTN_KICK_DIM,
    "attn_encoder_hidden_dim": K_ATTN_ENCODER_HIDDEN_DIM,
    "attn_project_kv": K_ATTN_PROJECT_KV,
    "attn_positional_encoding": K_ATTN_POSITIONAL_ENCODING,
    # `attn_gated_fusion` intentionally omitted — the nested attention path
    # (MultiHeadNetWithNestedHistory) does not consume it; setting it would
    # be a silent no-op and invite config drift.
    "attn_dropout": K_ATTN_DROPOUT,
    "attn_lr": K_ATTN_LR,
    "attn_weight_decay": K_ATTN_WEIGHT_DECAY,
    "attn_batch_size": K_ATTN_BATCH_SIZE,
    "attn_patience": K_ATTN_PATIENCE,
    "train_elasticnet": K_TRAIN_ELASTICNET,
    "enet_l1_ratios": K_ENET_L1_RATIOS,
    "train_lightgbm": K_TRAIN_LIGHTGBM,
    "lgbm_n_estimators": K_LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": K_LGBM_LEARNING_RATE,
    "lgbm_num_leaves": K_LGBM_NUM_LEAVES,
    "lgbm_max_depth": K_LGBM_MAX_DEPTH,
    "lgbm_subsample": K_LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": K_LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": K_LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": K_LGBM_REG_ALPHA,
    "lgbm_min_child_samples": K_LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": K_LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": K_LGBM_OBJECTIVE,
}


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

    # Per-kick records for the attention NN's inner pool.
    print("Loading per-kick records...")
    kicks_df = load_kicker_kicks(k_df)
    print(f"  Loaded {len(kicks_df)} kick records")

    # --- Cross-season split ---
    train_df, val_df, test_df = kicker_season_split(k_df)

    # Closure over kicks_df so the shared pipeline can build nested history
    # arrays for each split without knowing kicker specifics.
    kick_history_builder = functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=K_ATTN_KICK_STATS,
        max_games=K_ATTN_MAX_GAMES,
        max_kicks_per_game=K_ATTN_MAX_KICKS_PER_GAME,
    )

    cfg = dict(K_CONFIG)
    cfg["attn_history_builder_fn"] = kick_history_builder

    return run_pipeline("K", cfg, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_k_pipeline()
