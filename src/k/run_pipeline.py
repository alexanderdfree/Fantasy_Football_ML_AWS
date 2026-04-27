"""End-to-end K (Kicker) position model pipeline.

Uses PBP-reconstructed kicker data from 2015-2025 (post-PAT rule change).
Cross-season splits: train 2015-2023, val 2024, test 2025.
"""

import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.k.config import (
    ATTN_BATCH_SIZE,
    ATTN_D_MODEL,
    ATTN_DROPOUT,
    ATTN_ENCODER_HIDDEN_DIM,
    ATTN_KICK_DIM,
    ATTN_KICK_STATS,
    ATTN_LR,
    ATTN_MAX_GAMES,
    ATTN_MAX_KICKS_PER_GAME,
    ATTN_N_HEADS,
    ATTN_PATIENCE,
    ATTN_POSITIONAL_ENCODING,
    ATTN_PROJECT_KV,
    ATTN_STATIC_FEATURES,
    ATTN_WEIGHT_DECAY,
    CV_SPLIT_COLUMN,
    ENET_L1_RATIOS,
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
    NN_LR,
    NN_NON_NEGATIVE_TARGETS,
    NN_PATIENCE,
    NN_WEIGHT_DECAY,
    ONECYCLE_MAX_LR,
    ONECYCLE_PCT_START,
    RIDGE_ALPHA_GRIDS,
    RIDGE_CV_FOLDS,
    RIDGE_REFINE_POINTS,
    SCHEDULER_TYPE,
    SPECIFIC_FEATURES,
    TARGETS,
    TRAIN_ATTENTION_NN,
    TRAIN_ELASTICNET,
    TRAIN_LIGHTGBM,
)
from src.k.data import filter_to_position, load_data, load_kicks, season_split
from src.k.features import (
    add_specific_features,
    build_nested_kick_history,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.k.targets import compute_targets
from src.shared.pipeline import run_pipeline

# attn_history_builder_fn is injected at runtime (closes over kicks_df).
CONFIG = {
    "targets": TARGETS,
    "ridge_alpha_grids": RIDGE_ALPHA_GRIDS,
    "ridge_cv_folds": RIDGE_CV_FOLDS,
    "cv_split_column": CV_SPLIT_COLUMN,
    "ridge_refine_points": RIDGE_REFINE_POINTS,
    "specific_features": SPECIFIC_FEATURES,
    "filter_fn": filter_to_position,
    "compute_targets_fn": compute_targets,
    "add_features_fn": add_specific_features,
    "fill_nans_fn": fill_nans,
    "get_feature_columns_fn": get_feature_columns,
    "compute_adjustment_fn": None,
    "nn_backbone_layers": NN_BACKBONE_LAYERS,
    "nn_head_hidden": NN_HEAD_HIDDEN,
    "nn_dropout": NN_DROPOUT,
    "nn_head_hidden_overrides": None,
    "nn_non_negative_targets": NN_NON_NEGATIVE_TARGETS,
    "nn_lr": NN_LR,
    "nn_weight_decay": NN_WEIGHT_DECAY,
    "nn_epochs": NN_EPOCHS,
    "nn_batch_size": NN_BATCH_SIZE,
    "nn_patience": NN_PATIENCE,
    "loss_weights": LOSS_WEIGHTS,
    "huber_deltas": HUBER_DELTAS,
    "scheduler_type": SCHEDULER_TYPE,
    "onecycle_max_lr": ONECYCLE_MAX_LR,
    "onecycle_pct_start": ONECYCLE_PCT_START,
    # Attention NN (nested: inner pool per game, outer per-target attention).
    "train_attention_nn": TRAIN_ATTENTION_NN,
    "attn_history_structure": "nested",
    "attn_static_from_df": True,
    "attn_static_features": ATTN_STATIC_FEATURES,
    "attn_d_model": ATTN_D_MODEL,
    "attn_n_heads": ATTN_N_HEADS,
    "attn_kick_dim": ATTN_KICK_DIM,
    "attn_encoder_hidden_dim": ATTN_ENCODER_HIDDEN_DIM,
    "attn_project_kv": ATTN_PROJECT_KV,
    "attn_positional_encoding": ATTN_POSITIONAL_ENCODING,
    # `attn_gated_fusion` intentionally omitted — the nested attention path
    # (MultiHeadNetWithNestedHistory) does not consume it; setting it would
    # be a silent no-op and invite config drift.
    "attn_dropout": ATTN_DROPOUT,
    "attn_lr": ATTN_LR,
    "attn_weight_decay": ATTN_WEIGHT_DECAY,
    "attn_batch_size": ATTN_BATCH_SIZE,
    "attn_patience": ATTN_PATIENCE,
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
        kick_stats=ATTN_KICK_STATS,
        max_games=ATTN_MAX_GAMES,
        max_kicks_per_game=ATTN_MAX_KICKS_PER_GAME,
    )

    cfg = dict(CONFIG)
    cfg["attn_history_builder_fn"] = kick_history_builder

    return run_pipeline("K", cfg, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run()
