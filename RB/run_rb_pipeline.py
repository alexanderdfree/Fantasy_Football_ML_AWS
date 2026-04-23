"""End-to-end RB position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from RB.rb_config import (
    RB_ATTN_BATCH_SIZE,
    RB_ATTN_D_MODEL,
    RB_ATTN_DROPOUT,
    RB_ATTN_ENCODER_HIDDEN_DIM,
    RB_ATTN_GATE_HIDDEN,
    RB_ATTN_GATE_WEIGHT,
    RB_ATTN_GATED,
    RB_ATTN_GATED_FUSION,
    RB_ATTN_HISTORY_STATS,
    RB_ATTN_LR,
    RB_ATTN_MAX_SEQ_LEN,
    RB_ATTN_N_HEADS,
    RB_ATTN_PATIENCE,
    RB_ATTN_POSITIONAL_ENCODING,
    RB_ATTN_PROJECT_KV,
    RB_ATTN_STATIC_FEATURES,
    RB_ATTN_WEIGHT_DECAY,
    RB_COSINE_ETA_MIN,
    RB_COSINE_T0,
    RB_COSINE_T_MULT,
    RB_ENET_L1_RATIOS,
    RB_GATED_ORDINAL_TARGETS,
    RB_GATED_TARGETS,
    RB_HEAD_LOSSES,
    RB_HUBER_DELTAS,
    RB_LGBM_COLSAMPLE_BYTREE,
    RB_LGBM_LEARNING_RATE,
    RB_LGBM_MAX_DEPTH,
    RB_LGBM_MIN_CHILD_SAMPLES,
    RB_LGBM_MIN_SPLIT_GAIN,
    RB_LGBM_N_ESTIMATORS,
    RB_LGBM_NUM_LEAVES,
    RB_LGBM_OBJECTIVE,
    RB_LGBM_REG_ALPHA,
    RB_LGBM_REG_LAMBDA,
    RB_LGBM_SUBSAMPLE,
    RB_LOSS_WEIGHTS,
    RB_NN_BACKBONE_LAYERS,
    RB_NN_BATCH_SIZE,
    RB_NN_DROPOUT,
    RB_NN_EPOCHS,
    RB_NN_HEAD_HIDDEN,
    RB_NN_HEAD_HIDDEN_OVERRIDES,
    RB_NN_LR,
    RB_NN_PATIENCE,
    RB_NN_WEIGHT_DECAY,
    RB_ORDINAL_TARGETS,
    RB_RIDGE_ALPHA_GRIDS,
    RB_RIDGE_PCA_COMPONENTS,
    RB_SCHEDULER_TYPE,
    RB_SPECIFIC_FEATURES,
    RB_TARGETS,
    RB_TD_MODEL_TYPE,
    RB_TRAIN_ATTENTION_NN,
    RB_TRAIN_ELASTICNET,
    RB_TRAIN_LIGHTGBM,
    RB_TWO_STAGE_TARGETS,
)
from RB.rb_data import filter_to_rb
from RB.rb_features import add_rb_specific_features, fill_rb_nans, get_rb_feature_columns
from RB.rb_targets import compute_rb_targets
from shared.pipeline import run_cv_pipeline, run_pipeline

RB_CONFIG = {
    "targets": RB_TARGETS,
    "ridge_alpha_grids": RB_RIDGE_ALPHA_GRIDS,
    "two_stage_targets": RB_TWO_STAGE_TARGETS if RB_TD_MODEL_TYPE == "two_stage" else {},
    "classification_targets": (
        RB_ORDINAL_TARGETS
        if RB_TD_MODEL_TYPE == "ordinal"
        else RB_GATED_ORDINAL_TARGETS
        if RB_TD_MODEL_TYPE == "gated_ordinal"
        else {}
    ),
    "ridge_pca_components": RB_RIDGE_PCA_COMPONENTS,
    "specific_features": RB_SPECIFIC_FEATURES,
    "filter_fn": filter_to_rb,
    "compute_targets_fn": compute_rb_targets,
    "add_features_fn": add_rb_specific_features,
    "fill_nans_fn": fill_rb_nans,
    "get_feature_columns_fn": get_rb_feature_columns,
    "nn_backbone_layers": RB_NN_BACKBONE_LAYERS,
    "nn_head_hidden": RB_NN_HEAD_HIDDEN,
    "nn_dropout": RB_NN_DROPOUT,
    "nn_head_hidden_overrides": RB_NN_HEAD_HIDDEN_OVERRIDES,
    "nn_lr": RB_NN_LR,
    "nn_weight_decay": RB_NN_WEIGHT_DECAY,
    "nn_epochs": RB_NN_EPOCHS,
    "nn_batch_size": RB_NN_BATCH_SIZE,
    "nn_patience": RB_NN_PATIENCE,
    "loss_weights": RB_LOSS_WEIGHTS,
    "huber_deltas": RB_HUBER_DELTAS,
    "scheduler_type": RB_SCHEDULER_TYPE,
    "cosine_t0": RB_COSINE_T0,
    "cosine_t_mult": RB_COSINE_T_MULT,
    "cosine_eta_min": RB_COSINE_ETA_MIN,
    "train_attention_nn": RB_TRAIN_ATTENTION_NN,
    "attn_d_model": RB_ATTN_D_MODEL,
    "attn_n_heads": RB_ATTN_N_HEADS,
    "attn_max_seq_len": RB_ATTN_MAX_SEQ_LEN,
    "attn_history_stats": RB_ATTN_HISTORY_STATS,
    "attn_static_features": RB_ATTN_STATIC_FEATURES,
    "attn_encoder_hidden_dim": RB_ATTN_ENCODER_HIDDEN_DIM,
    "attn_project_kv": RB_ATTN_PROJECT_KV,
    "attn_positional_encoding": RB_ATTN_POSITIONAL_ENCODING,
    "attn_gated_fusion": RB_ATTN_GATED_FUSION,
    "attn_dropout": RB_ATTN_DROPOUT,
    "attn_lr": RB_ATTN_LR,
    "attn_weight_decay": RB_ATTN_WEIGHT_DECAY,
    "attn_batch_size": RB_ATTN_BATCH_SIZE,
    "attn_patience": RB_ATTN_PATIENCE,
    "attn_gated": RB_ATTN_GATED,
    "attn_gate_hidden": RB_ATTN_GATE_HIDDEN,
    "attn_gate_weight": RB_ATTN_GATE_WEIGHT,
    "gated_targets": RB_GATED_TARGETS,
    "head_losses": RB_HEAD_LOSSES,
    "train_elasticnet": RB_TRAIN_ELASTICNET,
    "enet_l1_ratios": RB_ENET_L1_RATIOS,
    "train_lightgbm": RB_TRAIN_LIGHTGBM,
    "lgbm_n_estimators": RB_LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": RB_LGBM_LEARNING_RATE,
    "lgbm_num_leaves": RB_LGBM_NUM_LEAVES,
    "lgbm_max_depth": RB_LGBM_MAX_DEPTH,
    "lgbm_subsample": RB_LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": RB_LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": RB_LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": RB_LGBM_REG_ALPHA,
    "lgbm_min_child_samples": RB_LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": RB_LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": RB_LGBM_OBJECTIVE,
}


def run_rb_pipeline(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("RB", config or RB_CONFIG, train_df, val_df, test_df, seed)


def run_rb_cv_pipeline(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("RB", config or RB_CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Use expanding-window CV")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use shrunk smoke-test config (from tests/_pipeline_e2e_utils)",
    )
    args = parser.parse_args()
    if args.tiny:
        from tests._pipeline_e2e_utils import build_tiny_config

        config = build_tiny_config("RB")
    else:
        config = RB_CONFIG
    if args.cv:
        run_rb_cv_pipeline(config=config)
    else:
        run_rb_pipeline(config=config)
