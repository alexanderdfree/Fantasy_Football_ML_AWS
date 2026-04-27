"""End-to-end WR position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_cv_pipeline, run_pipeline
from src.WR.wr_config import (
    WR_ATTN_D_MODEL,
    WR_ATTN_DROPOUT,
    WR_ATTN_ENCODER_HIDDEN_DIM,
    WR_ATTN_GATE_HIDDEN,
    WR_ATTN_GATE_WEIGHT,
    WR_ATTN_GATED,
    WR_ATTN_HISTORY_STATS,
    WR_ATTN_MAX_SEQ_LEN,
    WR_ATTN_N_HEADS,
    WR_ATTN_POSITIONAL_ENCODING,
    WR_ATTN_STATIC_FEATURES,
    WR_COSINE_ETA_MIN,
    WR_COSINE_T0,
    WR_COSINE_T_MULT,
    WR_ENET_L1_RATIOS,
    WR_GATED_TARGETS,
    WR_HEAD_LOSSES,
    WR_HUBER_DELTAS,
    WR_LGBM_COLSAMPLE_BYTREE,
    WR_LGBM_LEARNING_RATE,
    WR_LGBM_MAX_DEPTH,
    WR_LGBM_MIN_CHILD_SAMPLES,
    WR_LGBM_MIN_SPLIT_GAIN,
    WR_LGBM_N_ESTIMATORS,
    WR_LGBM_NUM_LEAVES,
    WR_LGBM_OBJECTIVE,
    WR_LGBM_REG_ALPHA,
    WR_LGBM_REG_LAMBDA,
    WR_LGBM_SUBSAMPLE,
    WR_LOSS_WEIGHTS,
    WR_NN_BACKBONE_LAYERS,
    WR_NN_BATCH_SIZE,
    WR_NN_DROPOUT,
    WR_NN_EPOCHS,
    WR_NN_HEAD_HIDDEN,
    WR_NN_HEAD_HIDDEN_OVERRIDES,
    WR_NN_LR,
    WR_NN_PATIENCE,
    WR_NN_WEIGHT_DECAY,
    WR_OPP_ATTN_HISTORY_STATS,
    WR_OPP_ATTN_MAX_SEQ_LEN,
    WR_RIDGE_ALPHA_GRIDS,
    WR_RIDGE_PCA_COMPONENTS,
    WR_SCHEDULER_TYPE,
    WR_SPECIFIC_FEATURES,
    WR_TARGETS,
    WR_TRAIN_ATTENTION_NN,
    WR_TRAIN_ELASTICNET,
    WR_TRAIN_LIGHTGBM,
)
from src.WR.wr_data import filter_to_wr
from src.WR.wr_features import add_wr_specific_features, fill_wr_nans, get_wr_feature_columns
from src.WR.wr_targets import compute_wr_targets

WR_CONFIG = {
    "targets": WR_TARGETS,
    "ridge_alpha_grids": WR_RIDGE_ALPHA_GRIDS,
    "ridge_pca_components": WR_RIDGE_PCA_COMPONENTS,
    "specific_features": WR_SPECIFIC_FEATURES,
    "filter_fn": filter_to_wr,
    "compute_targets_fn": compute_wr_targets,
    "add_features_fn": add_wr_specific_features,
    "fill_nans_fn": fill_wr_nans,
    "get_feature_columns_fn": get_wr_feature_columns,
    "aggregate_fn": aggregate_fn_for("WR"),
    "nn_backbone_layers": WR_NN_BACKBONE_LAYERS,
    "nn_head_hidden": WR_NN_HEAD_HIDDEN,
    "nn_dropout": WR_NN_DROPOUT,
    "nn_head_hidden_overrides": WR_NN_HEAD_HIDDEN_OVERRIDES,
    "nn_lr": WR_NN_LR,
    "nn_weight_decay": WR_NN_WEIGHT_DECAY,
    "nn_epochs": WR_NN_EPOCHS,
    "nn_batch_size": WR_NN_BATCH_SIZE,
    "nn_patience": WR_NN_PATIENCE,
    "loss_weights": WR_LOSS_WEIGHTS,
    "huber_deltas": WR_HUBER_DELTAS,
    "scheduler_type": WR_SCHEDULER_TYPE,
    "cosine_t0": WR_COSINE_T0,
    "cosine_t_mult": WR_COSINE_T_MULT,
    "cosine_eta_min": WR_COSINE_ETA_MIN,
    "train_attention_nn": WR_TRAIN_ATTENTION_NN,
    "attn_d_model": WR_ATTN_D_MODEL,
    "attn_n_heads": WR_ATTN_N_HEADS,
    "attn_max_seq_len": WR_ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": WR_ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": WR_ATTN_POSITIONAL_ENCODING,
    "attn_dropout": WR_ATTN_DROPOUT,
    "attn_history_stats": WR_ATTN_HISTORY_STATS,
    "attn_static_features": WR_ATTN_STATIC_FEATURES,
    "opp_attn_history_stats": WR_OPP_ATTN_HISTORY_STATS,
    "opp_attn_max_seq_len": WR_OPP_ATTN_MAX_SEQ_LEN,
    "attn_gated": WR_ATTN_GATED,
    "attn_gate_hidden": WR_ATTN_GATE_HIDDEN,
    "attn_gate_weight": WR_ATTN_GATE_WEIGHT,
    "gated_targets": WR_GATED_TARGETS,
    "head_losses": WR_HEAD_LOSSES,
    "train_elasticnet": WR_TRAIN_ELASTICNET,
    "enet_l1_ratios": WR_ENET_L1_RATIOS,
    "train_lightgbm": WR_TRAIN_LIGHTGBM,
    "lgbm_n_estimators": WR_LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": WR_LGBM_LEARNING_RATE,
    "lgbm_num_leaves": WR_LGBM_NUM_LEAVES,
    "lgbm_max_depth": WR_LGBM_MAX_DEPTH,
    "lgbm_subsample": WR_LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": WR_LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": WR_LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": WR_LGBM_REG_ALPHA,
    "lgbm_min_child_samples": WR_LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": WR_LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": WR_LGBM_OBJECTIVE,
}


def run_wr_pipeline(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("WR", config or WR_CONFIG, train_df, val_df, test_df, seed)


def run_wr_cv_pipeline(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("WR", config or WR_CONFIG, full_df, test_df, seed)


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

        config = build_tiny_config("WR")
    else:
        config = WR_CONFIG
    if args.cv:
        run_wr_cv_pipeline(config=config)
    else:
        run_wr_pipeline(config=config)
