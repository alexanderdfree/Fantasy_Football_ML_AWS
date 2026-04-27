"""End-to-end QB position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.qb.config import (
    QB_ATTN_BATCH_SIZE,
    QB_ATTN_D_MODEL,
    QB_ATTN_DROPOUT,
    QB_ATTN_ENCODER_HIDDEN_DIM,
    QB_ATTN_GATE_HIDDEN,
    QB_ATTN_GATE_WEIGHT,
    QB_ATTN_GATED,
    QB_ATTN_HISTORY_STATS,
    QB_ATTN_LR,
    QB_ATTN_MAX_SEQ_LEN,
    QB_ATTN_N_HEADS,
    QB_ATTN_PATIENCE,
    QB_ATTN_POSITIONAL_ENCODING,
    QB_ATTN_STATIC_FEATURES,
    QB_ATTN_WEIGHT_DECAY,
    QB_COSINE_ETA_MIN,
    QB_COSINE_T0,
    QB_COSINE_T_MULT,
    QB_ENET_L1_RATIOS,
    QB_HEAD_LOSSES,
    QB_HUBER_DELTAS,
    QB_LGBM_COLSAMPLE_BYTREE,
    QB_LGBM_LEARNING_RATE,
    QB_LGBM_MIN_CHILD_SAMPLES,
    QB_LGBM_MIN_SPLIT_GAIN,
    QB_LGBM_N_ESTIMATORS,
    QB_LGBM_NUM_LEAVES,
    QB_LGBM_OBJECTIVE,
    QB_LGBM_REG_ALPHA,
    QB_LGBM_REG_LAMBDA,
    QB_LGBM_SUBSAMPLE,
    QB_LOSS_WEIGHTS,
    QB_NN_BACKBONE_LAYERS,
    QB_NN_BATCH_SIZE,
    QB_NN_DROPOUT,
    QB_NN_EPOCHS,
    QB_NN_HEAD_HIDDEN,
    QB_NN_HEAD_HIDDEN_OVERRIDES,
    QB_NN_LR,
    QB_NN_PATIENCE,
    QB_NN_WEIGHT_DECAY,
    QB_OPP_ATTN_HISTORY_STATS,
    QB_OPP_ATTN_MAX_SEQ_LEN,
    QB_RIDGE_ALPHA_GRIDS,
    QB_SCHEDULER_TYPE,
    QB_SPECIFIC_FEATURES,
    QB_TARGETS,
    QB_TRAIN_ATTENTION_NN,
    QB_TRAIN_ELASTICNET,
    QB_TRAIN_LIGHTGBM,
)
from src.qb.data import filter_to_qb
from src.qb.features import add_qb_specific_features, fill_qb_nans, get_qb_feature_columns
from src.qb.targets import compute_qb_targets
from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_cv_pipeline, run_pipeline

QB_CONFIG = {
    "targets": QB_TARGETS,
    "ridge_alpha_grids": QB_RIDGE_ALPHA_GRIDS,
    "specific_features": QB_SPECIFIC_FEATURES,
    "filter_fn": filter_to_qb,
    "compute_targets_fn": compute_qb_targets,
    "add_features_fn": add_qb_specific_features,
    "fill_nans_fn": fill_qb_nans,
    "get_feature_columns_fn": get_qb_feature_columns,
    "aggregate_fn": aggregate_fn_for("QB"),
    "nn_backbone_layers": QB_NN_BACKBONE_LAYERS,
    "nn_head_hidden": QB_NN_HEAD_HIDDEN,
    "nn_dropout": QB_NN_DROPOUT,
    "nn_head_hidden_overrides": QB_NN_HEAD_HIDDEN_OVERRIDES,
    "nn_lr": QB_NN_LR,
    "nn_weight_decay": QB_NN_WEIGHT_DECAY,
    "nn_epochs": QB_NN_EPOCHS,
    "nn_batch_size": QB_NN_BATCH_SIZE,
    "nn_patience": QB_NN_PATIENCE,
    "loss_weights": QB_LOSS_WEIGHTS,
    "huber_deltas": QB_HUBER_DELTAS,
    "scheduler_type": QB_SCHEDULER_TYPE,
    "cosine_t0": QB_COSINE_T0,
    "cosine_t_mult": QB_COSINE_T_MULT,
    "cosine_eta_min": QB_COSINE_ETA_MIN,
    "train_attention_nn": QB_TRAIN_ATTENTION_NN,
    "attn_d_model": QB_ATTN_D_MODEL,
    "attn_n_heads": QB_ATTN_N_HEADS,
    "attn_max_seq_len": QB_ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": QB_ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": QB_ATTN_POSITIONAL_ENCODING,
    "attn_dropout": QB_ATTN_DROPOUT,
    "attn_patience": QB_ATTN_PATIENCE,
    "attn_lr": QB_ATTN_LR,
    "attn_weight_decay": QB_ATTN_WEIGHT_DECAY,
    "attn_batch_size": QB_ATTN_BATCH_SIZE,
    "attn_history_stats": QB_ATTN_HISTORY_STATS,
    "attn_static_features": QB_ATTN_STATIC_FEATURES,
    "opp_attn_history_stats": QB_OPP_ATTN_HISTORY_STATS,
    "opp_attn_max_seq_len": QB_OPP_ATTN_MAX_SEQ_LEN,
    "attn_gated": QB_ATTN_GATED,
    "attn_gate_hidden": QB_ATTN_GATE_HIDDEN,
    "attn_gate_weight": QB_ATTN_GATE_WEIGHT,
    "head_losses": QB_HEAD_LOSSES,
    "train_elasticnet": QB_TRAIN_ELASTICNET,
    "enet_l1_ratios": QB_ENET_L1_RATIOS,
    "train_lightgbm": QB_TRAIN_LIGHTGBM,
    "lgbm_n_estimators": QB_LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": QB_LGBM_LEARNING_RATE,
    "lgbm_num_leaves": QB_LGBM_NUM_LEAVES,
    "lgbm_subsample": QB_LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": QB_LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": QB_LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": QB_LGBM_REG_ALPHA,
    "lgbm_min_child_samples": QB_LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": QB_LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": QB_LGBM_OBJECTIVE,
}


def run_qb_pipeline(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("QB", config or QB_CONFIG, train_df, val_df, test_df, seed)


def run_qb_cv_pipeline(full_df=None, test_df=None, seed=42, config=None):
    return run_cv_pipeline("QB", config or QB_CONFIG, full_df, test_df, seed)


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

        config = build_tiny_config("QB")
    else:
        config = QB_CONFIG
    if args.cv:
        run_qb_cv_pipeline(config=config)
    else:
        run_qb_pipeline(config=config)
