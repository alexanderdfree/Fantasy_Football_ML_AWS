"""End-to-end TE position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.pipeline import run_pipeline
from src.TE.te_config import (
    TE_ATTN_D_MODEL,
    TE_ATTN_DROPOUT,
    TE_ATTN_ENCODER_HIDDEN_DIM,
    TE_ATTN_GATE_HIDDEN,
    TE_ATTN_GATE_WEIGHT,
    TE_ATTN_GATED,
    TE_ATTN_HISTORY_STATS,
    TE_ATTN_MAX_SEQ_LEN,
    TE_ATTN_N_HEADS,
    TE_ATTN_POSITIONAL_ENCODING,
    TE_ATTN_STATIC_FEATURES,
    TE_ENET_L1_RATIOS,
    TE_GATED_TARGETS,
    TE_HEAD_LOSSES,
    TE_HUBER_DELTAS,
    TE_LGBM_COLSAMPLE_BYTREE,
    TE_LGBM_LEARNING_RATE,
    TE_LGBM_MAX_DEPTH,
    TE_LGBM_MIN_CHILD_SAMPLES,
    TE_LGBM_MIN_SPLIT_GAIN,
    TE_LGBM_N_ESTIMATORS,
    TE_LGBM_NUM_LEAVES,
    TE_LGBM_OBJECTIVE,
    TE_LGBM_REG_ALPHA,
    TE_LGBM_REG_LAMBDA,
    TE_LGBM_SUBSAMPLE,
    TE_LOSS_WEIGHTS,
    TE_NN_BACKBONE_LAYERS,
    TE_NN_BATCH_SIZE,
    TE_NN_DROPOUT,
    TE_NN_EPOCHS,
    TE_NN_HEAD_HIDDEN,
    TE_NN_HEAD_HIDDEN_OVERRIDES,
    TE_NN_LR,
    TE_NN_PATIENCE,
    TE_NN_WEIGHT_DECAY,
    TE_ONECYCLE_MAX_LR,
    TE_ONECYCLE_PCT_START,
    TE_OPP_ATTN_HISTORY_STATS,
    TE_OPP_ATTN_MAX_SEQ_LEN,
    TE_RIDGE_ALPHA_GRIDS,
    TE_SCHEDULER_TYPE,
    TE_SPECIFIC_FEATURES,
    TE_TARGETS,
    TE_TRAIN_ATTENTION_NN,
    TE_TRAIN_ELASTICNET,
    TE_TRAIN_LIGHTGBM,
)
from src.TE.te_data import filter_to_te
from src.TE.te_features import add_te_specific_features, fill_te_nans, get_te_feature_columns
from src.TE.te_targets import compute_te_targets

TE_CONFIG = {
    "targets": TE_TARGETS,
    "ridge_alpha_grids": TE_RIDGE_ALPHA_GRIDS,
    "specific_features": TE_SPECIFIC_FEATURES,
    "filter_fn": filter_to_te,
    "compute_targets_fn": compute_te_targets,
    "add_features_fn": add_te_specific_features,
    "fill_nans_fn": fill_te_nans,
    "get_feature_columns_fn": get_te_feature_columns,
    "nn_backbone_layers": TE_NN_BACKBONE_LAYERS,
    "nn_head_hidden": TE_NN_HEAD_HIDDEN,
    "nn_dropout": TE_NN_DROPOUT,
    "nn_head_hidden_overrides": TE_NN_HEAD_HIDDEN_OVERRIDES,
    "nn_lr": TE_NN_LR,
    "nn_weight_decay": TE_NN_WEIGHT_DECAY,
    "nn_epochs": TE_NN_EPOCHS,
    "nn_batch_size": TE_NN_BATCH_SIZE,
    "nn_patience": TE_NN_PATIENCE,
    "loss_weights": TE_LOSS_WEIGHTS,
    "huber_deltas": TE_HUBER_DELTAS,
    "scheduler_type": TE_SCHEDULER_TYPE,
    "onecycle_max_lr": TE_ONECYCLE_MAX_LR,
    "onecycle_pct_start": TE_ONECYCLE_PCT_START,
    "train_attention_nn": TE_TRAIN_ATTENTION_NN,
    "attn_d_model": TE_ATTN_D_MODEL,
    "attn_n_heads": TE_ATTN_N_HEADS,
    "attn_max_seq_len": TE_ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": TE_ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": TE_ATTN_POSITIONAL_ENCODING,
    "attn_dropout": TE_ATTN_DROPOUT,
    "attn_history_stats": TE_ATTN_HISTORY_STATS,
    "attn_static_features": TE_ATTN_STATIC_FEATURES,
    "opp_attn_history_stats": TE_OPP_ATTN_HISTORY_STATS,
    "opp_attn_max_seq_len": TE_OPP_ATTN_MAX_SEQ_LEN,
    "attn_gated": TE_ATTN_GATED,
    "attn_gate_hidden": TE_ATTN_GATE_HIDDEN,
    "attn_gate_weight": TE_ATTN_GATE_WEIGHT,
    "gated_targets": TE_GATED_TARGETS,
    "head_losses": TE_HEAD_LOSSES,
    "train_elasticnet": TE_TRAIN_ELASTICNET,
    "enet_l1_ratios": TE_ENET_L1_RATIOS,
    "train_lightgbm": TE_TRAIN_LIGHTGBM,
    "lgbm_n_estimators": TE_LGBM_N_ESTIMATORS,
    "lgbm_learning_rate": TE_LGBM_LEARNING_RATE,
    "lgbm_num_leaves": TE_LGBM_NUM_LEAVES,
    "lgbm_max_depth": TE_LGBM_MAX_DEPTH,
    "lgbm_subsample": TE_LGBM_SUBSAMPLE,
    "lgbm_colsample_bytree": TE_LGBM_COLSAMPLE_BYTREE,
    "lgbm_reg_lambda": TE_LGBM_REG_LAMBDA,
    "lgbm_reg_alpha": TE_LGBM_REG_ALPHA,
    "lgbm_min_child_samples": TE_LGBM_MIN_CHILD_SAMPLES,
    "lgbm_min_split_gain": TE_LGBM_MIN_SPLIT_GAIN,
    "lgbm_objective": TE_LGBM_OBJECTIVE,
}


def run_te_pipeline(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("TE", config or TE_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use shrunk smoke-test config (from tests/_pipeline_e2e_utils)",
    )
    args = parser.parse_args()
    if args.tiny:
        from tests._pipeline_e2e_utils import build_tiny_config

        config = build_tiny_config("TE")
    else:
        config = TE_CONFIG
    run_te_pipeline(config=config)
