"""End-to-end TE position model pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.pipeline import run_pipeline
from src.te.config import (
    ATTN_D_MODEL,
    ATTN_DROPOUT,
    ATTN_ENCODER_HIDDEN_DIM,
    ATTN_GATE_HIDDEN,
    ATTN_GATE_WEIGHT,
    ATTN_GATED,
    ATTN_HISTORY_STATS,
    ATTN_MAX_SEQ_LEN,
    ATTN_N_HEADS,
    ATTN_POSITIONAL_ENCODING,
    ATTN_STATIC_FEATURES,
    ENET_L1_RATIOS,
    GATED_TARGETS,
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
    ONECYCLE_MAX_LR,
    ONECYCLE_PCT_START,
    OPP_ATTN_HISTORY_STATS,
    OPP_ATTN_MAX_SEQ_LEN,
    RIDGE_ALPHA_GRIDS,
    SCHEDULER_TYPE,
    SPECIFIC_FEATURES,
    TARGETS,
    TRAIN_ATTENTION_NN,
    TRAIN_ELASTICNET,
    TRAIN_LIGHTGBM,
)
from src.te.data import filter_to_position
from src.te.features import add_specific_features, fill_nans, get_feature_columns
from src.te.targets import compute_targets

CONFIG = {
    "targets": TARGETS,
    "ridge_alpha_grids": RIDGE_ALPHA_GRIDS,
    "specific_features": SPECIFIC_FEATURES,
    "filter_fn": filter_to_position,
    "compute_targets_fn": compute_targets,
    "add_features_fn": add_specific_features,
    "fill_nans_fn": fill_nans,
    "get_feature_columns_fn": get_feature_columns,
    "aggregate_fn": aggregate_fn_for("TE"),
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
    "scheduler_type": SCHEDULER_TYPE,
    "onecycle_max_lr": ONECYCLE_MAX_LR,
    "onecycle_pct_start": ONECYCLE_PCT_START,
    "train_attention_nn": TRAIN_ATTENTION_NN,
    "attn_d_model": ATTN_D_MODEL,
    "attn_n_heads": ATTN_N_HEADS,
    "attn_max_seq_len": ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": ATTN_POSITIONAL_ENCODING,
    "attn_dropout": ATTN_DROPOUT,
    "attn_history_stats": ATTN_HISTORY_STATS,
    "attn_static_features": ATTN_STATIC_FEATURES,
    "opp_attn_history_stats": OPP_ATTN_HISTORY_STATS,
    "opp_attn_max_seq_len": OPP_ATTN_MAX_SEQ_LEN,
    "attn_gated": ATTN_GATED,
    "attn_gate_hidden": ATTN_GATE_HIDDEN,
    "attn_gate_weight": ATTN_GATE_WEIGHT,
    "gated_targets": GATED_TARGETS,
    "head_losses": HEAD_LOSSES,
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


def run(train_df=None, val_df=None, test_df=None, seed=42, config=None):
    return run_pipeline("TE", config or CONFIG, train_df, val_df, test_df, seed)


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
        config = CONFIG
    run(config=config)
