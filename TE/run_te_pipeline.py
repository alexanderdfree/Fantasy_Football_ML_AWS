"""End-to-end TE position model pipeline."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from TE.te_config import (
    TE_TARGETS, TE_RIDGE_ALPHA_GRIDS, TE_SPECIFIC_FEATURES,
    TE_NN_BACKBONE_LAYERS, TE_NN_HEAD_HIDDEN, TE_NN_HEAD_HIDDEN_OVERRIDES,
    TE_NN_DROPOUT, TE_NN_LR, TE_NN_WEIGHT_DECAY, TE_NN_EPOCHS, TE_NN_BATCH_SIZE,
    TE_NN_PATIENCE,
    TE_LOSS_WEIGHTS, TE_LOSS_W_TOTAL, TE_HUBER_DELTAS,
    TE_SCHEDULER_TYPE, TE_ONECYCLE_MAX_LR, TE_ONECYCLE_PCT_START,
    TE_TRAIN_ATTENTION_NN, TE_ATTN_D_MODEL, TE_ATTN_N_HEADS,
    TE_ATTN_ENCODER_HIDDEN_DIM, TE_ATTN_MAX_SEQ_LEN,
    TE_ATTN_POSITIONAL_ENCODING, TE_ATTN_DROPOUT,
    TE_ATTN_HISTORY_STATS,
)
from TE.te_data import filter_to_te
from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment
from TE.te_features import add_te_specific_features, get_te_feature_columns, fill_te_nans
from shared.pipeline import run_pipeline

TE_CONFIG = {
    "targets": TE_TARGETS,
    "ridge_alpha_grids": TE_RIDGE_ALPHA_GRIDS,
    "specific_features": TE_SPECIFIC_FEATURES,
    "filter_fn": filter_to_te,
    "compute_targets_fn": compute_te_targets,
    "add_features_fn": add_te_specific_features,
    "fill_nans_fn": fill_te_nans,
    "get_feature_columns_fn": get_te_feature_columns,
    "compute_adjustment_fn": compute_te_fumble_adjustment,
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
    "loss_w_total": TE_LOSS_W_TOTAL,
    "huber_deltas": TE_HUBER_DELTAS,
    "scheduler_type": TE_SCHEDULER_TYPE,
    "onecycle_max_lr": TE_ONECYCLE_MAX_LR,
    "onecycle_pct_start": TE_ONECYCLE_PCT_START,
    "train_weather_nn": True,
    "train_attention_nn": TE_TRAIN_ATTENTION_NN,
    "attn_d_model": TE_ATTN_D_MODEL,
    "attn_n_heads": TE_ATTN_N_HEADS,
    "attn_max_seq_len": TE_ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": TE_ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": TE_ATTN_POSITIONAL_ENCODING,
    "attn_dropout": TE_ATTN_DROPOUT,
    "attn_history_stats": TE_ATTN_HISTORY_STATS,
}


def run_te_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    return run_pipeline("TE", TE_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_te_pipeline()
