"""End-to-end WR position model pipeline."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from WR.wr_config import (
    WR_TARGETS, WR_RIDGE_ALPHA_GRIDS, WR_RIDGE_PCA_COMPONENTS, WR_SPECIFIC_FEATURES,
    WR_NN_BACKBONE_LAYERS, WR_NN_HEAD_HIDDEN, WR_NN_DROPOUT,
    WR_NN_LR, WR_NN_WEIGHT_DECAY, WR_NN_EPOCHS, WR_NN_BATCH_SIZE,
    WR_NN_PATIENCE,
    WR_LOSS_WEIGHTS, WR_LOSS_W_TOTAL, WR_HUBER_DELTAS,
    WR_SCHEDULER_TYPE, WR_COSINE_T0, WR_COSINE_T_MULT, WR_COSINE_ETA_MIN,
    WR_TRAIN_ATTENTION_NN, WR_ATTN_D_MODEL, WR_ATTN_N_HEADS,
    WR_ATTN_ENCODER_HIDDEN_DIM, WR_ATTN_MAX_SEQ_LEN,
    WR_ATTN_POSITIONAL_ENCODING, WR_ATTN_DROPOUT,
    WR_ATTN_HISTORY_STATS,
    WR_ATTN_GATED_TD, WR_ATTN_TD_GATE_HIDDEN, WR_ATTN_TD_GATE_WEIGHT,
)
from WR.wr_data import filter_to_wr
from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment
from WR.wr_features import add_wr_specific_features, get_wr_feature_columns, fill_wr_nans
from shared.pipeline import run_pipeline, run_cv_pipeline

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
    "compute_adjustment_fn": compute_wr_fumble_adjustment,
    "nn_backbone_layers": WR_NN_BACKBONE_LAYERS,
    "nn_head_hidden": WR_NN_HEAD_HIDDEN,
    "nn_dropout": WR_NN_DROPOUT,
    "nn_head_hidden_overrides": None,
    "nn_lr": WR_NN_LR,
    "nn_weight_decay": WR_NN_WEIGHT_DECAY,
    "nn_epochs": WR_NN_EPOCHS,
    "nn_batch_size": WR_NN_BATCH_SIZE,
    "nn_patience": WR_NN_PATIENCE,
    "loss_weights": WR_LOSS_WEIGHTS,
    "loss_w_total": WR_LOSS_W_TOTAL,
    "huber_deltas": WR_HUBER_DELTAS,
    "scheduler_type": WR_SCHEDULER_TYPE,
    "cosine_t0": WR_COSINE_T0,
    "cosine_t_mult": WR_COSINE_T_MULT,
    "cosine_eta_min": WR_COSINE_ETA_MIN,
    "train_weather_nn": True,
    "train_attention_nn": WR_TRAIN_ATTENTION_NN,
    "attn_d_model": WR_ATTN_D_MODEL,
    "attn_n_heads": WR_ATTN_N_HEADS,
    "attn_max_seq_len": WR_ATTN_MAX_SEQ_LEN,
    "attn_encoder_hidden_dim": WR_ATTN_ENCODER_HIDDEN_DIM,
    "attn_positional_encoding": WR_ATTN_POSITIONAL_ENCODING,
    "attn_dropout": WR_ATTN_DROPOUT,
    "attn_history_stats": WR_ATTN_HISTORY_STATS,
    "attn_gated_td": WR_ATTN_GATED_TD,
    "attn_td_gate_hidden": WR_ATTN_TD_GATE_HIDDEN,
    "attn_td_gate_weight": WR_ATTN_TD_GATE_WEIGHT,
}


def run_wr_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    return run_pipeline("WR", WR_CONFIG, train_df, val_df, test_df, seed)


def run_wr_cv_pipeline(full_df=None, test_df=None, seed=42):
    return run_cv_pipeline("WR", WR_CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Use expanding-window CV")
    args = parser.parse_args()
    if args.cv:
        run_wr_cv_pipeline()
    else:
        run_wr_pipeline()
