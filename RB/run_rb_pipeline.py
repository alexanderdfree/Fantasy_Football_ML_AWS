"""End-to-end RB position model pipeline."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from RB.rb_config import (
    RB_TARGETS, RB_RIDGE_ALPHA_GRIDS, RB_SPECIFIC_FEATURES,
    RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_DROPOUT,
    RB_NN_LR, RB_NN_WEIGHT_DECAY, RB_NN_EPOCHS, RB_NN_BATCH_SIZE,
    RB_NN_PATIENCE,
    RB_LOSS_WEIGHTS, RB_LOSS_W_TOTAL, RB_HUBER_DELTAS,
    RB_SCHEDULER_TYPE, RB_COSINE_T0, RB_COSINE_T_MULT, RB_COSINE_ETA_MIN,
)
from RB.rb_data import filter_to_rb
from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns, fill_rb_nans
from shared.pipeline import run_pipeline, run_cv_pipeline

RB_CONFIG = {
    "targets": RB_TARGETS,
    "ridge_alpha_grids": RB_RIDGE_ALPHA_GRIDS,
    "specific_features": RB_SPECIFIC_FEATURES,
    "filter_fn": filter_to_rb,
    "compute_targets_fn": compute_rb_targets,
    "add_features_fn": add_rb_specific_features,
    "fill_nans_fn": fill_rb_nans,
    "get_feature_columns_fn": get_rb_feature_columns,
    "compute_adjustment_fn": compute_fumble_adjustment,
    "nn_backbone_layers": RB_NN_BACKBONE_LAYERS,
    "nn_head_hidden": RB_NN_HEAD_HIDDEN,
    "nn_dropout": RB_NN_DROPOUT,
    "nn_head_hidden_overrides": None,
    "nn_lr": RB_NN_LR,
    "nn_weight_decay": RB_NN_WEIGHT_DECAY,
    "nn_epochs": RB_NN_EPOCHS,
    "nn_batch_size": RB_NN_BATCH_SIZE,
    "nn_patience": RB_NN_PATIENCE,
    "loss_weights": RB_LOSS_WEIGHTS,
    "loss_w_total": RB_LOSS_W_TOTAL,
    "huber_deltas": RB_HUBER_DELTAS,
    "scheduler_type": RB_SCHEDULER_TYPE,
    "cosine_t0": RB_COSINE_T0,
    "cosine_t_mult": RB_COSINE_T_MULT,
    "cosine_eta_min": RB_COSINE_ETA_MIN,
    "train_weather_nn": True,
}


def run_rb_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    return run_pipeline("RB", RB_CONFIG, train_df, val_df, test_df, seed)


def run_rb_cv_pipeline(full_df=None, test_df=None, seed=42):
    return run_cv_pipeline("RB", RB_CONFIG, full_df, test_df, seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Use expanding-window CV")
    args = parser.parse_args()
    if args.cv:
        run_rb_cv_pipeline()
    else:
        run_rb_pipeline()
