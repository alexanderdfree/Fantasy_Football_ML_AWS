"""End-to-end QB position model pipeline."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from QB.qb_config import (
    QB_TARGETS, QB_RIDGE_ALPHAS, QB_SPECIFIC_FEATURES,
    QB_NN_BACKBONE_LAYERS, QB_NN_HEAD_HIDDEN, QB_NN_DROPOUT,
    QB_NN_LR, QB_NN_WEIGHT_DECAY, QB_NN_EPOCHS, QB_NN_BATCH_SIZE,
    QB_NN_PATIENCE,
    QB_LOSS_WEIGHTS, QB_LOSS_W_TOTAL, QB_HUBER_DELTAS,
    QB_SCHEDULER_TYPE, QB_ONECYCLE_MAX_LR, QB_ONECYCLE_PCT_START,
)
from QB.qb_data import filter_to_qb
from QB.qb_targets import compute_qb_targets, compute_qb_adjustment
from QB.qb_features import add_qb_specific_features, get_qb_feature_columns, fill_qb_nans
from shared.pipeline import run_pipeline

QB_CONFIG = {
    "targets": QB_TARGETS,
    "ridge_alphas": QB_RIDGE_ALPHAS,
    "specific_features": QB_SPECIFIC_FEATURES,
    "filter_fn": filter_to_qb,
    "compute_targets_fn": compute_qb_targets,
    "add_features_fn": add_qb_specific_features,
    "fill_nans_fn": fill_qb_nans,
    "get_feature_columns_fn": get_qb_feature_columns,
    "compute_adjustment_fn": compute_qb_adjustment,
    "nn_backbone_layers": QB_NN_BACKBONE_LAYERS,
    "nn_head_hidden": QB_NN_HEAD_HIDDEN,
    "nn_dropout": QB_NN_DROPOUT,
    "nn_head_hidden_overrides": None,
    "nn_lr": QB_NN_LR,
    "nn_weight_decay": QB_NN_WEIGHT_DECAY,
    "nn_epochs": QB_NN_EPOCHS,
    "nn_batch_size": QB_NN_BATCH_SIZE,
    "nn_patience": QB_NN_PATIENCE,
    "loss_weights": QB_LOSS_WEIGHTS,
    "loss_w_total": QB_LOSS_W_TOTAL,
    "huber_deltas": QB_HUBER_DELTAS,
    "scheduler_type": QB_SCHEDULER_TYPE,
    "onecycle_max_lr": QB_ONECYCLE_MAX_LR,
    "onecycle_pct_start": QB_ONECYCLE_PCT_START,
}


def run_qb_pipeline(train_df=None, val_df=None, test_df=None, seed=42):
    return run_pipeline("QB", QB_CONFIG, train_df, val_df, test_df, seed)


if __name__ == "__main__":
    run_qb_pipeline()
