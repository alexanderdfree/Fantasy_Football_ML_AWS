"""Parity guard between per-position module-level constants and POSITION_CONFIG.

Each position's ``src/{pos}/config.py`` exposes two views of the same
hyperparameters: legacy module-level constants (``TARGETS``,
``NN_BACKBONE_LAYERS``, ...) consumed by existing importers, and a new
``POSITION_CONFIG: PositionConfig`` instance for generic consumers. The two
must stay in lockstep so we can migrate consumers one at a time without
splitting the source of truth. This test asserts the equality once for every
field every position actually sets.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# (position_string, module_path)
POSITIONS = [
    ("QB", "src.qb.config"),
    ("RB", "src.rb.config"),
    ("WR", "src.wr.config"),
    ("TE", "src.te.config"),
    ("K", "src.k.config"),
    ("DST", "src.dst.config"),
]


# Each entry maps a snake_case POSITION_CONFIG field → UPPER_CASE module-level
# constant the field is expected to mirror. Positions that don't set a given
# constant are skipped via ``hasattr``.
FIELD_MAP: dict[str, str] = {
    "targets": "TARGETS",
    "specific_features": "SPECIFIC_FEATURES",
    "include_features": "INCLUDE_FEATURES",
    "contextual_features": "CONTEXTUAL_FEATURES",
    "all_features": "ALL_FEATURES",
    "drop_features": "DROP_FEATURES",
    "ridge_alpha_grids": "RIDGE_ALPHA_GRIDS",
    "ridge_pca_components": "RIDGE_PCA_COMPONENTS",
    "ridge_cv_folds": "RIDGE_CV_FOLDS",
    "ridge_refine_points": "RIDGE_REFINE_POINTS",
    "cv_split_column": "CV_SPLIT_COLUMN",
    "train_elasticnet": "TRAIN_ELASTICNET",
    "enet_l1_ratios": "ENET_L1_RATIOS",
    "nn_backbone_layers": "NN_BACKBONE_LAYERS",
    "nn_head_hidden": "NN_HEAD_HIDDEN",
    "nn_dropout": "NN_DROPOUT",
    "nn_non_negative_targets": "NN_NON_NEGATIVE_TARGETS",
    "nn_lr": "NN_LR",
    "nn_weight_decay": "NN_WEIGHT_DECAY",
    "nn_epochs": "NN_EPOCHS",
    "nn_batch_size": "NN_BATCH_SIZE",
    "nn_patience": "NN_PATIENCE",
    "nn_head_hidden_overrides": "NN_HEAD_HIDDEN_OVERRIDES",
    "head_losses": "HEAD_LOSSES",
    "loss_weights": "LOSS_WEIGHTS",
    "huber_deltas": "HUBER_DELTAS",
    "poisson_targets": "POISSON_TARGETS",
    "gated_targets": "GATED_TARGETS",
    "scheduler_type": "SCHEDULER_TYPE",
    "cosine_t0": "COSINE_T0",
    "cosine_t_mult": "COSINE_T_MULT",
    "cosine_eta_min": "COSINE_ETA_MIN",
    "onecycle_max_lr": "ONECYCLE_MAX_LR",
    "onecycle_pct_start": "ONECYCLE_PCT_START",
    "train_attention_nn": "TRAIN_ATTENTION_NN",
    "attn_d_model": "ATTN_D_MODEL",
    "attn_n_heads": "ATTN_N_HEADS",
    "attn_encoder_hidden_dim": "ATTN_ENCODER_HIDDEN_DIM",
    "attn_max_seq_len": "ATTN_MAX_SEQ_LEN",
    "attn_positional_encoding": "ATTN_POSITIONAL_ENCODING",
    "attn_dropout": "ATTN_DROPOUT",
    "attn_lr": "ATTN_LR",
    "attn_weight_decay": "ATTN_WEIGHT_DECAY",
    "attn_batch_size": "ATTN_BATCH_SIZE",
    "attn_patience": "ATTN_PATIENCE",
    "attn_history_stats": "ATTN_HISTORY_STATS",
    "attn_static_features": "ATTN_STATIC_FEATURES",
    "attn_project_kv": "ATTN_PROJECT_KV",
    "attn_gated_fusion": "ATTN_GATED_FUSION",
    "attn_gated": "ATTN_GATED",
    "attn_gate_hidden": "ATTN_GATE_HIDDEN",
    "attn_gate_weight": "ATTN_GATE_WEIGHT",
    "opp_attn_history_stats": "OPP_ATTN_HISTORY_STATS",
    "opp_attn_max_seq_len": "OPP_ATTN_MAX_SEQ_LEN",
    "opp_attn_kind": "OPP_ATTN_KIND",
    "attn_max_games": "ATTN_MAX_GAMES",
    "attn_kick_dim": "ATTN_KICK_DIM",
    "attn_max_kicks_per_game": "ATTN_MAX_KICKS_PER_GAME",
    "attn_kick_stats": "ATTN_KICK_STATS",
    "train_lightgbm": "TRAIN_LIGHTGBM",
    "lgbm_n_estimators": "LGBM_N_ESTIMATORS",
    "lgbm_learning_rate": "LGBM_LEARNING_RATE",
    "lgbm_num_leaves": "LGBM_NUM_LEAVES",
    "lgbm_max_depth": "LGBM_MAX_DEPTH",
    "lgbm_subsample": "LGBM_SUBSAMPLE",
    "lgbm_colsample_bytree": "LGBM_COLSAMPLE_BYTREE",
    "lgbm_reg_lambda": "LGBM_REG_LAMBDA",
    "lgbm_reg_alpha": "LGBM_REG_ALPHA",
    "lgbm_min_child_samples": "LGBM_MIN_CHILD_SAMPLES",
    "lgbm_min_split_gain": "LGBM_MIN_SPLIT_GAIN",
    "lgbm_objective": "LGBM_OBJECTIVE",
    "td_model_type": "TD_MODEL_TYPE",
    "two_stage_targets": "TWO_STAGE_TARGETS",
    "ordinal_targets": "ORDINAL_TARGETS",
    "gated_ordinal_targets": "GATED_ORDINAL_TARGETS",
    "seasons": "SEASONS",
    "min_games": "MIN_GAMES",
}


@pytest.mark.parametrize("position,module_path", POSITIONS)
@pytest.mark.unit
def test_position_config_name_matches(position: str, module_path: str) -> None:
    module = importlib.import_module(module_path)
    assert module.POSITION_CONFIG.name == position


@pytest.mark.parametrize("position,module_path", POSITIONS)
@pytest.mark.unit
def test_position_config_mirrors_module_constants(position: str, module_path: str) -> None:
    module = importlib.import_module(module_path)
    pc = module.POSITION_CONFIG

    mismatches: list[tuple[str, str, Any, Any]] = []
    for field_name, const_name in FIELD_MAP.items():
        if not hasattr(module, const_name):
            continue
        module_value = getattr(module, const_name)
        config_value = getattr(pc, field_name)
        if module_value != config_value:
            mismatches.append((field_name, const_name, config_value, module_value))

    assert not mismatches, (
        f"{position}: POSITION_CONFIG drifted from module-level constants: "
        + ", ".join(f"{fld}={cv!r} vs {cn}={mv!r}" for fld, cn, cv, mv in mismatches)
    )


@pytest.mark.unit
def test_position_config_names_match_canonical_set() -> None:
    """POSITION_CONFIG.name values cover the six canonical position strings."""
    actual = set()
    for _, module_path in POSITIONS:
        module = importlib.import_module(module_path)
        actual.add(module.POSITION_CONFIG.name)
    assert actual == {"QB", "RB", "WR", "TE", "K", "DST"}
