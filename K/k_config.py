# === K Seasons (post-PAT rule change: 2015+) ===
K_SEASONS = list(range(2015, 2026))  # 2015-2025

# === K Target Decomposition ===
# 4 non-negative raw-value heads. Total fantasy points = sum with signs
# [+1, +1, -1, -1] applied at inference (see shared/registry.py K entry).
K_TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]

# === K-Specific Features ===
K_SPECIFIC_FEATURES = [
    # Rolling performance
    "fg_attempts_L3",
    "fg_accuracy_L5",
    "pat_volume_L3",
    "total_k_pts_L3",
    "long_fg_rate_L3",
    "k_pts_trend",
    "k_pts_std_L3",
    # PBP: distance & difficulty
    "avg_fg_distance_L3",
    "avg_fg_prob_L3",
    # PBP: situational accuracy
    "fg_pct_40plus_L5",
    "q4_fg_rate_L5",
    "xp_accuracy_L5",
]

# Contextual features available for kickers
K_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "implied_team_total",
    "total_line",
    # PBP Tier 1: game-level weather/venue
    "is_dome",
    "game_wind",
    "game_temp",
]

K_ALL_FEATURES = K_SPECIFIC_FEATURES + K_CONTEXTUAL_FEATURES

# No general features apply to kickers — all dropped
K_DROP_FEATURES = set()  # Not used; kickers bypass the general feature pipeline

# === Ridge ===
import numpy as np

K_RIDGE_ALPHA_GRIDS = {
    "fg_yard_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "pat_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "fg_misses": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "xp_misses": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}
K_RIDGE_CV_FOLDS = 3
K_CV_SPLIT_COLUMN = "season"
K_RIDGE_REFINE_POINTS = 0

# === Neural Net (2015-2025 dataset: more data allows larger model) ===
K_NN_BACKBONE_LAYERS = [64, 32]
K_NN_HEAD_HIDDEN = 16
K_NN_DROPOUT = 0.25
K_NN_LR = 3e-4
K_NN_WEIGHT_DECAY = 2e-4
K_NN_EPOCHS = 250
K_NN_BATCH_SIZE = 128
K_NN_PATIENCE = 30

# === Loss Weights ===
# Equal per-target weights: training objective now aligned with evaluation
# metric (total MAE). w_total raised to 1.0.
K_LOSS_WEIGHTS = {
    "fg_yard_points": 1.0,
    "pat_points": 1.0,
    "fg_misses": 1.0,
    "xp_misses": 1.0,
}
K_LOSS_W_TOTAL = 1.0

# === Huber Deltas (per-target) ===
# Harmonized to 2.0 across targets.
K_HUBER_DELTAS = {
    "fg_yard_points": 2.0,
    "pat_points": 2.0,
    "fg_misses": 2.0,
    "xp_misses": 2.0,
    "total": 2.0,  # explicit delta for total aux loss (kickers have narrow range)
}

# === Non-negative NN targets ===
# All 4 K heads are non-negative raw counts/points; signs are applied only in
# the final fantasy total aggregation, not in the per-head outputs.
K_NN_NON_NEGATIVE_TARGETS = set(K_TARGETS)

# === LR Scheduler ===
K_SCHEDULER_TYPE = "onecycle"
K_ONECYCLE_MAX_LR = 1e-3
K_ONECYCLE_PCT_START = 0.3

# === Cross-season split (now matching other positions) ===
K_MIN_GAMES = 4

# === Attention NN (nested: per-kick inner pool, per-game outer attention) ===
K_TRAIN_ATTENTION_NN = True
# Outer attention over prior games — mirrors RB's proven d_model=32 / n_heads=2.
K_ATTN_D_MODEL = 32
K_ATTN_N_HEADS = 2
K_ATTN_ENCODER_HIDDEN_DIM = 32
K_ATTN_MAX_GAMES = 17
K_ATTN_PROJECT_KV = False
K_ATTN_POSITIONAL_ENCODING = True
K_ATTN_GATED_FUSION = False
K_ATTN_DROPOUT = 0.05
K_ATTN_LR = 1e-3
K_ATTN_WEIGHT_DECAY = 5e-5
K_ATTN_BATCH_SIZE = 256
K_ATTN_PATIENCE = 35

# Inner pool over kicks within a game.
K_ATTN_KICK_DIM = 16
K_ATTN_MAX_KICKS_PER_GAME = 10
# Per-kick features consumed by the inner pool. Game-level context (wind,
# is_home) is replicated across kicks in the same game so the inner pool
# can condition on conditions without requiring a parallel static channel.
K_ATTN_KICK_STATS = [
    "is_fg",
    "is_xp",
    "kick_distance",
    "kick_made",
    "fg_prob",
    "is_q4",
    "score_diff",
    "game_wind",
    "is_home",
]

# L1 (shift-1) rolling features. Engineered into the DataFrame by
# compute_k_features but intentionally excluded from K_ALL_FEATURES — the
# attention NN's static branch reads them directly via K_ATTN_STATIC_FEATURES
# so Ridge and the base NN continue to train only on K_SPECIFIC_FEATURES.
K_ATTN_L1_FEATURES = [
    "fg_attempts_L1",
    "fg_accuracy_L1",
    "pat_volume_L1",
    "total_k_pts_L1",
    "long_fg_rate_L1",
    "avg_fg_distance_L1",
    "avg_fg_prob_L1",
    "fg_pct_40plus_L1",
    "q4_fg_rate_L1",
    "xp_accuracy_L1",
]
K_ATTN_STATIC_FEATURES = K_ATTN_L1_FEATURES + K_CONTEXTUAL_FEATURES

# === LightGBM ===
K_TRAIN_LIGHTGBM = False
K_LGBM_N_ESTIMATORS = 300
K_LGBM_LEARNING_RATE = 0.05
K_LGBM_NUM_LEAVES = 15
K_LGBM_SUBSAMPLE = 0.8
K_LGBM_COLSAMPLE_BYTREE = 0.8
K_LGBM_REG_LAMBDA = 2.0
K_LGBM_REG_ALPHA = 0.1
K_LGBM_MIN_CHILD_SAMPLES = 30


# === Tiny config for E2E smoke tests ===
# Shrunk to 1 epoch with a 2-layer x 8-unit NN so the full pipeline runs
# in well under 20s on CPU. Used by K/tests/test_k_pipeline_e2e.py.
K_CONFIG_TINY = {
    "targets": K_TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in K_TARGETS},
    "ridge_cv_folds": 2,
    "cv_split_column": K_CV_SPLIT_COLUMN,
    "ridge_refine_points": 0,
    "specific_features": K_SPECIFIC_FEATURES,
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 1,
    "nn_log_every": 1,
    "loss_weights": K_LOSS_WEIGHTS,
    "loss_w_total": K_LOSS_W_TOTAL,
    "huber_deltas": K_HUBER_DELTAS,
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "train_lightgbm": False,
}

# Attention tiny config — reuses K_CONFIG_TINY base plus a shrunk attention
# branch. Used by the E2E attention test. `attn_history_builder_fn` must be
# supplied by the caller (it captures a kicks_df in its closure).
K_CONFIG_TINY_ATTN = {
    **K_CONFIG_TINY,
    "train_attention_nn": True,
    "attn_history_structure": "nested",
    "attn_static_from_df": True,
    "attn_static_features": K_ATTN_STATIC_FEATURES,
    "attn_d_model": 8,
    "attn_n_heads": 1,
    "attn_kick_dim": 4,
    "attn_encoder_hidden_dim": 0,
    "attn_project_kv": False,
    "attn_positional_encoding": True,
    "attn_dropout": 0.0,
    "attn_lr": 1e-3,
    "attn_weight_decay": 0.0,
    "attn_batch_size": 32,
    "attn_patience": 1,
}
