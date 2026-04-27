# === K Seasons (post-PAT rule change: 2015+) ===
SEASONS = list(range(2015, 2026))  # 2015-2025

# === K Target Decomposition ===
# 4 non-negative raw-value heads. Total fantasy points = sum with signs
# [+1, +1, -1, -1] applied at inference (see src/shared/registry.py K entry).
TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]

# === K-Specific Features ===
SPECIFIC_FEATURES = [
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
CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "implied_team_total",
    "total_line",
    # PBP Tier 1: game-level weather/venue
    "is_dome",
    "game_wind",
    "game_temp",
]

ALL_FEATURES = SPECIFIC_FEATURES + CONTEXTUAL_FEATURES

# No general features apply to kickers — all dropped
DROP_FEATURES = set()  # Not used; kickers bypass the general feature pipeline

# === Ridge ===
import numpy as np

RIDGE_ALPHA_GRIDS = {
    "fg_yard_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "pat_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "fg_misses": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "xp_misses": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}
RIDGE_CV_FOLDS = 3
CV_SPLIT_COLUMN = "season"
RIDGE_REFINE_POINTS = 0

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. Reuses RIDGE_ALPHA_GRIDS and searches over ENET_L1_RATIOS.
TRAIN_ELASTICNET = False
ENET_L1_RATIOS = [0.3, 0.5, 0.7]

# === Neural Net (2015-2025 dataset: more data allows larger model) ===
NN_BACKBONE_LAYERS = [64, 32]
NN_HEAD_HIDDEN = 16
NN_DROPOUT = 0.25
NN_LR = 3e-4
NN_WEIGHT_DECAY = 2e-4
NN_EPOCHS = 250
NN_BATCH_SIZE = 128
NN_PATIENCE = 30

# === Loss Weights ===
# Equal per-target weights.
LOSS_WEIGHTS = {
    "fg_yard_points": 1.0,
    "pat_points": 1.0,
    "fg_misses": 1.0,
    "xp_misses": 1.0,
}

# === Huber Deltas (per-target) ===
# Harmonized to 2.0 across targets.
HUBER_DELTAS = {
    "fg_yard_points": 2.0,
    "pat_points": 2.0,
    "fg_misses": 2.0,
    "xp_misses": 2.0,
}

# Per-head loss family. Default "huber"; PR 2 introduces "poisson_nll" and
# "hurdle_negbin" options. All heads on "huber" here = no behavior change.
HEAD_LOSSES = {t: "huber" for t in TARGETS}

# === Non-negative NN targets ===
# All 4 K heads are non-negative raw counts/points; signs are applied only in
# the final fantasy total aggregation, not in the per-head outputs.
NN_NON_NEGATIVE_TARGETS = set(TARGETS)

# === LR Scheduler ===
SCHEDULER_TYPE = "onecycle"
ONECYCLE_MAX_LR = 1e-3
ONECYCLE_PCT_START = 0.3

# === Cross-season split (now matching other positions) ===
MIN_GAMES = 4

# === Attention NN (nested: per-kick inner pool, per-game outer attention) ===
TRAIN_ATTENTION_NN = True
# Outer attention over prior games — mirrors RB's proven d_model=32 / n_heads=2.
ATTN_D_MODEL = 32
ATTN_N_HEADS = 2
ATTN_ENCODER_HIDDEN_DIM = 32
ATTN_MAX_GAMES = 17
ATTN_PROJECT_KV = False
ATTN_POSITIONAL_ENCODING = True
# Gated fusion is intentionally absent: MultiHeadNetWithNestedHistory does not
# implement it (only the flat MultiHeadNetWithHistory does), so exposing it
# here would be a no-op invite to config drift.
ATTN_DROPOUT = 0.05
ATTN_LR = 1e-3
ATTN_WEIGHT_DECAY = 5e-5
ATTN_BATCH_SIZE = 256
ATTN_PATIENCE = 35

# Inner pool over kicks within a game.
ATTN_KICK_DIM = 16
ATTN_MAX_KICKS_PER_GAME = 10
# Per-kick features consumed by the inner pool. Game-level context (wind,
# is_home) is replicated across kicks in the same game so the inner pool
# can condition on conditions without requiring a parallel static channel.
ATTN_KICK_STATS = [
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
# compute_features but intentionally excluded from ALL_FEATURES — the
# attention NN's static branch reads them directly via ATTN_STATIC_FEATURES
# so Ridge and the base NN continue to train only on SPECIFIC_FEATURES.
ATTN_L1_FEATURES = [
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
ATTN_STATIC_FEATURES = ATTN_L1_FEATURES + CONTEXTUAL_FEATURES

# === LightGBM ===
TRAIN_LIGHTGBM = True
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 15
LGBM_MAX_DEPTH = -1
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_REG_LAMBDA = 2.0
LGBM_REG_ALPHA = 0.1
LGBM_MIN_CHILD_SAMPLES = 30
LGBM_MIN_SPLIT_GAIN = 0.0
LGBM_OBJECTIVE = "huber"


# === Tiny config for E2E smoke tests ===
# Shrunk to 1 epoch with a 2-layer x 8-unit NN so the full pipeline runs
# in well under 20s on CPU. Used by K/tests/test_k_pipeline_e2e.py.
CONFIG_TINY = {
    "targets": TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in TARGETS},
    "ridge_cv_folds": 2,
    "cv_split_column": CV_SPLIT_COLUMN,
    "ridge_refine_points": 0,
    "specific_features": SPECIFIC_FEATURES,
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
    "loss_weights": LOSS_WEIGHTS,
    "huber_deltas": HUBER_DELTAS,
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "train_lightgbm": False,
}

# Attention tiny config — reuses CONFIG_TINY base plus a shrunk attention
# branch. Used by the E2E attention test. `attn_history_builder_fn` must be
# supplied by the caller (it captures a kicks_df in its closure).
CONFIG_TINY_ATTN = {
    **CONFIG_TINY,
    "train_attention_nn": True,
    "attn_history_structure": "nested",
    "attn_static_from_df": True,
    "attn_static_features": ATTN_STATIC_FEATURES,
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
