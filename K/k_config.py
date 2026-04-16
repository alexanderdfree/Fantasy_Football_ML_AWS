# === K Seasons (post-PAT rule change: 2015+) ===
K_SEASONS = list(range(2015, 2026))  # 2015-2025

# === K Target Decomposition ===
K_TARGETS = ["fg_points", "pat_points"]

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
    "fg_points":  [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "pat_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
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
# FG points dominate variance; heavier weight.
K_LOSS_WEIGHTS = {
    "fg_points": 1.5,
    "pat_points": 1.0,
}
K_LOSS_W_TOTAL = 0.5

# === Huber Deltas (per-target) ===
K_HUBER_DELTAS = {
    "fg_points": 3.0,
    "pat_points": 1.5,
}

# === LR Scheduler ===
K_SCHEDULER_TYPE = "onecycle"
K_ONECYCLE_MAX_LR = 1e-3
K_ONECYCLE_PCT_START = 0.3

# === Cross-season split (now matching other positions) ===
K_MIN_GAMES = 4

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
