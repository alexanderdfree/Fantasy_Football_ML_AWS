# === K Target Decomposition ===
K_TARGETS = ["fg_points", "pat_points"]

# === K-Specific Features ===
K_SPECIFIC_FEATURES = [
    "fg_attempts_L3",
    "fg_accuracy_L5",
    "pat_volume_L3",
    "total_k_pts_L3",
    "total_k_pts_L5",
    "long_fg_rate_L3",
    "k_pts_trend",
    "k_pts_std_L3",
]

# Contextual features available for kickers
K_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "implied_team_total",
    "total_line",
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
K_CV_SPLIT_COLUMN = "week"
K_RIDGE_REFINE_POINTS = 0

# === Neural Net (small — limited dataset, tuned) ===
K_NN_BACKBONE_LAYERS = [48, 24]
K_NN_HEAD_HIDDEN = 12
K_NN_DROPOUT = 0.35
K_NN_LR = 3e-4
K_NN_WEIGHT_DECAY = 3e-4
K_NN_EPOCHS = 200
K_NN_BATCH_SIZE = 32
K_NN_PATIENCE = 25

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

# === Within-season split (kicker data is 2025 only) ===
K_VAL_WEEKS = 3
K_TEST_WEEKS = 3
K_MIN_GAMES = 4
