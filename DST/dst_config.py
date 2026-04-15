# === DST Target Decomposition ===
DST_TARGETS = ["defensive_scoring", "td_points", "pts_allowed_bonus"]

# === DST-Specific Features ===
DST_SPECIFIC_FEATURES = [
    "sacks_L3",
    "turnovers_L3",
    "pts_allowed_L3",
    "pts_allowed_L5",
    "dst_pts_L3",
    "dst_pts_L5",
    "sack_trend",
    "pts_allowed_std_L3",
]

# Contextual / matchup features
DST_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "opp_scoring_L5",
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

DST_ALL_FEATURES = DST_SPECIFIC_FEATURES + DST_CONTEXTUAL_FEATURES

# No general features — D/ST bypasses the player-level feature pipeline
DST_DROP_FEATURES = set()

# === Ridge ===
import numpy as np
DST_RIDGE_ALPHA_GRIDS = {
    "defensive_scoring": [round(x, 4) for x in np.logspace(-1, 3, 15)],
    "td_points":         [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "pts_allowed_bonus": [round(x, 4) for x in np.logspace(-1, 3, 15)],
}

# === Neural Net (2012+ dataset: relaxed regularization, larger batches) ===
DST_NN_BACKBONE_LAYERS = [96, 48]
DST_NN_HEAD_HIDDEN = 24
DST_NN_HEAD_HIDDEN_OVERRIDES = {"td_points": 32}  # Larger head for sparse target
DST_NN_DROPOUT = 0.25
DST_NN_LR = 6e-4
DST_NN_WEIGHT_DECAY = 2e-4
DST_NN_EPOCHS = 250
DST_NN_BATCH_SIZE = 128
DST_NN_PATIENCE = 25

# === Loss Weights ===
# pts_allowed_bonus has highest variance and importance.
# td_points is sparse/zero-inflated — elevated weight.
DST_LOSS_WEIGHTS = {
    "defensive_scoring": 1.0,
    "td_points": 2.5,
    "pts_allowed_bonus": 1.5,
}
DST_LOSS_W_TOTAL = 0.4

# === Huber Deltas (per-target) ===
DST_HUBER_DELTAS = {
    "defensive_scoring": 2.0,
    "td_points": 3.0,
    "pts_allowed_bonus": 3.0,
}

# === LR Scheduler ===
DST_SCHEDULER_TYPE = "cosine_warm_restarts"
DST_COSINE_T0 = 25
DST_COSINE_T_MULT = 2
DST_COSINE_ETA_MIN = 1e-5
