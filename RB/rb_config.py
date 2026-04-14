# === RB Target Decomposition ===
RB_TARGETS = ["rushing_floor", "receiving_floor", "td_points"]

# === RB-Specific Features ===
RB_SPECIFIC_FEATURES = [
    "yards_per_carry_L3",
    "reception_rate_L3",
    "weighted_opportunities_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "first_down_rate_L3",
    "yac_per_reception_L3",
]

# Features to drop from the general pipeline for RB model
RB_DROP_FEATURES = set()
for _stat in ["passing_yards", "attempts"]:
    for _window in [3, 5, 8]:
        for _agg in ["mean", "std", "max"]:
            RB_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L{_window}")
for _span in [3, 5]:
    RB_DROP_FEATURES.add(f"ewma_passing_yards_L{_span}")
for _stat in ["passing_yards", "attempts"]:
    for _agg in ["mean", "std", "max"]:
        RB_DROP_FEATURES.add(f"prior_season_{_agg}_{_stat}")
RB_DROP_FEATURES |= {"pos_QB", "pos_RB", "pos_WR", "pos_TE"}

# Drop current-week features that cause data leakage (not available at prediction time)
RB_DROP_FEATURES |= {"snap_pct", "air_yards_share"}

# === Ridge ===
import numpy as np
RB_RIDGE_ALPHAS = [round(x, 4) for x in np.logspace(-2, 3, 13)]

# === Neural Net (deeper backbone — medium-large dataset) ===
RB_NN_BACKBONE_LAYERS = [128, 64, 32]
RB_NN_HEAD_HIDDEN = 32
RB_NN_DROPOUT = 0.3
RB_NN_LR = 8e-4
RB_NN_WEIGHT_DECAY = 2e-4
RB_NN_EPOCHS = 250
RB_NN_BATCH_SIZE = 256
RB_NN_PATIENCE = 20

# === Loss Weights ===
# Rushing is the primary RB floor component; slight boost.
# TD weight elevated for discrete/zero-inflated nature.
RB_LOSS_WEIGHTS = {
    "rushing_floor": 1.2,
    "receiving_floor": 1.0,
    "td_points": 2.0,
}
RB_LOSS_W_TOTAL = 0.4

# === Huber Deltas (per-target) ===
RB_HUBER_DELTAS = {
    "rushing_floor": 1.0,
    "receiving_floor": 1.0,
    "td_points": 2.0,
}

# === LR Scheduler ===
RB_SCHEDULER_TYPE = "cosine_warm_restarts"
RB_COSINE_T0 = 30
RB_COSINE_T_MULT = 2
RB_COSINE_ETA_MIN = 1e-5
