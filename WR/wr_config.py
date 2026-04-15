# === WR Target Decomposition ===
WR_TARGETS = ["receiving_floor", "rushing_floor", "td_points"]

# === WR-Specific Features ===
WR_SPECIFIC_FEATURES = [
    "yards_per_reception_L3",
    "yards_per_target_L3",
    "reception_rate_L3",
    "air_yards_per_target_L3",
    "yac_per_reception_L3",
    "team_wr_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
]

# Features to drop from the general pipeline for WR model
WR_DROP_FEATURES = set()
# WRs don't pass — drop QB passing rolling features
for _stat in ["passing_yards", "attempts"]:
    for _window in [3, 5, 8]:
        for _agg in ["mean", "std", "max"]:
            WR_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L{_window}")
for _span in [3, 5]:
    WR_DROP_FEATURES.add(f"ewma_passing_yards_L{_span}")
for _stat in ["passing_yards", "attempts"]:
    for _agg in ["mean", "std", "max"]:
        WR_DROP_FEATURES.add(f"prior_season_{_agg}_{_stat}")
# Position encoding (all WR, no variance)
WR_DROP_FEATURES |= {"pos_QB", "pos_RB", "pos_WR", "pos_TE"}
# Leakage
WR_DROP_FEATURES |= {"snap_pct", "air_yards_share"}

# === Ridge ===
import numpy as np
WR_RIDGE_ALPHA_GRIDS = {
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor":   [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net ===
# Single wide layer outperforms 3-layer funnel: [128,96,48] had 35K params
# (0.3:1 ratio) and MAE 4.299.  [96] has ~13K params and MAE 4.233.
# Depth compresses information through bottlenecks; width preserves it.
WR_NN_BACKBONE_LAYERS = [96]
WR_NN_HEAD_HIDDEN = 32
WR_NN_DROPOUT = 0.25
WR_NN_LR = 1e-3
WR_NN_WEIGHT_DECAY = 1e-4
WR_NN_EPOCHS = 250
WR_NN_BATCH_SIZE = 512
WR_NN_PATIENCE = 20

# === Loss Weights ===
# Receiving floor is the dominant WR component; boost its weight.
# Rushing is negligible for most WRs; reduce further.
WR_LOSS_WEIGHTS = {
    "receiving_floor": 1.5,
    "rushing_floor": 0.3,
    "td_points": 2.0,
}
WR_LOSS_W_TOTAL = 0.3

# === Huber Deltas (per-target) ===
WR_HUBER_DELTAS = {
    "receiving_floor": 1.5,
    "rushing_floor": 0.5,
    "td_points": 2.0,
}

# === LR Scheduler ===
WR_SCHEDULER_TYPE = "cosine_warm_restarts"
WR_COSINE_T0 = 40
WR_COSINE_T_MULT = 2
WR_COSINE_ETA_MIN = 1e-5
