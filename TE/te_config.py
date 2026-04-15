# === TE Target Decomposition ===
TE_TARGETS = ["receiving_floor", "rushing_floor", "td_points"]

# === TE-Specific Features ===
TE_SPECIFIC_FEATURES = [
    "yards_per_reception_L3",
    "reception_rate_L3",
    "yac_per_reception_L3",
    "team_te_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
    "air_yards_per_target_L3",
    "td_rate_per_target_L3",
]

# Features to drop from the general pipeline for TE model
TE_DROP_FEATURES = set()
# TEs don't pass — drop QB passing rolling features
for _stat in ["passing_yards", "attempts"]:
    for _window in [3, 5, 8]:
        for _agg in ["mean", "std", "max"]:
            TE_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L{_window}")
for _span in [3, 5]:
    TE_DROP_FEATURES.add(f"ewma_passing_yards_L{_span}")
for _stat in ["passing_yards", "attempts"]:
    for _agg in ["mean", "std", "max"]:
        TE_DROP_FEATURES.add(f"prior_season_{_agg}_{_stat}")
# Position encoding (all TE, no variance)
TE_DROP_FEATURES |= {"pos_QB", "pos_RB", "pos_WR", "pos_TE"}
# Leakage
TE_DROP_FEATURES |= {"snap_pct", "air_yards_share"}

# === Ridge ===
import numpy as np
TE_RIDGE_ALPHA_GRIDS = {
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor":   [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net (moderate data, more regularization) ===
TE_NN_BACKBONE_LAYERS = [96, 48]
TE_NN_HEAD_HIDDEN = 24
TE_NN_HEAD_HIDDEN_OVERRIDES = {"td_points": 32}  # larger TD head for boom/bust TEs
TE_NN_DROPOUT = 0.35
TE_NN_LR = 5e-4
TE_NN_WEIGHT_DECAY = 5e-4
TE_NN_EPOCHS = 300
TE_NN_BATCH_SIZE = 128
TE_NN_PATIENCE = 25

# === Loss Weights ===
# TEs are very TD-dependent (boom/bust) — highest TD weight across positions.
# Rushing is negligible; reduce further.
TE_LOSS_WEIGHTS = {
    "receiving_floor": 1.2,
    "rushing_floor": 0.2,
    "td_points": 3.0,
}
TE_LOSS_W_TOTAL = 0.3

# === Huber Deltas (per-target) ===
TE_HUBER_DELTAS = {
    "receiving_floor": 1.5,
    "rushing_floor": 0.5,
    "td_points": 3.0,
}

# === LR Scheduler ===
TE_SCHEDULER_TYPE = "onecycle"
TE_ONECYCLE_MAX_LR = 2e-3
TE_ONECYCLE_PCT_START = 0.3
