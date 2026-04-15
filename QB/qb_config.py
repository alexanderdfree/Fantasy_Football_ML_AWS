# === QB Target Decomposition ===
QB_TARGETS = ["passing_floor", "rushing_floor", "td_points"]

# === QB-Specific Features ===
QB_SPECIFIC_FEATURES = [
    "completion_pct_L3",
    "yards_per_attempt_L3",
    "td_rate_L3",
    "int_rate_L3",
    "sack_rate_L3",
    "qb_rushing_share_L3",
    "passing_epa_per_dropback_L3",
    "deep_ball_rate_L3",
    "pass_first_down_rate_L3",
    "rushing_epa_per_carry_L3",
    "rush_first_down_rate_L3",
    "yac_rate_L3",
    "sack_damage_per_dropback_L3",
]

# Features to drop from the general pipeline for QB model
QB_DROP_FEATURES = set()
# QBs don't receive passes — drop receiver-centric rolling features
for _stat in ["targets", "receptions", "receiving_yards"]:
    for _window in [3, 5, 8]:
        for _agg in ["mean", "std", "max"]:
            QB_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L{_window}")
for _span in [3, 5]:
    for _stat in ["targets", "receiving_yards"]:
        QB_DROP_FEATURES.add(f"ewma_{_stat}_L{_span}")
for _stat in ["targets", "receptions", "receiving_yards"]:
    for _agg in ["mean", "std", "max"]:
        QB_DROP_FEATURES.add(f"prior_season_{_agg}_{_stat}")
# Drop share features designed for skill position players
QB_DROP_FEATURES |= {"target_share_L3", "target_share_L5"}
QB_DROP_FEATURES |= {"air_yards_share"}
# Position encoding (all QB, no variance)
QB_DROP_FEATURES |= {"pos_QB", "pos_RB", "pos_WR", "pos_TE"}
# QBs have ~0 targets — trend_targets is pure noise
QB_DROP_FEATURES.add("trend_targets")
# NOTE: snap_pct is already lagged (shift=1) in engineer.py, safe to keep.

# Drop EWMA features — they correlate >0.98 with rolling means of the same stat,
# adding multicollinearity without unique signal.
from src.config import EWMA_STATS, EWMA_SPANS
_already_dropped_ewma = {"targets", "receiving_yards", "passing_yards"}
for _stat in EWMA_STATS:
    if _stat not in _already_dropped_ewma:
        for _span in EWMA_SPANS:
            QB_DROP_FEATURES.add(f"ewma_{_stat}_L{_span}")

# Drop L5 rolling means/std/max — sandwiched between L3 and L8 with >0.97 corr to both,
# contributing to ill-conditioned feature matrix without meaningful unique signal.
for _stat in ["fantasy_points", "fantasy_points_floor", "carries", "rushing_yards",
              "passing_yards", "attempts"]:
    for _agg in ["mean", "std", "max"]:
        QB_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L5")

# === Ridge ===
import numpy as np
QB_RIDGE_ALPHA_GRIDS = {
    "passing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "td_points":     [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net (2012+ dataset: wider backbone, relaxed regularization) ===
QB_NN_BACKBONE_LAYERS = [96, 48]
QB_NN_HEAD_HIDDEN = 20
QB_NN_DROPOUT = 0.35
QB_NN_LR = 5e-4
QB_NN_WEIGHT_DECAY = 3e-4
QB_NN_EPOCHS = 300
QB_NN_BATCH_SIZE = 128
QB_NN_PATIENCE = 25

# === Loss Weights ===
# Passing floor is the primary QB scoring driver; boost its weight.
# TD weight elevated for discrete/zero-inflated nature.
QB_LOSS_WEIGHTS = {
    "passing_floor": 1.5,
    "rushing_floor": 0.8,
    "td_points": 3.0,
}
QB_LOSS_W_TOTAL = 0.3

# === Huber Deltas (per-target) ===
QB_HUBER_DELTAS = {
    "passing_floor": 1.5,
    "rushing_floor": 1.0,
    "td_points": 3.0,
}

# === LR Scheduler ===
QB_SCHEDULER_TYPE = "onecycle"
QB_ONECYCLE_MAX_LR = 2e-3
QB_ONECYCLE_PCT_START = 0.3
