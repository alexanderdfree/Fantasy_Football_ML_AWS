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

# NOTE: snap_pct and air_yards_share are already lagged (shift=1) in engineer.py
# so they ARE available at prediction time and should NOT be dropped.

# Drop EWMA features — they correlate >0.98 with rolling means of the same stat,
# adding multicollinearity without unique signal.
from src.config import EWMA_STATS, EWMA_SPANS
for _stat in EWMA_STATS:
    if _stat not in ("passing_yards",):  # already dropped above
        for _span in EWMA_SPANS:
            RB_DROP_FEATURES.add(f"ewma_{_stat}_L{_span}")

# Drop L5 rolling means/std/max — sandwiched between L3 and L8 with >0.97 corr to both,
# contributing to ill-conditioned feature matrix without meaningful unique signal.
for _stat in ["fantasy_points", "fantasy_points_floor", "targets", "receptions",
              "carries", "rushing_yards", "receiving_yards", "snap_pct"]:
    for _agg in ["mean", "std", "max"]:
        RB_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L5")

# === Ridge ===
import numpy as np
RB_RIDGE_ALPHAS = [round(x, 4) for x in np.logspace(-2, 3, 13)]

# === Neural Net (deeper backbone — medium-large dataset) ===
RB_NN_BACKBONE_LAYERS = [128, 64, 32]
RB_NN_HEAD_HIDDEN = 32
RB_NN_DROPOUT = 0.3
RB_NN_LR = 8e-4
RB_NN_WEIGHT_DECAY = 2e-4
RB_NN_EPOCHS = 300
RB_NN_BATCH_SIZE = 256
RB_NN_PATIENCE = 30

# === Loss Weights ===
# Rushing is the primary RB floor component; slight boost.
# TD weight elevated for discrete/zero-inflated nature.
RB_LOSS_WEIGHTS = {
    "rushing_floor": 1.2,
    "receiving_floor": 1.0,
    "td_points": 2.0,
}
RB_LOSS_W_TOTAL = 0.6

# === Huber Deltas (per-target) ===
RB_HUBER_DELTAS = {
    "rushing_floor": 1.0,
    "receiving_floor": 1.0,
    "td_points": 2.0,
}

# === LR Scheduler ===
# ReduceLROnPlateau: adaptively lowers LR when val loss stalls,
# better suited than cosine warm restarts for finding minima
# on a flat loss surface with this dataset size.
RB_SCHEDULER_TYPE = "plateau"
RB_PLATEAU_FACTOR = 0.5
RB_PLATEAU_PATIENCE = 8
