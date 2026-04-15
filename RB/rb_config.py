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
    "rushing_first_down_rate_L3",
    "receiving_first_down_rate_L3",
    "yac_per_reception_L3",
    "receiving_epa_per_target_L3",
    "air_yards_per_target_L3",
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
RB_RIDGE_ALPHA_GRIDS = {
    "rushing_floor":   [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 2.5, 20)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net ===
# [128, 64] two-layer backbone — single [128] was underfitting (early stop epoch 54,
# flat val loss from epoch 3). Added depth + larger heads + less regularization.
RB_NN_BACKBONE_LAYERS = [128, 64]
RB_NN_HEAD_HIDDEN = 48
RB_NN_DROPOUT = 0.15
RB_NN_LR = 1e-3
RB_NN_WEIGHT_DECAY = 5e-5
RB_NN_EPOCHS = 300
RB_NN_BATCH_SIZE = 256
RB_NN_PATIENCE = 30
# TD head gets a larger hidden layer — td_points has the highest MAE (zero-inflated,
# discrete) and benefits from more capacity to model the sparse signal.
RB_NN_HEAD_HIDDEN_OVERRIDES = {"td_points": 64}

# === Loss Weights ===
# Rushing is the primary RB floor component; slight boost.
# TD weight elevated for discrete/zero-inflated nature.
RB_LOSS_WEIGHTS = {
    "rushing_floor": 1.2,
    "receiving_floor": 1.0,
    "td_points": 2.0,
}
RB_LOSS_W_TOTAL = 0.25

# === Huber Deltas (per-target) ===
# Widened from 1.0/1.5/2.0 — tight deltas caused flat gradient plateau,
# encouraging mean-clustering. Wider deltas keep quadratic (MSE-like) gradient
# signal for errors up to 2-3 pts, only switching to robust linear for outliers.
RB_HUBER_DELTAS = {
    "rushing_floor": 2.0,
    "receiving_floor": 2.5,
    "td_points": 3.0,
}

# === LR Scheduler ===
RB_SCHEDULER_TYPE = "cosine_warm_restarts"
RB_COSINE_T0 = 40
RB_COSINE_T_MULT = 2
RB_COSINE_ETA_MIN = 1e-5

# === Attention NN (game history variant) ===
RB_TRAIN_ATTENTION_NN = True
# Keep d_model=32 (proven baseline) and n_heads=2 (larger values overfit on 15K samples).
RB_ATTN_D_MODEL = 32
RB_ATTN_N_HEADS = 2
RB_ATTN_MAX_SEQ_LEN = 17
# K/V projections disabled — at d_model=32 the 2K extra params hurt optimization
# more than they help (tested: 4.330 MAE with vs 4.228 without).
RB_ATTN_PROJECT_KV = False
# Positional encoding: lightweight (17×32=544 params) temporal ordering signal so
# attention can distinguish recent games from older ones.
RB_ATTN_POSITIONAL_ENCODING = True
RB_ATTN_GATED_FUSION = False
# Very light attention dropout for regularization.
RB_ATTN_DROPOUT = 0.05
# Standard training params match the base NN.
RB_ATTN_LR = 1e-3
RB_ATTN_WEIGHT_DECAY = 5e-5
RB_ATTN_BATCH_SIZE = 256
RB_ATTN_PATIENCE = 35
RB_ATTN_HISTORY_STATS = [
    "fantasy_points", "fantasy_points_floor",
    "rushing_yards", "receiving_yards",
    "rushing_tds", "receiving_tds",
    "carries", "targets", "receptions",
    "snap_pct",
    "rushing_first_downs", "receiving_first_downs",
]
