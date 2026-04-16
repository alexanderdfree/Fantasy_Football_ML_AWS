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
# is_home has zero variance in training data (rank-deficient column)
WR_DROP_FEATURES.add("is_home")
# NOTE: snap_pct and air_yards_share are already lagged (shift=1) in engineer.py, safe to keep.

# Drop EWMA features — they correlate >0.98 with rolling means of the same stat,
# adding multicollinearity without unique signal.
from src.config import EWMA_STATS, EWMA_SPANS
_already_dropped_ewma = {"passing_yards"}
for _stat in EWMA_STATS:
    if _stat not in _already_dropped_ewma:
        for _span in EWMA_SPANS:
            WR_DROP_FEATURES.add(f"ewma_{_stat}_L{_span}")

# Drop L5 rolling means/std/max — sandwiched between L3 and L8 with >0.97 corr to both,
# contributing to ill-conditioned feature matrix without meaningful unique signal.
for _stat in ["fantasy_points", "fantasy_points_floor", "targets", "receptions",
              "carries", "rushing_yards", "receiving_yards"]:
    for _agg in ["mean", "std", "max"]:
        WR_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L5")

# Weather/Vegas drops — keep 6 features with signal:
#   implied_opp_total, is_dome, implied_total_x_dome, wind_adjusted,
#   temp_adjusted, total_line
WR_DROP_FEATURES |= {
    "is_grass", "rest_advantage", "implied_total_x_wind",
    "is_divisional", "days_rest_improved", "implied_team_total",
}

# === Ridge ===
import numpy as np
# PCR: 30 components. Benchmark showed -0.094 MAE vs no-PCA baseline (4.507 → 4.413).
# PCA removes collinear directions the alpha grid can't fully address.
WR_RIDGE_PCA_COMPONENTS = 30
WR_RIDGE_ALPHA_GRIDS = {
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor":   [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net ===
# 2012+ dataset: widened from [96] to [128] to exploit largest training set.
# Largest position dataset can support more capacity with less overfitting risk.
WR_NN_BACKBONE_LAYERS = [128]
WR_NN_HEAD_HIDDEN = 32
WR_NN_DROPOUT = 0.20
WR_NN_LR = 1e-3
WR_NN_WEIGHT_DECAY = 1e-4
WR_NN_EPOCHS = 250
WR_NN_BATCH_SIZE = 512
WR_NN_PATIENCE = 25

# === Loss Weights ===
# Receiving floor is the dominant WR component; boost its weight.
# Rushing is negligible for most WRs; reduce further.
WR_LOSS_WEIGHTS = {
    "receiving_floor": 1.5,
    "rushing_floor": 0.3,
    "td_points": 2.0,
}
WR_LOSS_W_TOTAL = 0.4

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

# === Attention NN (game history variant) ===
WR_TRAIN_ATTENTION_NN = True
WR_ATTN_D_MODEL = 32
WR_ATTN_N_HEADS = 2
WR_ATTN_ENCODER_HIDDEN_DIM = 0
WR_ATTN_MAX_SEQ_LEN = 17
WR_ATTN_POSITIONAL_ENCODING = True
WR_ATTN_DROPOUT = 0.0
WR_ATTN_HISTORY_STATS = [
    "fantasy_points", "fantasy_points_floor",
    "receiving_yards", "rushing_yards",
    "receiving_tds", "rushing_tds",
    "targets", "receptions", "carries",
    "snap_pct",
]
