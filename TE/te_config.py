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
# NOTE: snap_pct and air_yards_share are already lagged (shift=1) in engineer.py, safe to keep.

# Drop EWMA features — they correlate >0.98 with rolling means of the same stat,
# adding multicollinearity without unique signal.
from src.config import EWMA_STATS, EWMA_SPANS
_already_dropped_ewma = {"passing_yards"}
for _stat in EWMA_STATS:
    if _stat not in _already_dropped_ewma:
        for _span in EWMA_SPANS:
            TE_DROP_FEATURES.add(f"ewma_{_stat}_L{_span}")

# Drop L5 rolling means/std/max — sandwiched between L3 and L8 with >0.97 corr to both,
# contributing to ill-conditioned feature matrix without meaningful unique signal.
for _stat in ["fantasy_points", "fantasy_points_floor", "targets", "receptions",
              "carries", "rushing_yards", "receiving_yards"]:
    for _agg in ["mean", "std", "max"]:
        TE_DROP_FEATURES.add(f"rolling_{_agg}_{_stat}_L5")

# Weather/Vegas drops — keep 3 features with signal:
#   implied_team_total (r=-0.035), implied_opp_total (r=0.029), is_dome (r=0.027)
TE_DROP_FEATURES |= {
    "is_grass", "temp_adjusted", "wind_adjusted", "implied_total_x_wind",
    "total_line", "is_divisional", "days_rest_improved",
    "rest_advantage", "implied_total_x_dome",
}

# === Ridge ===
import numpy as np
TE_RIDGE_ALPHA_GRIDS = {
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor":   [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net (2012+ dataset: relaxed regularization) ===
TE_NN_BACKBONE_LAYERS = [96, 48]
TE_NN_HEAD_HIDDEN = 24
TE_NN_HEAD_HIDDEN_OVERRIDES = {"td_points": 32}  # larger TD head for boom/bust TEs
TE_NN_DROPOUT = 0.30
TE_NN_LR = 5e-4
TE_NN_WEIGHT_DECAY = 3e-4
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
TE_LOSS_W_TOTAL = 0.4

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

# === Attention NN (game history variant) ===
TE_TRAIN_ATTENTION_NN = True
TE_ATTN_D_MODEL = 32
TE_ATTN_N_HEADS = 2
TE_ATTN_ENCODER_HIDDEN_DIM = 0
TE_ATTN_MAX_SEQ_LEN = 17
TE_ATTN_POSITIONAL_ENCODING = True
TE_ATTN_DROPOUT = 0.0
TE_ATTN_HISTORY_STATS = [
    "fantasy_points", "fantasy_points_floor",
    "receiving_yards", "rushing_yards",
    "receiving_tds", "rushing_tds",
    "targets", "receptions", "carries",
    "snap_pct",
]
