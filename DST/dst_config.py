# === DST Target Decomposition ===
DST_TARGETS = ["defensive_scoring", "td_points", "pts_allowed_bonus"]

# === DST-Specific Features ===
# Rolling defense stats (15 features)
# NOTE: turnovers_L3 removed — exactly ints_L3 + fumble_rec_L3 (perfect linear dependency)
DST_SPECIFIC_FEATURES = [
    # Core production rolling windows
    "sacks_L3",
    "sacks_L5",              # Longer sack stability anchor
    "ints_L3",               # INTs separated — secondary quality signal
    "fumble_rec_L3",         # Fumble recoveries — more stochastic component
    "pts_allowed_L3",
    "pts_allowed_L5",
    "dst_pts_L3",
    "dst_pts_L5",
    "dst_pts_L8",            # Longer stability anchor
    # EWMA features (faster adaptation to regime changes)
    "pts_allowed_ewma",      # Exponential-weighted points allowed
    "dst_pts_ewma",          # Exponential-weighted D/ST scoring
    # Momentum / trend indicators
    "sack_trend",
    "turnover_trend",        # Turnover production trajectory
    "pts_allowed_trend",     # Defensive improvement/decline
    # Consistency metrics
    "pts_allowed_std_L3",
    "dst_scoring_std_L3",    # Base scoring consistency
]

# Contextual / matchup features (17 features)
DST_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "opp_scoring_L3",       # Short-window opponent quality (responsive)
    "opp_scoring_L5",
    "opp_turnovers_L5",     # Opponent turnover proneness
    "opp_sacks_allowed_L5", # Opponent OL sack vulnerability
    # Opposing QB quality (isolates QB signal from team-level noise)
    "opp_qb_epa_L5",        # QB efficiency — low EPA → struggling offense
    "opp_qb_int_rate_L5",   # QB interception tendency — predicts def_ints component
    "opp_qb_sack_rate_L5",  # QB pocket vulnerability — complements OL signal
    "opp_qb_rush_yds_L5",   # QB mobility — mobile QBs suppress sack production
    "rest_days",             # Days since last game (short rest = worse D)
    "div_game",              # Divisional games — different scoring patterns
    "is_dome",               # Dome games — weather-controlled
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

DST_ALL_FEATURES = DST_SPECIFIC_FEATURES + DST_CONTEXTUAL_FEATURES

# No general features — D/ST bypasses the player-level feature pipeline
DST_DROP_FEATURES = set()

# === Ridge ===
import numpy as np
DST_RIDGE_PCA_COMPONENTS = 20  # 32 features → 20 components; removes collinear dimensions

DST_RIDGE_ALPHA_GRIDS = {
    "defensive_scoring": [round(x, 4) for x in np.logspace(-1, 3.5, 20)],
    "td_points":         [round(x, 4) for x in np.logspace(0, 4, 15)],   # Higher floor — sparse target needs strong reg
    "pts_allowed_bonus": [round(x, 4) for x in np.logspace(-1, 3.5, 20)],
}

# === Neural Net (34 features, 2012-2025 dataset) ===
DST_NN_BACKBONE_LAYERS = [128, 64]   # Wider backbone for 34 features
DST_NN_HEAD_HIDDEN = 32
DST_NN_HEAD_HIDDEN_OVERRIDES = {
    "td_points": 16,             # ST-only target is simpler (mostly 0), smaller head
    "pts_allowed_bonus": 48,     # Wide range [-4, +10] with discrete tiers needs capacity
}
DST_NN_DROPOUT = 0.30             # Slightly higher — slows convergence, better generalization
# pts_allowed_bonus ranges from -4 to +10 — must NOT be clamped to >= 0
DST_NN_NON_NEGATIVE_TARGETS = {"defensive_scoring", "td_points"}
DST_NN_LR = 3e-4                  # Lower — more exploration before convergence
DST_NN_WEIGHT_DECAY = 3e-4
DST_NN_EPOCHS = 300
DST_NN_BATCH_SIZE = 128
DST_NN_PATIENCE = 35

# === Loss Weights ===
# Equal per-target weights: training objective now aligned with evaluation
# metric (total MAE). Previous scheme over-weighted pts_allowed_bonus (2.0x).
# w_total raised to 1.0.
DST_LOSS_WEIGHTS = {
    "defensive_scoring": 1.0,
    "td_points": 1.0,
    "pts_allowed_bonus": 1.0,
}
DST_LOSS_W_TOTAL = 1.0

# === Huber Deltas (per-target) ===
# Harmonized to 2.0 across targets.
DST_HUBER_DELTAS = {
    "defensive_scoring": 2.0,
    "td_points": 2.0,
    "pts_allowed_bonus": 2.0,
    "total": 3.0,               # explicit delta for total aux loss
}

# === LR Scheduler ===
DST_SCHEDULER_TYPE = "cosine_warm_restarts"
DST_COSINE_T0 = 30              # Longer first cycle for wider backbone
DST_COSINE_T_MULT = 2
DST_COSINE_ETA_MIN = 1e-5

# === LightGBM ===
DST_TRAIN_LIGHTGBM = False
DST_LGBM_N_ESTIMATORS = 300
DST_LGBM_LEARNING_RATE = 0.03
DST_LGBM_NUM_LEAVES = 15
DST_LGBM_SUBSAMPLE = 0.75
DST_LGBM_COLSAMPLE_BYTREE = 0.8
DST_LGBM_REG_LAMBDA = 2.0
DST_LGBM_REG_ALPHA = 0.1
DST_LGBM_MIN_CHILD_SAMPLES = 25
