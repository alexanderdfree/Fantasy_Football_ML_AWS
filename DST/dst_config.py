# === DST Target Decomposition ===
DST_TARGETS = [
    "defensive_production",
    "def_td_points",
    "st_production",
    "points_allowed",
    "yards_allowed",
]

# === DST-Specific Features ===
# Rolling defense stats
# NOTE: turnovers_L3 removed — exactly ints_L3 + fumble_rec_L3 (perfect linear dependency)
DST_SPECIFIC_FEATURES = [
    # Core production rolling windows
    "sacks_L3",
    "sacks_L5",  # Longer sack stability anchor
    "ints_L3",  # INTs separated — secondary quality signal
    "fumble_rec_L3",  # Fumble recoveries — more stochastic component
    "forced_fumbles_L3",  # Forced fumbles — pressure proxy
    "blocked_kicks_L5",  # Blocked kicks — rare, needs longer window
    "pts_allowed_L3",
    "pts_allowed_L5",
    "yards_allowed_L3",
    "yards_allowed_L5",
    "yards_allowed_ewma",
    "dst_pts_L3",
    "dst_pts_L5",
    "dst_pts_L8",  # Longer stability anchor
    # EWMA features (faster adaptation to regime changes)
    "pts_allowed_ewma",  # Exponential-weighted points allowed
    "dst_pts_ewma",  # Exponential-weighted D/ST scoring
    # Momentum / trend indicators
    "sack_trend",
    "turnover_trend",  # Turnover production trajectory
    "pts_allowed_trend",  # Defensive improvement/decline
    # Consistency metrics
    "pts_allowed_std_L3",
    "dst_scoring_std_L3",  # Base scoring consistency
]

# Contextual / matchup features (17 features)
DST_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "opp_scoring_L3",  # Short-window opponent quality (responsive)
    "opp_scoring_L5",
    "opp_turnovers_L5",  # Opponent turnover proneness
    "opp_sacks_allowed_L5",  # Opponent OL sack vulnerability
    # Opposing QB quality (isolates QB signal from team-level noise)
    "opp_qb_epa_L5",  # QB efficiency — low EPA → struggling offense
    "opp_qb_int_rate_L5",  # QB interception tendency — predicts def_ints component
    "opp_qb_sack_rate_L5",  # QB pocket vulnerability — complements OL signal
    "opp_qb_rush_yds_L5",  # QB mobility — mobile QBs suppress sack production
    "rest_days",  # Days since last game (short rest = worse D)
    "div_game",  # Divisional games — different scoring patterns
    "is_dome",  # Dome games — weather-controlled
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

DST_ALL_FEATURES = DST_SPECIFIC_FEATURES + DST_CONTEXTUAL_FEATURES

# No general features — D/ST bypasses the player-level feature pipeline
DST_DROP_FEATURES = set()

# === Ridge ===
import numpy as np

DST_RIDGE_PCA_COMPONENTS = 20  # features → 20 components; removes collinear dimensions

# Raw PA/YA targets have much larger magnitudes (0–55 and 0–600) than the
# production heads, so their alpha grid tops out higher to cover stronger L2.
_DST_ALPHA_STANDARD = [round(x, 4) for x in np.logspace(-1, 3.5, 20)]
_DST_ALPHA_RAW_SCALE = [round(x, 4) for x in np.logspace(-1, 5, 20)]

DST_RIDGE_ALPHA_GRIDS = {
    "defensive_production": _DST_ALPHA_STANDARD,
    "def_td_points": [
        round(x, 4) for x in np.logspace(0, 4, 15)
    ],  # Higher floor — sparse target needs strong reg
    "st_production": [
        round(x, 4) for x in np.logspace(0, 4, 15)
    ],  # Sparse (mostly 0); needs strong reg
    "points_allowed": _DST_ALPHA_RAW_SCALE,  # Raw 0–55 scale
    "yards_allowed": _DST_ALPHA_RAW_SCALE,  # Raw 0–600 scale
}

# === Neural Net ===
DST_NN_BACKBONE_LAYERS = [128, 64]  # Wider backbone for many features
DST_NN_HEAD_HIDDEN = 32
DST_NN_HEAD_HIDDEN_OVERRIDES = {
    "def_td_points": 16,  # Sparse target; smaller head
    "st_production": 16,  # Sparse target; smaller head
    "points_allowed": 48,  # Raw 0–55 scale needs capacity
    "yards_allowed": 48,  # Raw 0–600 scale needs capacity
}
DST_NN_DROPOUT = 0.30  # Slightly higher — slows convergence, better generalization
# All 5 heads are non-negative counts/magnitudes.
DST_NN_NON_NEGATIVE_TARGETS = {
    "defensive_production",
    "def_td_points",
    "st_production",
    "points_allowed",
    "yards_allowed",
}
DST_NN_LR = 3e-4  # Lower — more exploration before convergence
DST_NN_WEIGHT_DECAY = 3e-4
DST_NN_EPOCHS = 300
DST_NN_BATCH_SIZE = 128
DST_NN_PATIENCE = 35

# === Loss Weights ===
# Equal-weight 1.0 default across all 5 heads. Raw PA/YA have much larger
# magnitudes than the production heads, so callers may wish to down-weight
# them (e.g. 0.1) if the aux loss is dominated by those heads; ship 1.0
# and let the training run tune.
DST_LOSS_WEIGHTS = {
    "defensive_production": 1.0,
    "def_td_points": 1.0,
    "st_production": 1.0,
    "points_allowed": 1.0,
    "yards_allowed": 1.0,
}
DST_LOSS_W_TOTAL = 1.0

# === Huber Deltas (per-target) ===
# Production heads ~0-20 pts: delta=2.0; raw PA ~0-55: delta=5.0;
# raw YA ~0-600: delta=30.0.
DST_HUBER_DELTAS = {
    "defensive_production": 2.0,
    "def_td_points": 2.0,
    "st_production": 2.0,
    "points_allowed": 5.0,
    "yards_allowed": 30.0,
    "total": 3.0,  # explicit delta for total aux loss
}

# === LR Scheduler ===
DST_SCHEDULER_TYPE = "cosine_warm_restarts"
DST_COSINE_T0 = 30  # Longer first cycle for wider backbone
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


# ===========================================================================
# DST_CONFIG_TINY — shrunk config for E2E smoke tests.
# Used by DST/tests/test_dst_pipeline_e2e.py to exercise the full pipeline
# end-to-end in < 20s on tiny synthetic data: 2 backbone layers x 8 units,
# 1 epoch, no LightGBM, no attention.  Only the NN-training hyperparameters
# are shrunk; the rest of the config matches production to keep coverage
# representative.
# ===========================================================================
DST_CONFIG_TINY = {
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 1,
    "nn_head_hidden_overrides": None,
    "nn_non_negative_targets": DST_NN_NON_NEGATIVE_TARGETS,
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "train_lightgbm": False,
    "train_attention_nn": False,
    "ridge_pca_components": None,  # tiny synthetic data has few features
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "loss_w_total": 1.0,
    "nn_log_every": 100,  # suppress per-epoch prints during tests
}
