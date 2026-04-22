import numpy as np

# === DST Raw-Stat Targets ===
# Predict the 10 raw NFL stats that make up a D/ST's fantasy-point score.
# Fantasy points are computed post-prediction via
# ``shared.aggregate_targets.predictions_to_fantasy_points("DST", preds)``,
# which applies the linear coefficients (sacks×1, INT×2, ...) and the PA/YA
# tier bonuses in a single place.
DST_TARGETS = [
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
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

DST_RIDGE_PCA_COMPONENTS = 20  # features → 20 components; removes collinear dimensions

# Alpha grids tuned per-target magnitude:
#   Sparse counts (TDs, safeties, blocked kicks, ST TDs) need a higher-floor grid
#   because their means are <0.1 — weak L2 produces unstable fits.
#   Regular counts (sacks, ints, fum_rec, FF) use a standard grid.
#   Raw PA/YA span 0-55 / 0-600 and tolerate stronger L2.
_DST_ALPHA_SPARSE = [round(x, 4) for x in np.logspace(0, 4, 15)]
_DST_ALPHA_STANDARD = [round(x, 4) for x in np.logspace(-1, 3.5, 20)]
_DST_ALPHA_RAW_SCALE = [round(x, 4) for x in np.logspace(-1, 5, 20)]

DST_RIDGE_ALPHA_GRIDS = {
    "def_sacks": _DST_ALPHA_STANDARD,
    "def_ints": _DST_ALPHA_STANDARD,
    "def_fumble_rec": _DST_ALPHA_STANDARD,
    "def_fumbles_forced": _DST_ALPHA_STANDARD,
    "def_safeties": _DST_ALPHA_SPARSE,
    "def_tds": _DST_ALPHA_SPARSE,
    "def_blocked_kicks": _DST_ALPHA_SPARSE,
    "special_teams_tds": _DST_ALPHA_SPARSE,
    "points_allowed": _DST_ALPHA_RAW_SCALE,
    "yards_allowed": _DST_ALPHA_RAW_SCALE,
}

# === Neural Net ===
DST_NN_BACKBONE_LAYERS = [128, 64]  # Wider backbone for many features
DST_NN_HEAD_HIDDEN = 32
DST_NN_HEAD_HIDDEN_OVERRIDES = {
    # Sparse targets — smaller heads regularize, rare events don't need capacity
    "def_safeties": 16,
    "def_tds": 16,
    "def_blocked_kicks": 16,
    "special_teams_tds": 16,
    # Raw-scale targets — wider head to learn the 0-55 / 0-600 magnitude
    "points_allowed": 48,
    "yards_allowed": 48,
}
DST_NN_DROPOUT = 0.30  # Slightly higher — slows convergence, better generalization
# All 10 raw-count heads are non-negative.
DST_NN_NON_NEGATIVE_TARGETS = set(DST_TARGETS)
DST_NN_LR = 3e-4  # Lower — more exploration before convergence
DST_NN_WEIGHT_DECAY = 3e-4
DST_NN_EPOCHS = 300
DST_NN_BATCH_SIZE = 128
DST_NN_PATIENCE = 35

# === Loss Weights ===
# Scaled inversely to each target's Huber delta per the CLAUDE.md rule
# (≈ 2.0/δ) so every head contributes comparable gradient magnitude. Without
# this, PA (δ=5) and YA (δ=30) would dominate the sparse count heads (δ=0.25)
# by 20-120× per sample, collapsing the count heads to the mean.
DST_LOSS_WEIGHTS = {
    "def_sacks": 2.0,  # 2.0 / 1.0
    "def_ints": 4.0,  # 2.0 / 0.5
    "def_fumble_rec": 4.0,
    "def_fumbles_forced": 4.0,
    "def_safeties": 8.0,  # 2.0 / 0.25
    "def_tds": 8.0,
    "def_blocked_kicks": 8.0,
    "special_teams_tds": 8.0,
    "points_allowed": 0.4,  # 2.0 / 5.0
    "yards_allowed": 0.067,  # 2.0 / 30.0
}

# === Huber Deltas (per-target, raw-stat units) ===
# Deltas roughly match each target's typical variance so outliers are robust.
DST_HUBER_DELTAS = {
    "def_sacks": 1.0,
    "def_ints": 0.5,
    "def_fumble_rec": 0.5,
    "def_fumbles_forced": 0.5,
    "def_safeties": 0.25,
    "def_tds": 0.25,
    "def_blocked_kicks": 0.25,
    "special_teams_tds": 0.25,
    "points_allowed": 5.0,
    "yards_allowed": 30.0,
}

# === LR Scheduler ===
DST_SCHEDULER_TYPE = "cosine_warm_restarts"
DST_COSINE_T0 = 30  # Longer first cycle for wider backbone
DST_COSINE_T_MULT = 2
DST_COSINE_ETA_MIN = 1e-5

# === Attention NN (game history variant) ===
# Copies RB's attention hyperparameter shape (the most advanced position model);
# no gating per design (no GATED_FUSION, no hurdle gate). The attention branch
# learns its own temporal representation from per-game defensive + opponent
# history, so rolling/EWMA/trend features are stripped from its static input.
DST_TRAIN_ATTENTION_NN = True
DST_ATTN_D_MODEL = 32
DST_ATTN_N_HEADS = 2
DST_ATTN_ENCODER_HIDDEN_DIM = 32  # 2-layer game encoder before attention
DST_ATTN_MAX_SEQ_LEN = 17  # Full NFL regular season
DST_ATTN_PROJECT_KV = False
DST_ATTN_POSITIONAL_ENCODING = True
DST_ATTN_GATED_FUSION = False
DST_ATTN_GATED = False
DST_ATTN_GATE_HIDDEN = 16
DST_ATTN_GATE_WEIGHT = 1.0

# Per-head loss family. Default "huber"; PR 2 introduces "poisson_nll" and
# "hurdle_negbin" options. All heads on "huber" here = no behavior change.
DST_HEAD_LOSSES = {t: "huber" for t in DST_TARGETS}

DST_ATTN_DROPOUT = 0.05
DST_ATTN_LR = DST_NN_LR
DST_ATTN_WEIGHT_DECAY = DST_NN_WEIGHT_DECAY
DST_ATTN_BATCH_SIZE = DST_NN_BATCH_SIZE
DST_ATTN_PATIENCE = 35
# Per-game stats fed into the attention sequence. The 10 raw target stats
# plus 4 opponent-side per-game values (not rolling) so attention can weigh
# recent games against recent opponent strength.
# Derived combos (defensive_production, st_production, fantasy_points) are
# intentionally excluded — they're linear functions of the raw stats already
# in the sequence and would add collinear columns.
DST_ATTN_HISTORY_STATS = [
    # Own raw defensive + ST stats (mirror the 10 targets)
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
    "points_allowed",
    "yards_allowed",
    # Per-game opponent context (not pre-rolled)
    "opp_scoring",
    "opp_fumbles",
    "opp_interceptions",
    "opp_qb_epa",
]
# Explicit whitelist of static-branch features for the attention NN. DST
# doesn't use the category-dict shape that QB/RB/WR/TE do, so we enumerate
# the allowed columns directly. All DST_SPECIFIC_FEATURES (rolling/ewma/
# trend/std) and the opp_*_L{3,5} columns are excluded — the attention
# branch already sees that signal via DST_ATTN_HISTORY_STATS. Prior-season
# means stay (different season than the lookback window).
DST_ATTN_STATIC_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "rest_days",
    "div_game",
    "is_dome",
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

# === LightGBM ===
DST_TRAIN_LIGHTGBM = True
DST_LGBM_N_ESTIMATORS = 300
DST_LGBM_LEARNING_RATE = 0.03
DST_LGBM_NUM_LEAVES = 15
DST_LGBM_MAX_DEPTH = -1
DST_LGBM_SUBSAMPLE = 0.75
DST_LGBM_COLSAMPLE_BYTREE = 0.8
DST_LGBM_REG_LAMBDA = 2.0
DST_LGBM_REG_ALPHA = 0.1
DST_LGBM_MIN_CHILD_SAMPLES = 25
DST_LGBM_MIN_SPLIT_GAIN = 0.0
DST_LGBM_OBJECTIVE = "huber"


# ===========================================================================
# DST_CONFIG_TINY — shrunk config for E2E smoke tests.
# Used by DST/tests/test_dst_pipeline_e2e.py to exercise the full pipeline
# end-to-end in < 20s on tiny synthetic data: 2 backbone layers x 8 units,
# 1 epoch, no LightGBM, attention off by default.  Only the NN-training
# hyperparameters are shrunk; the rest of the config matches production to
# keep coverage representative.  Tests that need to exercise the attention
# path override ``train_attention_nn`` to True.
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
    "nn_log_every": 100,  # suppress per-epoch prints during tests
}
