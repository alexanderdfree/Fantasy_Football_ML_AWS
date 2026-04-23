# === QB Raw-Stat Targets ===
# Predictions are raw NFL stats; fantasy points are aggregated post-prediction
# via shared.aggregate_targets.predictions_to_fantasy_points("QB", preds).
QB_TARGETS = [
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
]

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

# === QB Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_QB_ROLLING_STATS = [
    "carries",
    "rushing_yards",
    "passing_yards",
    "attempts",
    "snap_pct",
]

QB_INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    "rolling": [
        f"rolling_{a}_{stat}_L{w}"
        for stat in _QB_ROLLING_STATS
        for w in [3, 5, 8]
        for a in ["mean", "std", "max"]
        if w != 5 or stat == "snap_pct"
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}" for stat in _QB_ROLLING_STATS for a in ["mean", "std", "max"]
    ],
    # Keep passing_yards EWMA only — other EWMA >0.98 corr with rolling means
    "ewma": ["ewma_passing_yards_L3", "ewma_passing_yards_L5"],
    "trend": ["trend_carries", "trend_snap_pct"],
    # No target_share/air_yards_share — QBs have ~0 targets
    "share": ["carry_share_L3", "carry_share_L5", "snap_pct"],
    "matchup": [
        "opp_fantasy_pts_allowed_to_pos",
        "opp_rush_pts_allowed_to_pos",
        "opp_recv_pts_allowed_to_pos",
        "opp_def_rank_vs_pos",
    ],
    "defense": [
        "opp_def_sacks_L5",
        "opp_def_pass_yds_allowed_L5",
        "opp_def_pass_td_allowed_L5",
        "opp_def_ints_L5",
        "opp_def_rush_yds_allowed_L5",
        "opp_def_pts_allowed_L5",
    ],
    "contextual": [
        "is_home",
        "week",
        "is_returning_from_absence",
        "days_rest",
        "practice_status",
        "game_status",
        "depth_chart_rank",
    ],
    "weather_vegas": [
        "implied_team_total",
        "implied_opp_total",
        "wind_adjusted",
        "is_dome",
        "is_divisional",
        "temp_adjusted",
    ],
    "specific": QB_SPECIFIC_FEATURES,
}

# === Ridge ===
# Yards targets stay on the (-2, 3) alpha grid; count targets (TDs, INTs,
# fumbles_lost) use the (-1, 4) grid because the smaller target scale lets
# stronger regularization dominate.
import numpy as np

QB_RIDGE_ALPHA_GRIDS = {
    "passing_yards": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_yards": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "passing_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "rushing_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "interceptions": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "fumbles_lost": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. When enabled, the pipeline reuses QB_RIDGE_ALPHA_GRIDS and
# searches over QB_ENET_L1_RATIOS. Skips PCA — L1 on a rotated basis doesn't
# zero original features, so PCA defeats the reason to pick ElasticNet.
QB_TRAIN_ELASTICNET = False
QB_ENET_L1_RATIOS = [0.3, 0.5, 0.7]

# === Neural Net (2012+ dataset: wider backbone, relaxed regularization) ===
QB_NN_BACKBONE_LAYERS = [128]
QB_NN_HEAD_HIDDEN = 32
QB_NN_DROPOUT = 0.20
QB_NN_LR = 5e-4
QB_NN_WEIGHT_DECAY = 3e-4
QB_NN_EPOCHS = 300
QB_NN_BATCH_SIZE = 128
QB_NN_PATIENCE = 25

# Count heads (passing/rushing TDs) moved from Huber to Poisson NLL — dispersion
# 1.03-1.17 with negligible zero-excess, so plain Poisson fits; the extra
# 64-unit capacity the Huber setup needed to fight the zero-mass is unnecessary.
QB_NN_HEAD_HIDDEN_OVERRIDES = {}

# === Per-Head Loss Families ===
# TDs + INTs + fumbles: Poisson NLL. QB TD distributions are not zero-inflated
# (median ~2 TDs/start), so a plain Poisson rate model is the right fit and the
# Huber-count-head-collapse failure mode (count heads regressing to the mean
# under yards-dominated gradients) is avoided without needing wider heads.
# No gated targets on QB (see QB_ATTN_GATED below).
QB_HEAD_LOSSES = {
    "passing_yards": "huber",
    "rushing_yards": "huber",
    "passing_tds": "poisson_nll",
    "rushing_tds": "poisson_nll",
    "interceptions": "poisson_nll",
    "fumbles_lost": "poisson_nll",
}

# === Loss Weights ===
# Yards heads keep 2.0/delta rebalance so count-head gradients aren't drowned
# out (pre-rebalance: fantasy-point MAE regressed 6.33 -> 6.63; fumbles_lost
# R2 = -0.34). Poisson NLL heads use weight 1.0 — at QB-scale rates (~1.5 TDs,
# ~0.7 INTs, ~0.4 fumbles) the Poisson NLL is O(1), matching weighted yards.
QB_LOSS_WEIGHTS = {
    "passing_yards": 0.08,  # 2.0 / 25  (Huber)
    "rushing_yards": 0.133,  # 2.0 / 15  (Huber)
    "passing_tds": 1.0,  # Poisson NLL
    "rushing_tds": 1.0,  # Poisson NLL
    "interceptions": 1.0,  # Poisson NLL
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (raw-stat units) ===
# Only Huber heads need a delta — count heads moved to Poisson NLL.
QB_HUBER_DELTAS = {
    "passing_yards": 25.0,
    "rushing_yards": 15.0,
}

# === LR Scheduler ===
QB_SCHEDULER_TYPE = "cosine_warm_restarts"
QB_COSINE_T0 = 40
QB_COSINE_T_MULT = 2
QB_COSINE_ETA_MIN = 1e-5

# === Attention NN (game history variant) ===
QB_TRAIN_ATTENTION_NN = True
QB_ATTN_D_MODEL = 32  # projection dim for each game vector
QB_ATTN_N_HEADS = 2
QB_ATTN_ENCODER_HIDDEN_DIM = 0
QB_ATTN_MAX_SEQ_LEN = 17
QB_ATTN_POSITIONAL_ENCODING = True
QB_ATTN_DROPOUT = 0.05
QB_ATTN_PATIENCE = 35
QB_ATTN_LR = 1e-3
QB_ATTN_WEIGHT_DECAY = 5e-5
QB_ATTN_BATCH_SIZE = 256
QB_ATTN_HISTORY_STATS = [
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "attempts",
    "completions",
    "carries",
    "interceptions",
    "snap_pct",
    "sacks",
    "sack_yards",
]
# Categories of QB_INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# QB_ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
QB_ATTN_STATIC_CATEGORIES = [
    "prior_season",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
]
QB_ATTN_STATIC_FEATURES = [c for cat in QB_ATTN_STATIC_CATEGORIES for c in QB_INCLUDE_FEATURES[cat]]
# Gated hurdle heads are DISABLED for QB. QBs throw so many TDs that the zero-
# inflation assumption behind the hurdle model does not hold (median TD count
# per start is ~2); a plain regression head outperforms the two-stage gate.
QB_ATTN_GATED = False
QB_ATTN_GATE_HIDDEN = 16
QB_ATTN_GATE_WEIGHT = 1.0

# === LightGBM (Optuna-tuned, 50 trials, CV MAE 5.7415) ===
QB_TRAIN_LIGHTGBM = True
QB_LGBM_N_ESTIMATORS = 1500
QB_LGBM_LEARNING_RATE = 0.0612763
QB_LGBM_NUM_LEAVES = 31
QB_LGBM_SUBSAMPLE = 0.867443
QB_LGBM_COLSAMPLE_BYTREE = 0.907776
QB_LGBM_REG_LAMBDA = 6.75023
QB_LGBM_REG_ALPHA = 0.00309259
QB_LGBM_MIN_CHILD_SAMPLES = 59
QB_LGBM_MIN_SPLIT_GAIN = 0.0632242
QB_LGBM_OBJECTIVE = "fair"
