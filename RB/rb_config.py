import numpy as np

# === RB Raw-Stat Targets ===
RB_TARGETS = [
    "rushing_tds",
    "receiving_tds",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fumbles_lost",
]

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
    "career_carries",
    "team_rb_carry_hhi_L3",
    "team_rb_target_hhi_L3",
    "opportunity_index_L3",
]

# === RB Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_RB_ROLLING_STATS = [
    "fantasy_points",
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

RB_INCLUDE_FEATURES = {
    # L3/L8 only — all L5 dropped (>0.97 corr with L3/L8).
    # min variant only exists for fantasy_points (kept at all windows).
    "rolling": [
        col
        for stat in _RB_ROLLING_STATS
        for w in [3, 8]
        for col in (
            [f"rolling_{a}_{stat}_L{w}" for a in ["mean", "std", "max"]]
            + ([f"rolling_min_{stat}_L{w}"] if stat == "fantasy_points" else [])
        )
    ]
    + ["rolling_min_fantasy_points_L5"],
    "prior_season": [
        f"prior_season_{a}_{stat}" for stat in _RB_ROLLING_STATS for a in ["mean", "std", "max"]
    ],
    # All EWMA dropped (>0.98 corr with rolling means)
    "ewma": [],
    "trend": ["trend_fantasy_points", "trend_targets", "trend_carries", "trend_snap_pct"],
    "share": [
        "target_share_L3",
        "target_share_L5",
        "carry_share_L3",
        "carry_share_L5",
        "snap_pct",
        "air_yards_share",
    ],
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
    # implied_team + implied_opp encodes both game total and spread direction
    # without the perfect collinearity of keeping total_line alongside either.
    # is_dome: dome premium on receiving (r=0.023 receiving_floor).
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome", "rest_advantage"],
    "specific": RB_SPECIFIC_FEATURES,
}

# === Ridge ===
# Raw-stat grids — yards need broader high end (large variance vs counts).
RB_RIDGE_ALPHA_GRIDS = {
    "rushing_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "receiving_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "rushing_yards": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "receiving_yards": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "receptions": [round(x, 4) for x in np.logspace(-2, 2.5, 20)],
    "fumbles_lost": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# Two-stage zero-inflated models: both rushing_tds and receiving_tds.
# Threshold + hyperparams preserved from the pre-migration td_points config;
# rebuilding per-TD afterwards gives two parallel classify-then-regress stacks.
RB_TWO_STAGE_TARGETS = {
    "rushing_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
    "receiving_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
}

# Ordinal classification over raw TD counts {0,1,2,3+} per TD target.
RB_ORDINAL_TARGETS = {
    "rushing_tds": {
        "type": "ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
    },
    "receiving_tds": {
        "type": "ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
    },
}

# Gated ordinal: binary gate + ordinal on positives, per TD target.
RB_GATED_ORDINAL_TARGETS = {
    "rushing_tds": {
        "type": "gated_ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
        "clf_C": 0.001,
        "threshold": 0.5,
    },
    "receiving_tds": {
        "type": "gated_ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
        "clf_C": 0.001,
        "threshold": 0.5,
    },
}

# Which TD model variant to use: "ridge" | "two_stage" | "ordinal" | "gated_ordinal"
RB_TD_MODEL_TYPE = "gated_ordinal"

# PCR: 80 components retains 99.8% variance, drops condition number from 1.8e8
# (after is_home removal) to 49.8.  Both yard targets improve by ~0.002 MAE.
RB_RIDGE_PCA_COMPONENTS = 80

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
# Larger head for the hurdle-NegBin reception head (two value outputs: mu +
# log_alpha). TD heads moved to plain Poisson NLL (dispersion ~1.03-1.17, no
# zero-excess) and no longer need the extra capacity the Huber+gate setup did.
RB_NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 64}

# === Per-Head Loss Families ===
# TDs + fumbles: plain Poisson NLL. Empirical dispersion 1.03-1.17 with
# negligible zero-excess — plain Poisson fits; the old BCE gate on (TD>0) was
# unmotivated and comes off here.
# Receptions: zero-truncated NegBin-2 hurdle. Variance/mean ~2.0 (overdispersed)
# with zero-excess up to +0.13 — textbook hurdle fit. Gate BCE is added via
# RB_GATED_TARGETS below; the ZTNB NLL trains on positive samples only, scaled
# by fraction-positive inside the batch.
RB_HEAD_LOSSES = {
    "rushing_tds": "poisson_nll",
    "receiving_tds": "poisson_nll",
    "rushing_yards": "huber",
    "receiving_yards": "huber",
    "receptions": "hurdle_negbin",
    "fumbles_lost": "poisson_nll",
}

# === Loss Weights ===
# Yards heads: keep the 2.0/delta rebalance that stops yards gradients from
# dominating (without this, fantasy-point MAE regressed 4.23 -> 5.21; see the
# pre-PR-1 archive entry).
# Poisson NLL heads: picked so the expected weighted per-sample loss sits near
# the Huber contributions. At mean-TD-rate ~0.3, Poisson NLL ~ O(0.5); weight
# 1.0 keeps the contribution in the same 0.5-1.0 band as weighted yards Huber.
# hurdle_negbin reception head: weight 1.0. Value loss is already scaled by
# fraction-positive inside hurdle_negbin_value_loss, so no further rescaling.
RB_LOSS_WEIGHTS = {
    "rushing_tds": 1.0,  # Poisson NLL
    "receiving_tds": 1.0,  # Poisson NLL
    "rushing_yards": 0.133,  # 2.0 / 15  (Huber)
    "receiving_yards": 0.133,
    "receptions": 1.0,  # hurdle_negbin, fraction-scaled internally
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (per-target, raw-stat units) ===
# Only Huber heads need a delta. TD / fumble / reception heads use Poisson /
# hurdle-NegBin, which don't consume a delta.
RB_HUBER_DELTAS = {
    "rushing_yards": 15.0,
    "receiving_yards": 15.0,
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
# 2-layer nonlinear game encoder (Linear→ReLU→LayerNorm→Linear→ReLU) so each
# game is represented as a richer event embedding before attention, instead of
# a near-linear projection of raw stats.
RB_ATTN_ENCODER_HIDDEN_DIM = 32
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
    "fantasy_points",
    "rushing_yards",
    "receiving_yards",
    "rushing_tds",
    "receiving_tds",
    "carries",
    "targets",
    "receptions",
    "snap_pct",
    "rushing_first_downs",
    "receiving_first_downs",
    "game_carry_share",
    "game_target_share",
    "game_carry_hhi",
    "game_target_hhi",
]
# Categories of RB_INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# RB_ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
RB_ATTN_STATIC_CATEGORIES = [
    "prior_season",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
]
RB_ATTN_STATIC_FEATURES = [c for cat in RB_ATTN_STATIC_CATEGORIES for c in RB_INCLUDE_FEATURES[cat]]
# Single hurdle gate on receptions (variance/mean ~2.0, zero-excess ~+0.13).
# TD heads dropped from the gated list: dispersion ~1.0 and ~0 zero-excess make
# the gate unmotivated on counts (kept behind an ablation script —
# scripts/ablate_rb_gate.py — so the call is verifiable).
RB_ATTN_GATED = True
RB_GATED_TARGETS = ["receptions"]
RB_ATTN_GATE_HIDDEN = 16
RB_ATTN_GATE_WEIGHT = 1.0

# === LightGBM (Optuna-tuned, 50 trials, CV MAE 4.5149) ===
RB_TRAIN_LIGHTGBM = True
RB_LGBM_N_ESTIMATORS = 1400
RB_LGBM_LEARNING_RATE = 0.0704275
RB_LGBM_NUM_LEAVES = 54
RB_LGBM_MAX_DEPTH = 9
RB_LGBM_SUBSAMPLE = 0.830769
RB_LGBM_COLSAMPLE_BYTREE = 0.401929
RB_LGBM_REG_LAMBDA = 0.0371884
RB_LGBM_REG_ALPHA = 0.718914
RB_LGBM_MIN_CHILD_SAMPLES = 69
RB_LGBM_MIN_SPLIT_GAIN = 0.302108
RB_LGBM_OBJECTIVE = "fair"

# === Tiny config for end-to-end smoke tests ===
# Shrunk copy of the production config: 1 epoch, 2-layer x 8-unit backbone,
# attention and LightGBM disabled to keep the E2E smoke under 20s.
# Keeps every behavior toggle identical to RB_CONFIG so test coverage
# exercises the same code paths.
RB_NN_BACKBONE_LAYERS_TINY = [8, 8]
RB_NN_HEAD_HIDDEN_TINY = 4
RB_NN_EPOCHS_TINY = 1
RB_NN_BATCH_SIZE_TINY = 64
RB_NN_PATIENCE_TINY = 1

# Tiny configs for test fixtures: single-alpha grids and a flattened loss
# weight map keyed to the new targets.
RB_CONFIG_TINY = {
    "targets": RB_TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in RB_TARGETS},
    "loss_weights": {t: 1.0 for t in RB_TARGETS},
    "huber_deltas": RB_HUBER_DELTAS,
    "nn_backbone_layers": RB_NN_BACKBONE_LAYERS_TINY,
    "nn_head_hidden": RB_NN_HEAD_HIDDEN_TINY,
    "nn_epochs": RB_NN_EPOCHS_TINY,
    "nn_batch_size": RB_NN_BATCH_SIZE_TINY,
    "nn_patience": RB_NN_PATIENCE_TINY,
}
