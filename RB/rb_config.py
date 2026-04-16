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
    "career_carries",
    "team_rb_carry_hhi_L3",
    "team_rb_target_hhi_L3",
    "opportunity_index_L3",
]

# === RB Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_RB_ROLLING_STATS = [
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "snap_pct",
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
    ] + [f"rolling_min_fantasy_points_L5"],
    "prior_season": [
        f"prior_season_{a}_{stat}"
        for stat in _RB_ROLLING_STATS
        for a in ["mean", "std", "max"]
    ],
    # All EWMA dropped (>0.98 corr with rolling means)
    "ewma": [],
    "trend": ["trend_fantasy_points", "trend_targets", "trend_carries", "trend_snap_pct"],
    "share": [
        "target_share_L3", "target_share_L5",
        "carry_share_L3", "carry_share_L5",
        "snap_pct", "air_yards_share",
    ],
    "matchup": [
        "opp_fantasy_pts_allowed_to_pos", "opp_rush_pts_allowed_to_pos",
        "opp_recv_pts_allowed_to_pos", "opp_def_rank_vs_pos",
    ],
    "defense": [
        "opp_def_sacks_L5", "opp_def_pass_yds_allowed_L5",
        "opp_def_pass_td_allowed_L5", "opp_def_ints_L5",
        "opp_def_rush_yds_allowed_L5", "opp_def_pts_allowed_L5",
    ],
    "contextual": [
        "is_home", "week", "is_returning_from_absence", "days_rest",
        "practice_status", "game_status", "depth_chart_rank",
    ],
    # implied_team + implied_opp encodes both game total and spread direction
    # without the perfect collinearity of keeping total_line alongside either.
    # is_dome: dome premium on receiving (r=0.023 receiving_floor).
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome", "rest_advantage"],
    "specific": RB_SPECIFIC_FEATURES,
}

# === Ridge ===
import numpy as np
RB_RIDGE_ALPHA_GRIDS = {
    "rushing_floor":   [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "receiving_floor": [round(x, 4) for x in np.logspace(-2, 2.5, 20)],
    "td_points":       [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# Two-stage model for td_points: zero-inflated (73.5% zeros), discrete (0,6,12,...).
# Hard-threshold classify-then-regress drops td_points MAE from 2.259 to 1.851.
RB_TWO_STAGE_TARGETS = {
    "td_points": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
}

# Ordinal classification for td_points: classes = {0,1,2,3+} TDs, predicts
# E[td_points] via class probabilities.  Monotonic cumulative probs.
RB_ORDINAL_TARGETS = {
    "td_points": {
        "type": "ordinal",
        "class_values": [0, 6, 12, 18],  # 0/1/2/3+ TDs * 6 pts each
        "alpha": 1.0,                      # mord LogisticAT regularization
    },
}

# Gated ordinal: binary gate (like two-stage) + ordinal on positives.
RB_GATED_ORDINAL_TARGETS = {
    "td_points": {
        "type": "gated_ordinal",
        "class_values": [0, 6, 12, 18],
        "alpha": 1.0,
        "clf_C": 0.001,
        "threshold": 0.5,
    },
}

# Which td_points model to use: "ridge" | "two_stage" | "ordinal" | "gated_ordinal"
RB_TD_MODEL_TYPE = "gated_ordinal"

# PCR: 80 components retains 99.8% variance, drops condition number from 1.8e8
# (after is_home removal) to 49.8.  Both floor targets improve by ~0.002 MAE.
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
    "td_points": 2.0,  # tightened from 3.0 — gated TD head handles zero-mass
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
RB_ATTN_ENCODER_HIDDEN_DIM = 0
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
    "game_carry_share", "game_target_share",
    "game_carry_hhi", "game_target_hhi",
]
# Two-stage gated TD head: sigmoid gate P(TD>0) × Softplus value E[TD|TD>0]
# Tuned: gate_weight 3.0 > 1.0 (stronger BCE signal), gate_hidden 24 > 16,
# Huber delta tightened 3.0 → 2.0 (gate handles zero-mass so value head needs less tolerance)
RB_ATTN_GATED_TD = True
RB_ATTN_TD_GATE_HIDDEN = 24
RB_ATTN_TD_GATE_WEIGHT = 3.0
