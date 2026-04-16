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

# === TE Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_TE_ROLLING_STATS = [
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "snap_pct",
]

TE_INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    # min variant only exists for fantasy_points (kept at all windows).
    "rolling": [
        col
        for stat in _TE_ROLLING_STATS
        for w in [3, 5, 8]
        for col in (
            ([f"rolling_{a}_{stat}_L{w}" for a in ["mean", "std", "max"]]
             if w != 5 or stat == "snap_pct" else [])
            + ([f"rolling_min_{stat}_L{w}"] if stat == "fantasy_points" else [])
        )
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}"
        for stat in _TE_ROLLING_STATS
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
    # TE keeps is_home (unlike RB/WR where it's zero-variance)
    "contextual": [
        "is_home", "week", "is_returning_from_absence", "days_rest",
        "practice_status", "game_status", "depth_chart_rank",
    ],
    # Keep 3 features with signal:
    #   implied_team_total (r=-0.035), implied_opp_total (r=0.029), is_dome (r=0.027)
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome"],
    "specific": TE_SPECIFIC_FEATURES,
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
# Two-stage gated TD head: sigmoid gate P(TD>0) × Softplus value E[TD|TD>0]
TE_ATTN_GATED_TD = True
TE_ATTN_TD_GATE_HIDDEN = 16
TE_ATTN_TD_GATE_WEIGHT = 1.0

# === LightGBM (Optuna-tuned, 50 trials, CV MAE 3.6091) ===
TE_TRAIN_LIGHTGBM = True
TE_LGBM_N_ESTIMATORS = 1300
TE_LGBM_LEARNING_RATE = 0.0900384
TE_LGBM_NUM_LEAVES = 29
TE_LGBM_SUBSAMPLE = 0.854792
TE_LGBM_COLSAMPLE_BYTREE = 0.469154
TE_LGBM_REG_LAMBDA = 0.011196
TE_LGBM_REG_ALPHA = 0.0129827
TE_LGBM_MIN_CHILD_SAMPLES = 80
TE_LGBM_MIN_SPLIT_GAIN = 0.267235
TE_LGBM_OBJECTIVE = "fair"
