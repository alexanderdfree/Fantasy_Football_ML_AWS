# === QB Target Decomposition ===
QB_TARGETS = ["passing_floor", "rushing_floor", "td_points"]

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
    "fantasy_points",
    "fantasy_points_floor",
    "carries",
    "rushing_yards",
    "passing_yards",
    "attempts",
    "snap_pct",
]

QB_INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # min variant only exists for fantasy_points (kept at all windows).
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    "rolling": [
        col
        for stat in _QB_ROLLING_STATS
        for w in [3, 5, 8]
        for col in (
            (
                [f"rolling_{a}_{stat}_L{w}" for a in ["mean", "std", "max"]]
                if w != 5 or stat == "snap_pct"
                else []
            )
            + ([f"rolling_min_{stat}_L{w}"] if stat == "fantasy_points" else [])
        )
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}" for stat in _QB_ROLLING_STATS for a in ["mean", "std", "max"]
    ],
    # Keep passing_yards EWMA only — other EWMA >0.98 corr with rolling means
    "ewma": ["ewma_passing_yards_L3", "ewma_passing_yards_L5"],
    "trend": ["trend_fantasy_points", "trend_carries", "trend_snap_pct"],
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
import numpy as np

QB_RIDGE_ALPHA_GRIDS = {
    "passing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "td_points": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net (2012+ dataset: wider backbone, relaxed regularization) ===
QB_NN_BACKBONE_LAYERS = [128]
QB_NN_HEAD_HIDDEN = 32
QB_NN_DROPOUT = 0.20
QB_NN_LR = 5e-4
QB_NN_WEIGHT_DECAY = 3e-4
QB_NN_EPOCHS = 300
QB_NN_BATCH_SIZE = 128
QB_NN_PATIENCE = 25

# === Loss Weights ===
# Equal per-target weights: training objective now aligned with evaluation
# metric (total MAE), where all targets contribute equally to the total.
# Previous scheme (1.5/0.8/3.0) over-optimized td_points at the expense
# of floor predictions. w_total raised to 1.0 so total prediction quality
# gets equal gradient signal alongside per-target losses.
QB_LOSS_WEIGHTS = {
    "passing_floor": 1.0,
    "rushing_floor": 1.0,
    "td_points": 1.0,
}
QB_LOSS_W_TOTAL = 1.0

# === Huber Deltas (per-target) ===
# Harmonized to 2.0 across targets so equal-magnitude errors get equal
# treatment regardless of which target they come from. Previous scheme
# (1.5/1.0/3.0) created uneven gradient landscapes.
QB_HUBER_DELTAS = {
    "passing_floor": 2.0,
    "rushing_floor": 2.0,
    "td_points": 2.0,
    "total": 4.0,  # explicit delta for total aux loss (QBs score highest)
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
    "fantasy_points",
    "fantasy_points_floor",
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
# Two-stage gated TD head: sigmoid gate P(TD>0) × Softplus value E[TD|TD>0]
QB_ATTN_GATED_TD = True
QB_ATTN_TD_GATE_HIDDEN = 16
QB_ATTN_TD_GATE_WEIGHT = 1.0

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
