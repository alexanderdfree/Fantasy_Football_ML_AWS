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
    "fantasy_points", "fantasy_points_floor", "carries",
    "rushing_yards", "passing_yards", "attempts", "snap_pct",
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
            ([f"rolling_{a}_{stat}_L{w}" for a in ["mean", "std", "max"]]
             if w != 5 or stat == "snap_pct" else [])
            + ([f"rolling_min_{stat}_L{w}"] if stat == "fantasy_points" else [])
        )
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}"
        for stat in _QB_ROLLING_STATS
        for a in ["mean", "std", "max"]
    ],
    # Keep passing_yards EWMA only — other EWMA >0.98 corr with rolling means
    "ewma": ["ewma_passing_yards_L3", "ewma_passing_yards_L5"],
    "trend": ["trend_fantasy_points", "trend_carries", "trend_snap_pct"],
    # No target_share/air_yards_share — QBs have ~0 targets
    "share": ["carry_share_L3", "carry_share_L5", "snap_pct"],
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
    # Keep 6 weather/Vegas features with |r|>0.04 or |r|>0.015+MI>0.005
    "weather_vegas": [
        "implied_opp_total", "wind_adjusted", "is_divisional",
        "implied_total_x_dome", "temp_adjusted", "is_dome",
    ],
    "specific": QB_SPECIFIC_FEATURES,
}

# === Ridge ===
import numpy as np
QB_RIDGE_ALPHA_GRIDS = {
    "passing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "rushing_floor": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "td_points":     [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net (2012+ dataset: wider backbone, relaxed regularization) ===
QB_NN_BACKBONE_LAYERS = [96, 48]
QB_NN_HEAD_HIDDEN = 20
QB_NN_DROPOUT = 0.35
QB_NN_LR = 5e-4
QB_NN_WEIGHT_DECAY = 3e-4
QB_NN_EPOCHS = 300
QB_NN_BATCH_SIZE = 128
QB_NN_PATIENCE = 25

# === Loss Weights ===
# Passing floor is the primary QB scoring driver; boost its weight.
# TD weight elevated for discrete/zero-inflated nature.
QB_LOSS_WEIGHTS = {
    "passing_floor": 1.5,
    "rushing_floor": 0.8,
    "td_points": 3.0,
}
QB_LOSS_W_TOTAL = 0.3

# === Huber Deltas (per-target) ===
QB_HUBER_DELTAS = {
    "passing_floor": 1.5,
    "rushing_floor": 1.0,
    "td_points": 3.0,
}

# === LR Scheduler ===
QB_SCHEDULER_TYPE = "onecycle"
QB_ONECYCLE_MAX_LR = 2e-3
QB_ONECYCLE_PCT_START = 0.3

# === Attention NN (game history variant) ===
QB_TRAIN_ATTENTION_NN = True
QB_ATTN_D_MODEL = 32        # projection dim for each game vector
QB_ATTN_N_HEADS = 2
QB_ATTN_ENCODER_HIDDEN_DIM = 0
QB_ATTN_MAX_SEQ_LEN = 17
QB_ATTN_POSITIONAL_ENCODING = True
QB_ATTN_DROPOUT = 0.05
QB_ATTN_PATIENCE = 25
QB_ATTN_HISTORY_STATS = [
    "fantasy_points", "fantasy_points_floor",
    "passing_yards", "rushing_yards",
    "passing_tds", "rushing_tds",
    "attempts", "completions", "carries",
    "interceptions", "snap_pct",
    "sacks", "sack_yards",
]
# Two-stage gated TD head: sigmoid gate P(TD>0) × Softplus value E[TD|TD>0]
QB_ATTN_GATED_TD = True
QB_ATTN_TD_GATE_HIDDEN = 16
QB_ATTN_TD_GATE_WEIGHT = 1.0
