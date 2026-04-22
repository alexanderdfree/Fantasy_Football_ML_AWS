# === WR Target Decomposition ===
# Raw-stat targets (see shared/aggregate_targets.py). Fantasy points are
# aggregated from these predictions via predictions_to_fantasy_points.
# Rushing targets dropped - WR rushing stats are too sparse for reliable signal.
WR_TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

# === WR-Specific Features ===
WR_SPECIFIC_FEATURES = [
    "yards_per_reception_L3",
    "yards_per_target_L3",
    "reception_rate_L3",
    "air_yards_per_target_L3",
    "yac_per_reception_L3",
    "team_wr_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
]

# === WR Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_WR_ROLLING_STATS = [
    "fantasy_points",
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

WR_INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    # min variant only exists for fantasy_points (kept at all windows).
    "rolling": [
        col
        for stat in _WR_ROLLING_STATS
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
        f"prior_season_{a}_{stat}" for stat in _WR_ROLLING_STATS for a in ["mean", "std", "max"]
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
    "weather_vegas": [
        "implied_team_total",
        "implied_opp_total",
        "wind_adjusted",
        "is_dome",
        "temp_adjusted",
    ],
    "specific": WR_SPECIFIC_FEATURES,
}

# === Ridge ===
import numpy as np

# PCR: 30 components. Benchmark showed -0.094 MAE vs no-PCA baseline (4.507 → 4.413).
# PCA removes collinear directions the alpha grid can't fully address.
WR_RIDGE_PCA_COMPONENTS = 30
# Alpha grids sized to each target's dynamic range: yards use a wider high-alpha
# tail; count-style targets (TDs, receptions, fumbles) stay in the standard band.
WR_RIDGE_ALPHA_GRIDS = {
    "receiving_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "receiving_yards": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "receptions": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "fumbles_lost": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === Neural Net ===
# 2012+ dataset: widened from [96] to [128] to exploit largest training set.
# Largest position dataset can support more capacity with less overfitting risk.
WR_NN_BACKBONE_LAYERS = [128]
WR_NN_HEAD_HIDDEN = 32
# Larger head for zero-inflated receiving_tds target.
WR_NN_HEAD_HIDDEN_OVERRIDES = {"receiving_tds": 64}
WR_NN_DROPOUT = 0.20
WR_NN_LR = 1e-3
WR_NN_WEIGHT_DECAY = 1e-4
WR_NN_EPOCHS = 250
WR_NN_BATCH_SIZE = 512
WR_NN_PATIENCE = 25

# === Loss Weights ===
# Per-target weights scaled inversely to Huber delta (~2.0/δ) so every head
# contributes comparable gradient magnitude during joint training. Without
# rebalancing, receiving_yards (δ=15) dominated count heads (δ=0.5) ~900× per
# sample. Receptions anchors the scale at weight 1.0.
WR_LOSS_WEIGHTS = {
    "receiving_tds": 4.0,  # 2.0 / 0.5
    "receiving_yards": 0.133,  # 2.0 / 15
    "receptions": 1.0,  # 2.0 / 2.0 (anchor)
    "fumbles_lost": 4.0,
}

# === Huber Deltas (per-target, raw-stat units) ===
WR_HUBER_DELTAS = {
    "receiving_tds": 0.5,
    "receiving_yards": 15.0,
    "receptions": 2.0,
    "fumbles_lost": 0.5,
}

# === LR Scheduler ===
WR_SCHEDULER_TYPE = "cosine_warm_restarts"
WR_COSINE_T0 = 40
WR_COSINE_T_MULT = 2
WR_COSINE_ETA_MIN = 1e-5

# === Attention NN (game history variant) ===
WR_TRAIN_ATTENTION_NN = True
WR_ATTN_D_MODEL = 32
WR_ATTN_N_HEADS = 2
WR_ATTN_ENCODER_HIDDEN_DIM = 0
WR_ATTN_MAX_SEQ_LEN = 17
WR_ATTN_POSITIONAL_ENCODING = True
WR_ATTN_DROPOUT = 0.0
WR_ATTN_HISTORY_STATS = [
    "fantasy_points",
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
    "targets",
    "receptions",
    "carries",
    "snap_pct",
]
# Categories of WR_INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# WR_ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
WR_ATTN_STATIC_CATEGORIES = [
    "prior_season",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
]
WR_ATTN_STATIC_FEATURES = [c for cat in WR_ATTN_STATIC_CATEGORIES for c in WR_INCLUDE_FEATURES[cat]]
# Two-stage gated TD head: sigmoid gate P(TD>0) × Softplus value E[TD|TD>0]
# Single gate on receiving_tds — the only WR TD target after rushing drop.
WR_ATTN_GATED_TD = True
WR_GATED_TD_TARGETS = ["receiving_tds"]
WR_ATTN_TD_GATE_HIDDEN = 16
WR_ATTN_TD_GATE_WEIGHT = 1.0

# === LightGBM (Optuna-tuned, 50 trials, CV MAE 4.7319) ===
WR_TRAIN_LIGHTGBM = True
WR_LGBM_N_ESTIMATORS = 900
WR_LGBM_LEARNING_RATE = 0.0183694
WR_LGBM_NUM_LEAVES = 58
WR_LGBM_SUBSAMPLE = 0.588592
WR_LGBM_COLSAMPLE_BYTREE = 0.401101
WR_LGBM_REG_LAMBDA = 9.57554
WR_LGBM_REG_ALPHA = 0.674656
WR_LGBM_MIN_CHILD_SAMPLES = 37
WR_LGBM_MIN_SPLIT_GAIN = 0.370048
WR_LGBM_OBJECTIVE = "fair"


# === Tiny-scale config for E2E smoke tests ===
# Shrunk copy of the production hyperparameters: 1 epoch, 2-layer NN with 8
# units, no attention, no LightGBM. Keeps the full-pipeline E2E test under
# ~20s while still exercising every stage of run_pipeline().
WR_CONFIG_TINY = {
    "targets": WR_TARGETS,
    "specific_features": WR_SPECIFIC_FEATURES,
    "ridge_alpha_grids": {t: [1.0] for t in WR_TARGETS},
    "ridge_pca_components": None,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 64,
    "nn_patience": 1,
    "nn_log_every": 1,
    "loss_weights": WR_LOSS_WEIGHTS,
    "huber_deltas": WR_HUBER_DELTAS,
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "train_attention_nn": False,
    "train_lightgbm": False,
}
