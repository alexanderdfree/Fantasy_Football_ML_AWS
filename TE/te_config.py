# === TE Target Decomposition ===
# Raw-stat targets; fantasy points are aggregated post-prediction via
# shared.aggregate_targets.predictions_to_fantasy_points("TE", ...).
# Rushing targets dropped — TE rushing stats are near-zero (noise > signal).
TE_TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

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
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

TE_INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    "rolling": [
        f"rolling_{a}_{stat}_L{w}"
        for stat in _TE_ROLLING_STATS
        for w in [3, 5, 8]
        for a in ["mean", "std", "max"]
        if w != 5 or stat == "snap_pct"
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}" for stat in _TE_ROLLING_STATS for a in ["mean", "std", "max"]
    ],
    # All EWMA dropped (>0.98 corr with rolling means)
    "ewma": [],
    "trend": ["trend_targets", "trend_carries", "trend_snap_pct"],
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
    # TE keeps is_home (unlike RB/WR where it's zero-variance)
    "contextual": [
        "is_home",
        "week",
        "is_returning_from_absence",
        "days_rest",
        "practice_status",
        "game_status",
        "depth_chart_rank",
    ],
    # Keep 3 features with signal:
    #   implied_team_total (r=-0.035), implied_opp_total (r=0.029), is_dome (r=0.027)
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome"],
    "specific": TE_SPECIFIC_FEATURES,
}

# === Ridge ===
import numpy as np

# Per-target alpha grids — count targets (TDs, fumbles_lost) tolerate smaller
# alphas given their tighter spread; yards/receptions span the classic range.
TE_RIDGE_ALPHA_GRIDS = {
    "receiving_tds": [round(x, 4) for x in np.logspace(-1, 4, 15)],
    "receiving_yards": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "receptions": [round(x, 4) for x in np.logspace(-2, 3, 15)],
    "fumbles_lost": [round(x, 4) for x in np.logspace(-1, 4, 15)],
}

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. Reuses TE_RIDGE_ALPHA_GRIDS and searches over TE_ENET_L1_RATIOS.
TE_TRAIN_ELASTICNET = False
TE_ENET_L1_RATIOS = [0.3, 0.5, 0.7]

# === Neural Net (2012+ dataset: relaxed regularization) ===
TE_NN_BACKBONE_LAYERS = [96, 48]
TE_NN_HEAD_HIDDEN = 24
# Tighter override on the hurdle-NegBin reception head (TE convention: smaller
# cap than RB/WR). receiving_tds moved to plain Poisson NLL.
TE_NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 32}
TE_NN_DROPOUT = 0.30
TE_NN_LR = 5e-4
TE_NN_WEIGHT_DECAY = 3e-4
TE_NN_EPOCHS = 300
TE_NN_BATCH_SIZE = 128
TE_NN_PATIENCE = 25

# === Loss Weights ===
# Yards head keeps the 2.0/delta rebalance; Poisson and hurdle-NegBin heads
# use weight 1.0 (loss already near ~1.0 at typical rates).
TE_LOSS_WEIGHTS = {
    "receiving_tds": 1.0,  # Poisson NLL
    "receiving_yards": 0.133,  # 2.0 / 15  (Huber)
    "receptions": 1.0,  # hurdle_negbin, fraction-scaled internally
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (raw-stat units) ===
# Only Huber heads need a delta.
TE_HUBER_DELTAS = {
    "receiving_yards": 15.0,
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
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
    "targets",
    "receptions",
    "carries",
    "snap_pct",
]
# Categories of TE_INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# TE_ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
# ``defense`` is also excluded: TE_OPP_ATTN_HISTORY_STATS feeds the opposing
# defense's trailing form through a parallel attention branch, which makes
# the L5 static aggregates redundant for the NN. (They stay in
# TE_INCLUDE_FEATURES["defense"] so Ridge / LightGBM still see them.)
TE_ATTN_STATIC_CATEGORIES = [
    "prior_season",
    "matchup",
    "contextual",
    "weather_vegas",
]
TE_ATTN_STATIC_FEATURES = [c for cat in TE_ATTN_STATIC_CATEGORIES for c in TE_INCLUDE_FEATURES[cat]]

# Per-game opponent-defense stats fed to the second attention branch. Mirror
# the L5 static aggregates (opp_def_*_L5) but unrolled per game, so the NN
# learns the trailing-form weighting itself instead of being handed a fixed
# 5-game mean. Built by src.features.engineer.build_opp_defense_history_arrays.
TE_OPP_ATTN_HISTORY_STATS = [
    "def_sacks",
    "def_pass_yds_allowed",
    "def_pass_td_allowed",
    "def_ints",
    "def_rush_yds_allowed",
    "def_pts_allowed",
]
TE_OPP_ATTN_MAX_SEQ_LEN = 17
# Hurdle gate on receptions + BCE gate on receiving_tds. Mirrors the RB
# "Variant C" config from scripts/ablate_rb_gate.py (see RB/rb_config.py
# for the ablation table). PR #96 benchmark review flagged a +0.052
# per-target MAE regression on receiving_tds when the gate came off;
# restoring the BCE gate brings that back without disturbing the
# reception hurdle. head_losses keeps receiving_tds on ``poisson_nll``
# (BCE gate is additive via ``gated_targets``).
TE_ATTN_GATED = True
TE_GATED_TARGETS = ["receptions", "receiving_tds"]
TE_ATTN_GATE_HIDDEN = 16
TE_ATTN_GATE_WEIGHT = 1.0

# Per-head loss family. TDs + fumbles on Poisson NLL; receptions on
# zero-truncated NegBin-2 hurdle.
TE_HEAD_LOSSES = {
    "receiving_tds": "poisson_nll",
    "receiving_yards": "huber",
    "receptions": "hurdle_negbin",
    "fumbles_lost": "poisson_nll",
}

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


# === Tiny config for E2E smoke tests ===
# Shrunken NN (2 layers x 8 units), 1 epoch — keeps pipeline round-trip
# under the 20s budget while exercising the full orchestration.
TE_CONFIG_TINY = {
    "targets": TE_TARGETS,
    # Tiny ridge grid — single alpha per target, no refinement.
    "ridge_alpha_grids": {t: [1.0] for t in TE_TARGETS},
    "specific_features": TE_SPECIFIC_FEATURES,
    # NN hyperparams: 2 layers x 8 units, 1 epoch, no LightGBM/attention.
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 10,
    "loss_weights": TE_LOSS_WEIGHTS,
    "huber_deltas": TE_HUBER_DELTAS,
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    # Disable attention and LightGBM — they balloon runtime.
    "train_attention_nn": False,
    "train_lightgbm": False,
}
