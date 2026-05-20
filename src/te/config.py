from src.shared.position_config import (
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_ATTN_MAX_SEQ_LEN,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# === TE Target Decomposition ===
# Raw-stat targets; fantasy points are aggregated post-prediction via
# src.shared.aggregate_targets.predictions_to_fantasy_points("TE", ...).
# Rushing targets dropped — TE rushing stats are near-zero (noise > signal).
TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

# === TE-Specific Features ===
SPECIFIC_FEATURES = [
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
_ROLLING_STATS = [
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

INCLUDE_FEATURES = {
    # L3/L8 for all stats; snap_pct also keeps L5.
    # L5 mean/std/max dropped (>0.97 corr with L3/L8) except snap_pct.
    "rolling": [
        f"rolling_{a}_{stat}_L{w}"
        for stat in _ROLLING_STATS
        for w in [3, 5, 8]
        for a in ["mean", "std", "max"]
        if w != 5 or stat == "snap_pct"
    ],
    "prior_season": [
        f"prior_season_{a}_{stat}" for stat in _ROLLING_STATS for a in ["mean", "std", "max"]
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
    "specific": SPECIFIC_FEATURES,
}

# === Ridge ===
# Per-target alpha grids — count targets (TDs, fumbles_lost) tolerate smaller
# alphas given their tighter spread; yards/receptions span the classic range.
_ALPHA_YARDS = alpha_grid(-2, 3, 15)
_ALPHA_COUNTS = alpha_grid(-1, 4, 15)

RIDGE_ALPHA_GRIDS = {
    "receiving_tds": _ALPHA_COUNTS,
    "receiving_yards": _ALPHA_YARDS,
    "receptions": _ALPHA_YARDS,
    "fumbles_lost": _ALPHA_COUNTS,
}

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. Reuses RIDGE_ALPHA_GRIDS and searches over ENET_L1_RATIOS.
TRAIN_ELASTICNET = False
ENET_L1_RATIOS = list(DEFAULT_ENET_L1_RATIOS)

# === Neural Net (2012+ dataset: relaxed regularization) ===
NN_BACKBONE_LAYERS = [96, 48]
NN_HEAD_HIDDEN = 24
# Tighter override on the hurdle-NegBin reception head (TE convention: smaller
# cap than RB/WR). receiving_tds moved to plain Poisson NLL.
NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 32}
NN_DROPOUT = 0.30
# All 4 TE heads are non-negative raw stats (TDs, yards, receptions, fumbles).
NN_NON_NEGATIVE_TARGETS = set(TARGETS)
NN_LR = 5e-4
NN_WEIGHT_DECAY = 3e-4
NN_EPOCHS = 300
NN_BATCH_SIZE = 128
NN_PATIENCE = 25

# === Loss Weights ===
# Yards head keeps the 2.0/delta rebalance; Poisson and hurdle-NegBin heads
# use weight 1.0 (loss already near ~1.0 at typical rates).
LOSS_WEIGHTS = {
    "receiving_tds": 1.0,  # Poisson NLL
    "receiving_yards": 0.133,  # 2.0 / 15  (Huber)
    "receptions": 1.0,  # hurdle_negbin, fraction-scaled internally
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (raw-stat units) ===
# Only Huber heads need a delta.
HUBER_DELTAS = {
    "receiving_yards": 15.0,
}

# === LR Scheduler ===
SCHEDULER_TYPE = "onecycle"
ONECYCLE_MAX_LR = 2e-3
ONECYCLE_PCT_START = 0.3

# === Attention NN (game history variant) ===
TRAIN_ATTENTION_NN = True
ATTN_D_MODEL = 32
ATTN_N_HEADS = 2
ATTN_ENCODER_HIDDEN_DIM = 0
ATTN_MAX_SEQ_LEN = 17
ATTN_POSITIONAL_ENCODING = True
ATTN_DROPOUT = 0.0
ATTN_HISTORY_STATS = [
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
    "targets",
    "receptions",
    "fumbles_lost",
    "carries",
    "snap_pct",
]
# Categories of INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
# ``defense`` is also excluded: OPP_ATTN_HISTORY_STATS feeds the opposing
# defense's trailing form through a parallel attention branch, which makes
# the L5 static aggregates redundant for the NN. (They stay in
# INCLUDE_FEATURES["defense"] so Ridge / LightGBM still see them.)
ATTN_STATIC_CATEGORIES = [
    "prior_season",
    "matchup",
    "contextual",
    "weather_vegas",
]
ATTN_STATIC_FEATURES = derive_attn_static_features(INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES)

# Per-game opponent-defense stats fed to the second attention branch. Mirror
# the L5 static aggregates (opp_def_*_L5) but unrolled per game, so the NN
# learns the trailing-form weighting itself instead of being handed a fixed
# 5-game mean. Built by src.features.engineer.build_opp_defense_history_arrays.
OPP_ATTN_HISTORY_STATS = list(DEFAULT_OPP_DEF_HISTORY_STATS)
OPP_ATTN_MAX_SEQ_LEN = DEFAULT_OPP_ATTN_MAX_SEQ_LEN
# Hurdle gate on receptions + BCE gate on receiving_tds. Mirrors the RB
# "Variant C" config from src/tuning/ablate_rb_gate.py (see src/rb/config.py
# for the ablation table). PR #96 benchmark review flagged a +0.052
# per-target MAE regression on receiving_tds when the gate came off;
# restoring the BCE gate brings that back without disturbing the
# reception hurdle. head_losses keeps receiving_tds on ``poisson_nll``
# (BCE gate is additive via ``gated_targets``).
ATTN_GATED = True
GATED_TARGETS = ["receptions", "receiving_tds"]
ATTN_GATE_HIDDEN = 16
ATTN_GATE_WEIGHT = 1.0

# Per-head loss family. TDs + fumbles on Poisson NLL; receptions on
# zero-truncated NegBin-2 hurdle.
HEAD_LOSSES = {
    "receiving_tds": "poisson_nll",
    "receiving_yards": "huber",
    "receptions": "hurdle_negbin",
    "fumbles_lost": "poisson_nll",
}

# === LightGBM (Optuna retune, 50 trials, CV MAE 3.5942) ===
# Flipped from ``"fair"`` to ``"huber"`` as part of the PR 3 LGBM unification
# (QB is the one exception — see LGBM_OBJECTIVE). Retuned on the huber
# objective, holdout comparison vs the old fair config:
#   Total MAE        3.534 -> 3.506  (-0.028)
#   Receiving Yards  15.42 -> 14.58  (-0.84)
#   Receiving Tds    0.260 -> 0.281  (+0.021)
#   Receptions       1.231 -> 1.220  (-0.011)
#   Top-12 hit rate  0.479 -> 0.487  (+0.008)
#   Spearman rho     0.569 -> 0.592  (+0.023)
# CV MAE improved (-0.015). Holdout Total MAE improved (-0.028). Full
# tune_lgbm_results.json in retune run 24823926033.
TRAIN_LIGHTGBM = True
LGBM_N_ESTIMATORS = 1900
LGBM_LEARNING_RATE = 0.08219987
LGBM_NUM_LEAVES = 15
LGBM_MAX_DEPTH = 8
LGBM_SUBSAMPLE = 0.7359385
LGBM_COLSAMPLE_BYTREE = 0.5750816
LGBM_REG_LAMBDA = 1.1011751
LGBM_REG_ALPHA = 1.2289840
LGBM_MIN_CHILD_SAMPLES = 51
LGBM_MIN_SPLIT_GAIN = 0.16878265
LGBM_OBJECTIVE = "huber"


# === Tiny config for E2E smoke tests ===
# Shrunken NN (2 layers x 8 units), 1 epoch — keeps pipeline round-trip
# under the 20s budget while exercising the full orchestration.
CONFIG_TINY = {
    "targets": TARGETS,
    # Tiny ridge grid — single alpha per target, no refinement.
    "ridge_alpha_grids": {t: [1.0] for t in TARGETS},
    "specific_features": SPECIFIC_FEATURES,
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
    "loss_weights": LOSS_WEIGHTS,
    "huber_deltas": HUBER_DELTAS,
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    # Disable attention and LightGBM — they balloon runtime.
    "train_attention_nn": False,
    "train_lightgbm": False,
}


# === Bundled config object ===
# Single source of truth for downstream consumers (registry / pipeline / serving).
POSITION_CONFIG = PositionConfig(
    name="TE",
    targets=TARGETS,
    specific_features=SPECIFIC_FEATURES,
    include_features=INCLUDE_FEATURES,
    ridge_alpha_grids=RIDGE_ALPHA_GRIDS,
    train_elasticnet=TRAIN_ELASTICNET,
    enet_l1_ratios=ENET_L1_RATIOS,
    nn_backbone_layers=NN_BACKBONE_LAYERS,
    nn_head_hidden=NN_HEAD_HIDDEN,
    nn_dropout=NN_DROPOUT,
    nn_non_negative_targets=NN_NON_NEGATIVE_TARGETS,
    nn_lr=NN_LR,
    nn_weight_decay=NN_WEIGHT_DECAY,
    nn_epochs=NN_EPOCHS,
    nn_batch_size=NN_BATCH_SIZE,
    nn_patience=NN_PATIENCE,
    nn_head_hidden_overrides=NN_HEAD_HIDDEN_OVERRIDES,
    head_losses=HEAD_LOSSES,
    loss_weights=LOSS_WEIGHTS,
    huber_deltas=HUBER_DELTAS,
    gated_targets=GATED_TARGETS,
    scheduler_type=SCHEDULER_TYPE,
    onecycle_max_lr=ONECYCLE_MAX_LR,
    onecycle_pct_start=ONECYCLE_PCT_START,
    train_attention_nn=TRAIN_ATTENTION_NN,
    attn_d_model=ATTN_D_MODEL,
    attn_n_heads=ATTN_N_HEADS,
    attn_encoder_hidden_dim=ATTN_ENCODER_HIDDEN_DIM,
    attn_max_seq_len=ATTN_MAX_SEQ_LEN,
    attn_positional_encoding=ATTN_POSITIONAL_ENCODING,
    attn_dropout=ATTN_DROPOUT,
    attn_history_stats=ATTN_HISTORY_STATS,
    attn_static_features=ATTN_STATIC_FEATURES,
    attn_gated=ATTN_GATED,
    attn_gate_hidden=ATTN_GATE_HIDDEN,
    attn_gate_weight=ATTN_GATE_WEIGHT,
    opp_attn_history_stats=OPP_ATTN_HISTORY_STATS,
    opp_attn_max_seq_len=OPP_ATTN_MAX_SEQ_LEN,
    train_lightgbm=TRAIN_LIGHTGBM,
    lgbm_n_estimators=LGBM_N_ESTIMATORS,
    lgbm_learning_rate=LGBM_LEARNING_RATE,
    lgbm_num_leaves=LGBM_NUM_LEAVES,
    lgbm_max_depth=LGBM_MAX_DEPTH,
    lgbm_subsample=LGBM_SUBSAMPLE,
    lgbm_colsample_bytree=LGBM_COLSAMPLE_BYTREE,
    lgbm_reg_lambda=LGBM_REG_LAMBDA,
    lgbm_reg_alpha=LGBM_REG_ALPHA,
    lgbm_min_child_samples=LGBM_MIN_CHILD_SAMPLES,
    lgbm_min_split_gain=LGBM_MIN_SPLIT_GAIN,
    lgbm_objective=LGBM_OBJECTIVE,
    accepts_dataframes=True,
    cpu_only=False,
    has_cv_runner=False,
)
