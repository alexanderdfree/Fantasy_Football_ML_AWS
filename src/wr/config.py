from src.shared.position_config import (
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_ATTN_MAX_SEQ_LEN,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# === WR Target Decomposition ===
# Raw-stat targets (see src/shared/aggregate_targets.py). Fantasy points are
# aggregated from these predictions via predictions_to_fantasy_points.
# Rushing targets dropped - WR rushing stats are too sparse for reliable signal.
TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

# === WR-Specific Features ===
SPECIFIC_FEATURES = [
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
    "specific": SPECIFIC_FEATURES,
}

# === Ridge ===
# PCR: 30 components. Benchmark showed -0.094 MAE vs no-PCA baseline (4.507 → 4.413).
# PCA removes collinear directions the alpha grid can't fully address.
RIDGE_PCA_COMPONENTS = 30
# Alpha grids sized to each target's dynamic range: yards use a wider high-alpha
# tail; count-style targets (TDs, receptions, fumbles) stay in the standard band.
_ALPHA_COUNTS = alpha_grid(-1, 4, 15)
_ALPHA_RECEPTIONS = alpha_grid(-2, 3, 15)

RIDGE_ALPHA_GRIDS = {
    "receiving_tds": _ALPHA_COUNTS,
    "receiving_yards": _ALPHA_COUNTS,
    "receptions": _ALPHA_RECEPTIONS,
    "fumbles_lost": _ALPHA_COUNTS,
}

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. Reuses RIDGE_ALPHA_GRIDS and searches over ENET_L1_RATIOS.
# Skips PCA regardless of RIDGE_PCA_COMPONENTS — L1 sparsity on a rotated
# basis zeros components, not original features.
TRAIN_ELASTICNET = False
ENET_L1_RATIOS = list(DEFAULT_ENET_L1_RATIOS)

# === Neural Net ===
# 2012+ dataset: widened from [96] to [128] to exploit largest training set.
# Largest position dataset can support more capacity with less overfitting risk.
NN_BACKBONE_LAYERS = [128]
NN_HEAD_HIDDEN = 32
# Larger head for the hurdle-NegBin reception head (two value outputs).
# receiving_tds moved to plain Poisson NLL so it no longer needs extra capacity.
NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 64}
NN_DROPOUT = 0.20
# All 4 WR heads are non-negative raw stats (TDs, yards, receptions, fumbles).
NN_NON_NEGATIVE_TARGETS = set(TARGETS)
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_EPOCHS = 250
NN_BATCH_SIZE = 512
NN_PATIENCE = 25

# === Loss Weights ===
# Yards head keeps the 2.0/delta rebalance (without it, yards gradients
# dominate the count heads). Poisson NLL and hurdle-NegBin heads use weight 1.0
# (their losses already sit near ~1.0 at typical rates).
LOSS_WEIGHTS = {
    "receiving_tds": 1.0,  # Poisson NLL
    "receiving_yards": 0.133,  # 2.0 / 15  (Huber)
    "receptions": 1.0,  # hurdle_negbin, fraction-scaled internally
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (per-target, raw-stat units) ===
# Only Huber heads need a delta.
HUBER_DELTAS = {
    "receiving_yards": 15.0,
}

# === LR Scheduler ===
SCHEDULER_TYPE = "cosine_warm_restarts"
COSINE_T0 = 40
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-5

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
# Hurdle gate on receptions + BCE gate on receiving_tds. Matches the
# "Variant C" config for RB (see src/rb/config.py for the ablation table).
# WR doesn't have its own ablation, but the mechanism is target-agnostic:
# the BCE gate on (y > 0) gives the attention branch per-target access to
# "did this player score a TD this week?" signal that's otherwise hidden
# inside the count-mean. The PR #96 benchmark review flagged a +0.049
# per-target MAE regression on receiving_tds when the gate came off; this
# restores the gate without disturbing the PR #96 reception hurdle win.
# head_losses below keeps receiving_tds on ``poisson_nll`` (BCE gate loss
# is added in addition to the Poisson NLL via ``gated_targets``).
ATTN_GATED = True
GATED_TARGETS = ["receptions", "receiving_tds"]
ATTN_GATE_HIDDEN = 16
ATTN_GATE_WEIGHT = 1.0

# Per-head loss family. TDs + fumbles on Poisson NLL; receptions on
# zero-truncated NegBin-2 hurdle (see GatedHead + hurdle_negbin_value_loss).
HEAD_LOSSES = {
    "receiving_tds": "poisson_nll",
    "receiving_yards": "huber",
    "receptions": "hurdle_negbin",
    "fumbles_lost": "poisson_nll",
}

# === LightGBM (Optuna retune, 50 trials, CV MAE 4.6876) ===
# Flipped from ``"fair"`` to ``"huber"`` as part of the PR 3 LGBM unification
# (QB is the one exception — see LGBM_OBJECTIVE). Retuned on the huber
# objective, holdout comparison vs the old fair config:
#   Total MAE        4.203 -> 4.221  (+0.018)
#   Receiving Yards  19.94 -> 19.87  (-0.07)
#   Receiving Tds    0.269 -> 0.293  (+0.024)
#   Receptions       1.368 -> 1.351  (-0.017)
#   Top-12 hit rate  0.389 -> 0.393  (+0.004)
#   Spearman rho     0.644 -> 0.648  (+0.004)
# CV MAE improved (-0.044). Holdout nudge (+0.018) is well inside the plan's
# ±0.05 tolerance. Full tune_lgbm_results.json in retune run 24823926033.
TRAIN_LIGHTGBM = True
LGBM_N_ESTIMATORS = 1600
LGBM_LEARNING_RATE = 0.08782007
LGBM_NUM_LEAVES = 31
LGBM_MAX_DEPTH = 9
LGBM_SUBSAMPLE = 0.7318847
LGBM_COLSAMPLE_BYTREE = 0.48205232
LGBM_REG_LAMBDA = 0.1113962
LGBM_REG_ALPHA = 1.2740795
LGBM_MIN_CHILD_SAMPLES = 63
LGBM_MIN_SPLIT_GAIN = 0.2346478
LGBM_OBJECTIVE = "huber"


# === Tiny-scale config for E2E smoke tests ===
# Shrunk copy of the production hyperparameters: 1 epoch, 2-layer NN with 8
# units, no attention, no LightGBM. Keeps the full-pipeline E2E test under
# ~20s while still exercising every stage of run_pipeline().
CONFIG_TINY = {
    "targets": TARGETS,
    "specific_features": SPECIFIC_FEATURES,
    "ridge_alpha_grids": {t: [1.0] for t in TARGETS},
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
    "loss_weights": LOSS_WEIGHTS,
    "huber_deltas": HUBER_DELTAS,
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "train_attention_nn": False,
    "train_lightgbm": False,
}


# === Bundled config object ===
# Single source of truth for downstream consumers (registry / pipeline / serving).
POSITION_CONFIG = PositionConfig(
    name="WR",
    targets=TARGETS,
    specific_features=SPECIFIC_FEATURES,
    include_features=INCLUDE_FEATURES,
    ridge_alpha_grids=RIDGE_ALPHA_GRIDS,
    ridge_pca_components=RIDGE_PCA_COMPONENTS,
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
    cosine_t0=COSINE_T0,
    cosine_t_mult=COSINE_T_MULT,
    cosine_eta_min=COSINE_ETA_MIN,
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
    has_cv_runner=True,
)
