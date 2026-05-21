"""QB hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers (training pipeline, serving, tests) read from the
exported ``POSITION_CONFIG`` exclusively; no module-level UPPERCASE constants
are exposed. Private helper variables stay scoped to construction.
"""

from src.shared.position_config import (
    DEFAULT_ATTN_STATIC_CATEGORIES,
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# === QB-specific rolling stats ===
# Feed both the rolling/L3-5-8 and prior-season aggregates inside
# INCLUDE_FEATURES below.
_ROLLING_STATS = [
    "carries",
    "rushing_yards",
    "passing_yards",
    "attempts",
    "snap_pct",
]

# Yards targets stay on the (-2, 3) alpha grid; count targets (TDs, INTs,
# fumbles_lost) use the (-1, 4) grid because the smaller target scale lets
# stronger regularization dominate.
_ALPHA_YARDS = alpha_grid(-2, 3, 15)
_ALPHA_COUNTS = alpha_grid(-1, 4, 15)

# Predictions are raw NFL stats; fantasy points are aggregated post-prediction
# via src.shared.aggregate_targets.predictions_to_fantasy_points("QB", preds).
_TARGETS = [
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
]

# Explicit feature whitelist — new columns must be opted in, preventing
# silent leakage. L5 mean/std/max dropped (>0.97 corr with L3/L8) except
# snap_pct; passing_yards EWMA kept (others >0.98 corr with rolling means).
_INCLUDE_FEATURES = {
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
    "ewma": ["ewma_passing_yards_L3", "ewma_passing_yards_L5"],
    "trend": ["trend_carries", "trend_snap_pct"],
    # No target_share/air_yards_share — QBs have ~0 targets.
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
    "specific": [
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
    ],
}

ATTN_STATIC_CATEGORIES = DEFAULT_ATTN_STATIC_CATEGORIES


# Single source of truth for downstream consumers (registry / pipeline / serving).
# Read from POSITION_CONFIG exclusively; no module-level UPPERCASE constants.
POSITION_CONFIG = PositionConfig(
    name="QB",
    targets=_TARGETS,
    specific_features=_INCLUDE_FEATURES["specific"],
    include_features=_INCLUDE_FEATURES,
    # Yards on (-2,3) grid; counts on (-1,4) per target scale.
    ridge_alpha_grids={
        "passing_yards": _ALPHA_YARDS,
        "rushing_yards": _ALPHA_YARDS,
        "passing_tds": _ALPHA_COUNTS,
        "rushing_tds": _ALPHA_COUNTS,
        "interceptions": _ALPHA_COUNTS,
        "fumbles_lost": _ALPHA_COUNTS,
    },
    # === ElasticNet (optional parallel linear baseline) — off by default.
    # When enabled, reuses ridge_alpha_grids and searches over enet_l1_ratios.
    # Skips PCA on purpose: L1 on a rotated basis doesn't zero original
    # features, so PCA defeats the reason to pick ElasticNet.
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    # === Neural Net (2012+ dataset: wider backbone, relaxed regularization) ===
    nn_backbone_layers=[128],
    nn_head_hidden=32,
    nn_dropout=0.20,
    # All 6 QB heads are non-negative raw stats (yards, TD counts, INTs, fumbles).
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=5e-4,
    nn_weight_decay=3e-4,
    nn_epochs=300,
    nn_batch_size=128,
    nn_patience=25,
    nn_head_hidden_overrides={},
    # === Per-head loss families ===
    # TDs + INTs + fumbles use Poisson NLL — QB TD distributions are not
    # zero-inflated (median ~2 TDs/start), so a plain Poisson rate model
    # fits cleanly and the Huber-count-head-collapse failure mode (count
    # heads regressing to the mean under yards-dominated gradients) is
    # avoided without needing wider heads.
    head_losses={
        "passing_yards": "huber",
        "rushing_yards": "huber",
        "passing_tds": "poisson_nll",
        "rushing_tds": "poisson_nll",
        "interceptions": "poisson_nll",
        "fumbles_lost": "poisson_nll",
    },
    # Yards heads keep the 2.0/delta rebalance (without it, FP MAE regressed
    # 6.33 -> 6.63; fumbles_lost R² = -0.34). Poisson NLL heads use weight
    # 1.0 — at QB-scale rates (~1.5 TDs, ~0.7 INTs, ~0.4 fumbles) the
    # Poisson NLL is O(1), matching weighted yards.
    loss_weights={
        "passing_yards": 0.08,  # 2.0 / 25  (Huber)
        "rushing_yards": 0.133,  # 2.0 / 15  (Huber)
        "passing_tds": 1.0,  # Poisson NLL
        "rushing_tds": 1.0,  # Poisson NLL
        "interceptions": 1.0,  # Poisson NLL
        "fumbles_lost": 1.0,  # Poisson NLL
    },
    # Only Huber heads need a delta — count heads moved to Poisson NLL.
    huber_deltas={
        "passing_yards": 25.0,
        "rushing_yards": 15.0,
    },
    scheduler_type="cosine_warm_restarts",
    cosine_t0=40,
    cosine_t_mult=2,
    cosine_eta_min=1e-5,
    # === Attention NN (game history variant) ===
    train_attention_nn=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=0,
    attn_max_seq_len=17,
    attn_positional_encoding=True,
    attn_dropout=0.05,
    attn_lr=1e-3,
    attn_weight_decay=5e-5,
    attn_batch_size=256,
    attn_patience=35,
    attn_history_stats=[
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "attempts",
        "completions",
        "carries",
        "interceptions",
        "fumbles_lost",
        "snap_pct",
        "sacks",
        "sack_yards",
    ],
    attn_static_features=derive_attn_static_features(_INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES),
    # Gated hurdle heads disabled for QB — QBs throw so many TDs that the
    # zero-inflation assumption behind the hurdle model does not hold
    # (median TD count per start is ~2); a plain regression head beats
    # the two-stage gate.
    attn_gated=False,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    # Per-game opponent-defense stats fed to the second attention branch.
    # Mirrors the L5 static aggregates (opp_def_*_L5) but unrolled per
    # game, so the NN learns the trailing-form weighting itself instead
    # of being handed a fixed 5-game mean. Built by
    # src.features.engineer.build_opp_defense_history_arrays.
    opp_attn_history_stats=list(DEFAULT_OPP_DEF_HISTORY_STATS),
    opp_attn_max_seq_len=17,
    # === LightGBM (Optuna-tuned, 50 trials, CV MAE 5.7415) ===
    # QB is the one position that keeps Fair after PR 3's unification attempt.
    # 50-trial Huber retune regressed QB total MAE 6.269 -> 6.479
    # (+0.210 pts/game), with passing_yards MAE jumping 66.1 -> 71.2.
    # Root cause: LightGBM's Huber uses alpha=0.9 as a *quantile* — the 90th
    # percentile of residuals demarcates the quadratic-to-linear transition.
    # On QB passing_yards (typical residuals 0-100 yards, tail to 200+),
    # that puts 90% of residuals in Huber's quadratic zone. Fair's
    # log-curvature-everywhere downweights the tail smoothly and beats
    # Huber by 0.21 pts on QB holdout. RB/WR/TE/K/DST don't have a
    # passing_yards-like heavy tail and tolerate Huber.
    train_lightgbm=True,
    lgbm_n_estimators=1500,
    lgbm_learning_rate=0.0612763,
    lgbm_num_leaves=31,
    lgbm_subsample=0.867443,
    lgbm_colsample_bytree=0.907776,
    lgbm_reg_lambda=6.75023,
    lgbm_reg_alpha=0.00309259,
    lgbm_min_child_samples=59,
    lgbm_min_split_gain=0.0632242,
    lgbm_objective="fair",
    accepts_dataframes=True,
    cpu_only=False,
    has_cv_runner=True,
)
