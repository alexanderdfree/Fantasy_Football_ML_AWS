"""WR hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers (training pipeline, serving, tests) read from the
exported ``POSITION_CONFIG`` exclusively. ``CONFIG_TINY`` and
``ATTN_STATIC_CATEGORIES`` remain at module level — the former is used by
direct-import e2e tests, the latter by the attention-static whitelist test.
"""

from src.shared.position_config import (
    DEFAULT_ATTN_STATIC_CATEGORIES,
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# Rushing targets dropped - WR rushing stats are too sparse for reliable signal.
_TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

_SPECIFIC_FEATURES = [
    "yards_per_reception_L3",
    "yards_per_target_L3",
    "reception_rate_L3",
    "air_yards_per_target_L3",
    "yac_per_reception_L3",
    "team_wr_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
]

_ROLLING_STATS = [
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

_INCLUDE_FEATURES = {
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
    # All EWMA dropped (>0.98 corr with rolling means).
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
    "specific": _SPECIFIC_FEATURES,
}

ATTN_STATIC_CATEGORIES = DEFAULT_ATTN_STATIC_CATEGORIES

# Tiny config for E2E smoke tests. Shrunk hyperparameters; keeps full
# pipeline orchestration under 20s on CPU. Used by tests/wr/test_pipeline_e2e.py.
CONFIG_TINY = {
    "targets": _TARGETS,
    "specific_features": _SPECIFIC_FEATURES,
    "ridge_alpha_grids": {t: [1.0] for t in _TARGETS},
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
    "loss_weights": {
        "receiving_tds": 1.0,
        "receiving_yards": 0.133,
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    "huber_deltas": {"receiving_yards": 15.0},
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "train_attention_nn": False,
    "train_lightgbm": False,
}


# Single source of truth for downstream consumers (registry / pipeline / serving).
POSITION_CONFIG = PositionConfig(
    name="WR",
    targets=_TARGETS,
    specific_features=_SPECIFIC_FEATURES,
    include_features=_INCLUDE_FEATURES,
    # Per-target alpha grids — yards/TDs/fumbles on the standard grid;
    # receptions wider since the head varies more.
    ridge_alpha_grids={
        "receiving_tds": alpha_grid(-1, 4, 15),
        "receiving_yards": alpha_grid(-1, 4, 15),
        "receptions": alpha_grid(-2, 3, 15),
        "fumbles_lost": alpha_grid(-1, 4, 15),
    },
    # PCR: 30 components. Benchmark showed -0.094 MAE vs no-PCA baseline
    # (4.507 → 4.413). PCA removes collinear directions the alpha grid
    # can't fully address.
    ridge_pca_components=30,
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    # 2012+ dataset: widened from [96] to [128] to exploit the largest
    # training set. Most data → most capacity with the least overfit risk.
    nn_backbone_layers=[128],
    nn_head_hidden=32,
    # Larger head for the hurdle-NegBin reception head (two value outputs).
    # receiving_tds moved to Poisson NLL so no extra capacity needed.
    nn_head_hidden_overrides={"receptions": 64},
    nn_dropout=0.20,
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=1e-3,
    nn_weight_decay=1e-4,
    nn_epochs=250,
    nn_batch_size=512,
    nn_patience=25,
    # TDs + fumbles on Poisson NLL; receptions on zero-truncated NegBin-2
    # hurdle (see GatedHead + hurdle_negbin_value_loss).
    head_losses={
        "receiving_tds": "poisson_nll",
        "receiving_yards": "huber",
        "receptions": "hurdle_negbin",
        "fumbles_lost": "poisson_nll",
    },
    # Yards head keeps 2.0/delta rebalance; Poisson/hurdle heads use 1.0
    # (their losses already sit near ~1.0 at typical rates).
    loss_weights={
        "receiving_tds": 1.0,
        "receiving_yards": 0.133,  # 2.0 / 15
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    huber_deltas={"receiving_yards": 15.0},
    # Hurdle gate on receptions + BCE gate on receiving_tds. Mirrors RB's
    # "Variant C" — the BCE gate restores per-target access to the
    # "did this player score a TD this week?" signal that's otherwise
    # hidden inside the count-mean. head_losses keeps receiving_tds on
    # poisson_nll (BCE gate is additive via gated_targets).
    gated_targets=["receptions", "receiving_tds"],
    scheduler_type="cosine_warm_restarts",
    cosine_t0=40,
    cosine_t_mult=2,
    cosine_eta_min=1e-5,
    train_attention_nn=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=0,
    attn_max_seq_len=17,
    attn_positional_encoding=True,
    attn_dropout=0.0,
    attn_history_stats=[
        "receiving_yards",
        "rushing_yards",
        "receiving_tds",
        "rushing_tds",
        "targets",
        "receptions",
        "fumbles_lost",
        "carries",
        "snap_pct",
    ],
    attn_static_features=derive_attn_static_features(_INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES),
    attn_gated=True,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    opp_attn_history_stats=list(DEFAULT_OPP_DEF_HISTORY_STATS),
    opp_attn_max_seq_len=17,
    # === LightGBM (Optuna retune, 50 trials, CV MAE 4.6876) ===
    # "fair" → "huber" via PR 3 LGBM unification. Holdout: Total MAE
    # 4.203 → 4.221 (+0.018, well inside ±0.05 tolerance), top-12 hit
    # rate +0.004, Spearman +0.004.
    train_lightgbm=True,
    lgbm_n_estimators=1600,
    lgbm_learning_rate=0.08782007,
    lgbm_num_leaves=31,
    lgbm_max_depth=9,
    lgbm_subsample=0.7318847,
    lgbm_colsample_bytree=0.48205232,
    lgbm_reg_lambda=0.1113962,
    lgbm_reg_alpha=1.2740795,
    lgbm_min_child_samples=63,
    lgbm_min_split_gain=0.2346478,
    lgbm_objective="huber",
    accepts_dataframes=True,
    cpu_only=False,
    has_cv_runner=True,
)
