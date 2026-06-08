"""WR hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers (training pipeline, serving, tests) read from the
exported ``POSITION_CONFIG`` exclusively. ``CONFIG_TINY`` and
``ATTN_STATIC_CATEGORIES`` remain at module level — the former is used by
direct-import e2e tests, the latter by the attention-static whitelist test.
"""

from src.shared.position_config import (
    DEFAULT_ATTN_STATIC_CATEGORIES,
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
    # Boom-tier red-zone / opportunity (rolling, leakage-safe shift=1; built in
    # src/wr/features.py). "specific" is NOT an ATTN_STATIC category, so these feed
    # Ridge+LGBM+plain-NN but stay out of the attention static branch (the rolling
    # stop-rule). Validated A/B src/tuning/ab_boom_signals_wr.py (+all).
    "redzone_targets_L3",
    "redzone_target_share_L3",
    "opportunity_index_L3",
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
    ]
    + [
        # ff_opportunity expected-production prior (static "opportunity" signal,
        # leakage-safe S-1 mean). Built in src.features.engineer from the
        # per-game *_exp columns merged by src.data.external_sources.
        "prior_season_mean_total_fantasy_points_exp",
        "prior_season_mean_rec_yards_gained_exp",
        "prior_season_mean_receptions_exp",
        # Boom-tier parity adds (prior-season, static-eligible; A/B +all,
        # src/tuning/ab_boom_signals_wr.py). Red-zone touch prior + games-played
        # are already baked into the splits (src.features.engineer, all positions);
        # catch_rate is derived in src/wr/features.py (S-1 mean recv / mean tgt).
        "prior_season_total_redzone_touches",
        "prior_season_mean_redzone_touches_per_game",
        "prior_season_games_played",
        "prior_season_mean_catch_rate",
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
        # Contract value — team-investment / expected-role prior. Static
        # player-season state, merged by src.data.external_sources.
        "contract_apy_cap_pct",
        "contract_guaranteed",
        "contract_years_remaining",
        "contract_age",
        # Role-inheritance (next-man-up): current-week vacancy when a higher-target
        # WR is Out/Doubtful. Built in src.features.engineer._build_inheritance_features
        # (WR role proxy = per-game targets); validated A/B src/tuning/ab_history_token.py
        # (modest-but-real on the inheritor-subgroup bias, all models). "contextual" is an
        # ATTN_STATIC category, so these feed the NN static branch too (NOT attn_history_stats).
        "is_top_available",
        "inherited_opportunity",
    ],
    # WR carries wind_adjusted/temp_adjusted (TE's config omits them by
    # design): the deep/sideline routes that drive WR receiving_yards are
    # far more wind- and temperature-sensitive than the short, middle-of-
    # field routes that dominate TE production, so the weather signal earns
    # its place in the WR whitelist but not the TE one. This asymmetry is
    # intentional — do not "sync" TE to match.
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
    # Non-None so the e2e shard exercises the receptions head-width override
    # plumbing (_TINY_OVERRIDES forces None; CONFIG_TINY merges last). (#565)
    "nn_head_hidden_overrides": {"receptions": 8},
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 64,
    "nn_patience": 1,
    "nn_log_every": 1,
    "loss_weights": {
        "receiving_tds": 1.0,
        "receiving_yards": 0.0667,  # 1 / 15 (MSE)
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
    # Relaxed from the global 6 → 1: relaxing the train-only filter lowers cold-start
    # MAE (attn would_filter +0.211±0.064, 8-seed) with no kept regression. TODO.md.
    min_games_per_season=1,
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
    # 2012+ dataset: widened from [96] to [128] to exploit the largest
    # training set. Most data → most capacity with the least overfit risk.
    nn_backbone_layers=[128],
    nn_head_hidden=32,
    # Larger head for the hurdle-NegBin reception head (two value outputs).
    # receiving_tds uses Poisson NLL (head_losses) plus a BCE gate via
    # gated_targets — both heads use the standard hidden size.
    nn_head_hidden_overrides={"receptions": 64},
    nn_dropout=0.20,
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=1e-3,
    nn_weight_decay=1e-4,
    nn_epochs=250,
    # 512 (vs 128/256 elsewhere) is intentional: WR has the largest training
    # set of any position, so the bigger batch gives smoother gradients and
    # faster epochs without hurting generalization — tuned, not a typo. (The
    # attention NN path below decouples to attn_batch_size=512.)
    nn_batch_size=512,
    nn_patience=25,
    # TDs + fumbles on Poisson NLL; receptions on zero-truncated NegBin-2
    # hurdle (see GatedHead + hurdle_negbin_value_loss).
    head_losses={
        "receiving_tds": "poisson_nll",
        "receiving_yards": "mse",
        "receptions": "hurdle_negbin",
        "fumbles_lost": "poisson_nll",
    },
    # receiving_yards switched Huber -> MSE to chase the elite tail; weight
    # = 1/delta (gradient-matched to the old 2.0/delta Huber, half its value).
    # Poisson/hurdle heads stay 1.0.
    loss_weights={
        "receiving_tds": 1.0,
        "receiving_yards": 0.0667,  # 1 / 15 (MSE)
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    # Characteristic error scale the MSE weight (1/delta) derives from.
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
    # Learned embedding for empty-history (season-opener) rows — reduces the
    # attention NN's opener over-prediction (player branch only; see
    # src/shared/neural_net.py and src/analysis/rb_lgbm_disagreement_findings.md).
    attn_no_history_embedding=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=0,
    attn_max_seq_len=17,
    attn_positional_encoding=True,
    attn_dropout=0.0,
    # Explicit attention-NN optim knobs (decoupled from the MLP nn_* path,
    # which uses batch=512 / wd=1e-4 here). The 2026-06 batch/LR ablation
    # selected b2_lrlin: double the attention batch and LR.
    attn_lr=2e-3,
    attn_weight_decay=5e-5,
    attn_batch_size=512,
    attn_patience=20,
    attn_cosine_eta_min=2e-5,
    attn_history_stats=[
        "receiving_yards",
        "rushing_yards",
        "receiving_tds",
        "rushing_tds",
        "targets",
        "receptions",
        "fumbles_lost",
        "carries",
        "snap_pct_raw",
        # Per-game ff_opportunity expected receiving stats, merged by
        # src.data.external_sources. Leakage-safe via build_game_history_arrays
        # (prior in-season games only).
        "rec_yards_gained_exp",
        "rec_touchdown_exp",
        "receptions_exp",
        "rec_first_down_exp",
        # Per-game game-script context + realized final score, already merged
        # onto every player-week row by build_position_features (schedule +
        # team_box_score merges). Gives the attention history the game
        # environment of each prior game so the NN can learn this WR's
        # conditional response to matchup/script — the boom/bust signal the
        # player-only history was blind to. Mirrors RB; current-week
        # counterparts live on the static branch (weather_vegas/contextual).
        # Leakage-safe via build_game_history_arrays (prior in-season games).
        "implied_team_total",
        "implied_opp_total",
        "is_home",
        "days_rest",
        "team_points_scored",
        "opp_team_points_scored",
        "team_pass_attempts",
        "team_passing_yards",
        "team_rush_attempts",
        # Boom-tier red-zone + per-game usage tokens (raw per-game, genuinely absent
        # from WR's sequence — AGENTS.md history reach #2, the RB pattern; NOT the
        # rejected windowed inheritance token). redzone_targets/_share come from the
        # splits; game_target_share/_hhi/_opportunity_index are built in
        # src/wr/features.py. Leakage-safe via build_game_history_arrays' own shift.
        # Validated A/B src/tuning/ab_boom_signals_wr.py (+all).
        "redzone_targets",
        "redzone_target_share",
        "game_target_share",
        "game_target_hhi",
        "game_opportunity_index",
    ],
    attn_static_features=derive_attn_static_features(_INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES),
    attn_gated=True,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    opp_attn_history_stats=list(DEFAULT_OPP_DEF_HISTORY_STATS),
    opp_attn_max_seq_len=17,
    # === LightGBM (Optuna retune, 50 trials, CV MAE 4.6876) ===
    # Switched "huber" → "regression" (L2/MSE) to chase the elite tail; was
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
    lgbm_objective="regression",
    accepts_dataframes=True,
    cpu_only=False,
)
