"""TE hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers read from the exported ``POSITION_CONFIG`` exclusively.
``CONFIG_TINY`` (e2e tests import by name) and ``ATTN_STATIC_CATEGORIES``
(attention-static whitelist test) remain at module level.
"""

from src.shared.position_config import (
    DEFAULT_ATTN_STATIC_CATEGORIES,
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# Rushing targets dropped — TE rushing stats are near-zero (noise > signal).
_TARGETS = ["receiving_tds", "receiving_yards", "receptions", "fumbles_lost"]

_SPECIFIC_FEATURES = [
    "yards_per_reception_L3",
    "reception_rate_L3",
    "yac_per_reception_L3",
    "team_te_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
    "air_yards_per_target_L3",
    "td_rate_per_target_L3",
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
    ],
    "ewma": [],
    "trend": ["trend_targets", "trend_carries", "trend_snap_pct"],
    # Inherited from WR's share block — kept both L3 and L5 here without a
    # TE-specific collinearity audit. RB dropped target_share_L5 / carry_share_L5
    # / carry_share_L3 after measuring r > 0.96 with their L3 / team-share
    # counterparts (see src/rb/config.py share-block comment). May revisit
    # once a TE audit is run.
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
    ],
    # Keep 3 features with signal: implied_team_total (r=-0.035),
    # implied_opp_total (r=0.029), is_dome (r=0.027).
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome"],
    "specific": _SPECIFIC_FEATURES,
}

ATTN_STATIC_CATEGORIES = DEFAULT_ATTN_STATIC_CATEGORIES

# Tiny config for E2E smoke tests. Shrunken NN (2 layers × 8 units, 1 epoch);
# pipeline round-trip under the 20s budget while exercising orchestration.
CONFIG_TINY = {
    "targets": _TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in _TARGETS},
    "specific_features": _SPECIFIC_FEATURES,
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 10,
    "loss_weights": {
        "receiving_tds": 1.0,
        "receiving_yards": 0.0667,  # 1 / 15 (MSE)
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    "huber_deltas": {"receiving_yards": 15.0},
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "train_attention_nn": False,
    "train_lightgbm": False,
}


POSITION_CONFIG = PositionConfig(
    name="TE",
    # Relaxed from the global 6 → 1: relaxing the train-only filter lowers cold-start
    # MAE (LGBM would_filter +0.471 deterministic, clean monotonic) with no kept
    # regression. See TODO.md.
    min_games_per_season=1,
    targets=_TARGETS,
    specific_features=_SPECIFIC_FEATURES,
    include_features=_INCLUDE_FEATURES,
    # Per-target alpha grids — count targets (TDs, fumbles) tolerate smaller
    # alphas given their tighter spread; yards/receptions span the classic range.
    ridge_alpha_grids={
        "receiving_tds": alpha_grid(-1, 4, 15),
        "receiving_yards": alpha_grid(-2, 3, 15),
        "receptions": alpha_grid(-2, 3, 15),
        "fumbles_lost": alpha_grid(-1, 4, 15),
    },
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    # 2012+ dataset: relaxed regularization.
    nn_backbone_layers=[96, 48],
    nn_head_hidden=24,
    # Larger head for the hurdle-NegBin reception head (two value outputs:
    # mu + log_alpha) — the +32 buys it capacity to fit both parameters.
    # Sized smaller than RB/WR (64) because TE receptions are lower-volume
    # and the smaller backbone (nn_head_hidden=24) keeps the head/backbone
    # ratio in line.
    nn_head_hidden_overrides={"receptions": 32},
    nn_dropout=0.30,
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=5e-4,
    nn_weight_decay=3e-4,
    nn_epochs=300,
    nn_batch_size=128,
    nn_patience=25,
    head_losses={
        "receiving_tds": "poisson_nll",
        "receiving_yards": "mse",
        "receptions": "hurdle_negbin",
        "fumbles_lost": "poisson_nll",
    },
    # receiving_yards switched Huber -> MSE to chase the elite tail (elite TE was
    # under-projected ~-1.5 FP); weight = 1/delta (half the old 2.0/delta Huber,
    # gradient-matched at e~delta). Poisson/hurdle heads stay 1.0.
    loss_weights={
        "receiving_tds": 1.0,
        "receiving_yards": 0.0667,  # 1 / 15 (MSE)
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    # Characteristic error scale the MSE weight (1/delta) derives from.
    huber_deltas={"receiving_yards": 15.0},
    # Hurdle gate on receptions + BCE gate on receiving_tds. Mirrors RB's
    # "Variant C". PR #96 review flagged +0.052 per-target MAE regression
    # on receiving_tds without the gate; restoring it brings that back
    # without disturbing the reception hurdle.
    gated_targets=["receptions", "receiving_tds"],
    scheduler_type="onecycle",
    onecycle_max_lr=2e-3,
    onecycle_pct_start=0.3,
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
    # Explicit attention-NN optimizer/batch knobs — mirrors RB. Without these,
    # src/shared/pipeline.py falls back to nn_lr / nn_weight_decay / nn_batch_size
    # via cfg.get(...) at lines 766-767 + 718, which is correct today but silent:
    # a future bump of nn_* would drag the attn NN along uninspected.
    # The 2026-06 batch/LR ablation selected b2_lrlin for TE.
    attn_lr=1e-3,
    attn_weight_decay=3e-4,
    attn_batch_size=256,
    attn_patience=20,
    attn_onecycle_max_lr=4e-3,
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
        # environment of each prior game so the NN can learn this TE's
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
    ],
    attn_static_features=derive_attn_static_features(_INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES),
    attn_gated=True,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    opp_attn_history_stats=list(DEFAULT_OPP_DEF_HISTORY_STATS),
    opp_attn_max_seq_len=17,
    # === LightGBM (Optuna retune, 50 trials, CV MAE 3.5942) ===
    # Switched "huber" → "regression" (L2/MSE) to chase the elite tail; was
    # "fair" → "huber" in PR 3 LGBM unification. Holdout: Total MAE
    # 3.534 → 3.506 (-0.028), top-12 hit rate +0.008, Spearman +0.023.
    train_lightgbm=True,
    lgbm_n_estimators=1900,
    lgbm_learning_rate=0.08219987,
    lgbm_num_leaves=15,
    lgbm_max_depth=8,
    lgbm_subsample=0.7359385,
    lgbm_colsample_bytree=0.5750816,
    lgbm_reg_lambda=1.1011751,
    lgbm_reg_alpha=1.2289840,
    lgbm_min_child_samples=51,
    lgbm_min_split_gain=0.16878265,
    lgbm_objective="regression",
    accepts_dataframes=True,
    cpu_only=False,
    has_cv_runner=True,
)
