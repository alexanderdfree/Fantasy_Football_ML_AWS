"""DST hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers read from the exported ``POSITION_CONFIG`` exclusively.
``CONFIG_TINY`` (e2e tests import by name) remains at module level.
"""

from src.shared.position_config import (
    DEFAULT_ENET_L1_RATIOS,
    PositionConfig,
    alpha_grid,
)

# Predict the 10 raw NFL stats that make up a D/ST's fantasy-point score.
# Fantasy points are computed post-prediction via
# ``src.shared.aggregate_targets.predictions_to_fantasy_points("DST", preds)``,
# which applies the linear coefficients (sacks×1, INT×2, ...) and the PA/YA
# tier bonuses in a single place.
_TARGETS = [
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
    "points_allowed",
    "yards_allowed",
]

# Rolling defense stats.
# NOTE: turnovers_L3 removed — exactly ints_L3 + fumble_rec_L3 (perfect
# linear dependency).
_SPECIFIC_FEATURES = [
    # Core production rolling windows
    "sacks_L3",
    "sacks_L5",  # Longer sack stability anchor
    "ints_L3",  # INTs separated — secondary quality signal
    "fumble_rec_L3",  # Fumble recoveries — more stochastic component
    "forced_fumbles_L3",  # Forced fumbles — pressure proxy
    "blocked_kicks_L5",  # Blocked kicks — rare, needs longer window
    "pts_allowed_L3",
    "pts_allowed_L5",
    "yards_allowed_L3",
    "yards_allowed_L5",
    "yards_allowed_ewma",
    "dst_pts_L3",
    "dst_pts_L5",
    "dst_pts_L8",  # Longer stability anchor
    # EWMA features (faster adaptation to regime changes)
    "pts_allowed_ewma",
    "dst_pts_ewma",
    # Momentum / trend indicators
    "sack_trend",
    "turnover_trend",
    "pts_allowed_trend",
    # Consistency metrics
    "pts_allowed_std_L3",
    "dst_scoring_std_L3",
]

_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "spread_line",
    "total_line",
    "opp_scoring_L3",
    "opp_scoring_L5",
    "opp_turnovers_L5",
    "opp_sacks_allowed_L5",
    # Opposing QB quality (isolates QB signal from team-level noise)
    "opp_qb_epa_L5",
    "opp_qb_int_rate_L5",
    "opp_qb_sack_rate_L5",
    "opp_qb_rush_yds_L5",
    "rest_days",
    "div_game",
    "is_dome",
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
]

_ALL_FEATURES = _SPECIFIC_FEATURES + _CONTEXTUAL_FEATURES

# Very-rare counts (mean 0.03-0.08, max 2 over 6K+ team-weeks). Empirical
# dispersion 0.98-1.07 with zero-excess ~0, so Poisson fits cleanly. Huber
# at delta=0.25 was effectively MSE in this range (scale-unaware,
# count-blind); Poisson NLL gives scale-aware gradients.
_POISSON_TARGETS = [
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
]


# ===========================================================================
# CONFIG_TINY — shrunk config for E2E smoke tests. 2 backbone layers x 8
# units, 1 epoch, no LightGBM, attention off by default for the baseline
# bit-identity test. Dedicated attention tests override ``train_attention_nn``
# to True and exercise the attention branch directly.
# ===========================================================================
CONFIG_TINY = {
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 1,
    "nn_head_hidden_overrides": None,
    "nn_non_negative_targets": set(_TARGETS),
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "train_lightgbm": False,
    "train_attention_nn": False,
    "ridge_pca_components": None,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "nn_log_every": 100,
}


# Single source of truth for downstream consumers (registry / pipeline / serving).
# DST has 10 raw-stat heads, no GATED_TARGETS, OPP_ATTN_KIND="offense" (it
# attends over opposing offenses, not defenses), and uses POISSON_TARGETS for
# the four very-rare counts — their loss weights diverge from the 2.0/δ rule
# per the loss_weights comments below.
POSITION_CONFIG = PositionConfig(
    name="DST",
    targets=_TARGETS,
    specific_features=_SPECIFIC_FEATURES,
    contextual_features=_CONTEXTUAL_FEATURES,
    all_features=_ALL_FEATURES,
    # No general features — D/ST bypasses the player-level feature pipeline.
    drop_features=set(),
    # Alpha grids tuned per-target magnitude:
    #   Sparse counts (TDs, safeties, blocked kicks, ST TDs) need a higher-floor
    #     grid because their means are <0.1 — weak L2 produces unstable fits.
    #   Regular counts (sacks, ints, fum_rec, FF) use a standard grid.
    #   Raw PA/YA span 0-55 / 0-600 and tolerate stronger L2.
    ridge_alpha_grids={
        "def_sacks": alpha_grid(-1, 3.5, 20),
        "def_ints": alpha_grid(-1, 3.5, 20),
        "def_fumble_rec": alpha_grid(-1, 3.5, 20),
        "def_fumbles_forced": alpha_grid(-1, 3.5, 20),
        "def_safeties": alpha_grid(0, 4, 15),
        "def_tds": alpha_grid(0, 4, 15),
        "def_blocked_kicks": alpha_grid(0, 4, 15),
        "special_teams_tds": alpha_grid(0, 4, 15),
        "points_allowed": alpha_grid(-1, 5, 20),
        "yards_allowed": alpha_grid(-1, 5, 20),
    },
    # 38 features (21 specific + 17 contextual) → 20 PCA components;
    # removes collinear dimensions before Ridge fit.
    ridge_pca_components=20,
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    nn_backbone_layers=[128, 64],
    nn_head_hidden=32,
    nn_head_hidden_overrides={
        # Sparse targets — smaller heads regularize, rare events don't need capacity.
        "def_safeties": 16,
        "def_tds": 16,
        "def_blocked_kicks": 16,
        "special_teams_tds": 16,
        # Raw-scale targets — wider head to learn the 0-55 / 0-600 magnitude.
        "points_allowed": 48,
        "yards_allowed": 48,
    },
    nn_dropout=0.30,  # Slightly higher — slows convergence, better generalization
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=3e-4,  # Lower — more exploration before convergence
    nn_weight_decay=3e-4,
    nn_epochs=300,
    nn_batch_size=128,
    nn_patience=35,
    head_losses={t: ("poisson_nll" if t in _POISSON_TARGETS else "mse") for t in _TARGETS},
    # Non-Poisson heads switched Huber -> MSE to chase the upper tail. Weight =
    # 1/delta (gradient-matched to the old 2.0/delta Huber at e~delta; half the
    # Huber weight). Without per-head scaling, PA (delta=5) and YA (delta=30)
    # would dominate the count heads. Poisson heads keep w ≈ 5.0 — picked so the
    # expected weighted per-sample PoissonNLL (~0.14-0.28 at these small lambdas)
    # sits in the same band as the MSE heads.
    loss_weights={
        "def_sacks": 1.0,  # 1 / 1.0 (MSE)
        "def_ints": 2.0,  # 1 / 0.5 (MSE)
        "def_fumble_rec": 2.0,
        "def_fumbles_forced": 2.0,
        "def_safeties": 5.0,  # Poisson NLL; lambda=0.030
        "def_tds": 5.0,  # Poisson NLL; lambda=0.084
        "def_blocked_kicks": 5.0,  # Poisson NLL; lambda=0.052
        "special_teams_tds": 5.0,  # Poisson NLL; lambda=0.044
        "points_allowed": 0.2,  # 1 / 5.0 (MSE)
        "yards_allowed": 0.0333,  # 1 / 30.0 (MSE)
    },
    # Deltas roughly match each target's typical variance. Very-rare
    # targets moved to Poisson NLL (see poisson_targets) and are absent
    # here — MultiTargetLoss picks the per-target loss by lookup, so
    # listing them would only add dead config.
    huber_deltas={
        "def_sacks": 1.0,
        "def_ints": 0.5,
        "def_fumble_rec": 0.5,
        "def_fumbles_forced": 0.5,
        "points_allowed": 5.0,
        "yards_allowed": 30.0,
    },
    poisson_targets=_POISSON_TARGETS,
    scheduler_type="cosine_warm_restarts",
    cosine_t0=30,  # Longer first cycle for wider backbone
    cosine_t_mult=2,
    cosine_eta_min=1e-5,
    # === Attention NN (game history variant) ===
    # Shares RB's attention architecture (d_model=32, n_heads=2, encoder
    # hidden=32) with heavier regularization; the 2026-06 batch/LR ablation
    # selected b2_lrlin for DST:
    # attn_lr=6e-4, attn_weight_decay=3e-4, attn_batch_size=256. DST
    # has far fewer rows than RB (32 teams x ~17 weeks vs every RB-game),
    # so the heavier regularization + smaller batch + gentler step size
    # are deliberate to avoid overfitting on the smaller sample. No gating
    # per design. The attention branch learns its own temporal
    # representation from per-game defensive + opponent history, so
    # rolling/EWMA/trend features are stripped from its static input.
    train_attention_nn=True,
    # Learned embedding for empty-history (season-opener) rows — reduces the
    # attention NN's opener over-prediction (player branch only; see
    # src/shared/neural_net.py and src/analysis/rb_lgbm_disagreement_findings.md).
    attn_no_history_embedding=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=32,  # 2-layer game encoder before attention
    attn_max_seq_len=17,
    attn_positional_encoding=True,
    attn_dropout=0.05,
    attn_lr=6e-4,
    attn_weight_decay=3e-4,
    attn_batch_size=256,
    attn_patience=20,
    attn_cosine_eta_min=2e-5,
    # Per-game stats fed into the attention sequence. The 10 raw target
    # stats plus 4 opponent-side per-game values (not rolling) so attention
    # can weigh recent games against recent opponent strength. Derived
    # combos (defensive_production, st_production, fantasy_points) are
    # intentionally excluded — linear functions of the raw stats already
    # in the sequence (would add collinear columns).
    attn_history_stats=[
        # Own raw defensive + ST stats (mirror the 10 targets)
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_fumbles_forced",
        "def_safeties",
        "def_tds",
        "def_blocked_kicks",
        "special_teams_tds",
        "points_allowed",
        "yards_allowed",
        # Per-game opponent context (not pre-rolled)
        "opp_scoring",
        "opp_fumbles",
        "opp_interceptions",
        "opp_qb_epa",
    ],
    # Explicit whitelist of static-branch features for the attention NN.
    # DST doesn't use the category-dict shape that QB/RB/WR/TE do, so
    # enumerate the allowed columns directly. All SPECIFIC_FEATURES
    # (rolling/ewma/trend/std) and the opp_*_L{3,5} columns are excluded —
    # the attention branch already sees that signal via attn_history_stats.
    # Prior-season means stay (different season than the lookback window).
    attn_static_features=[
        "is_home",
        "week",
        "spread_line",
        "total_line",
        "rest_days",
        "div_game",
        "is_dome",
        "prior_season_dst_pts_avg",
        "prior_season_pts_allowed_avg",
    ],
    attn_project_kv=False,
    attn_gated_fusion=False,
    attn_gated=False,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    # Per-game opp-OFFENSE stats fed to the second attention branch. DST
    # is the one position whose parallel branch attends over the opposing
    # offense (not defense) — the per-game `opp_scoring`/`opp_qb_epa`
    # entries in attn_history_stats are point-in-time signals about what
    # the opp did *vs this DST*; the parallel branch carries season-form.
    opp_attn_history_stats=[
        "off_pass_yards",
        "off_pass_tds",
        "off_rush_yards",
        "off_rush_tds",
        "off_ints",
        "off_fumbles_lost",
        "off_pts_scored",
    ],
    opp_attn_max_seq_len=17,
    # DST is the only position using "offense"; QB/RB/WR/TE default to
    # "defense" (their parallel branch attends over the opposing defense).
    opp_attn_kind="offense",
    train_lightgbm=True,
    lgbm_n_estimators=300,
    lgbm_learning_rate=0.03,
    lgbm_num_leaves=15,
    lgbm_max_depth=-1,
    lgbm_subsample=0.75,
    lgbm_colsample_bytree=0.8,
    lgbm_reg_lambda=2.0,
    lgbm_reg_alpha=0.1,
    lgbm_min_child_samples=25,
    lgbm_min_split_gain=0.0,
    # Single global objective applied to every LGBMRegressor head
    # (LightGBMMultiTarget takes one objective, not per-target). This
    # diverges from the attention NN's head_losses, which fit poisson_nll
    # for the four sparse counts (def_safeties/def_tds/def_blocked_kicks/
    # special_teams_tds). The divergence is intentional, not a bug: the
    # Poisson rationale above (scale-aware gradients where Huber@delta=0.25
    # collapses to ~MSE) is an NN-loss argument; gradient-boosted trees on
    # those lambda~0.03-0.08 targets split on the raw count regardless of
    # objective, and the ensemble averages the two estimators' outputs
    # rather than requiring matched objectives. Threading per-target
    # Poisson into LGBM was evaluated and is not worth the complexity here.
    # Switched "huber" → "regression" (L2/MSE) to chase the elite tail.
    lgbm_objective="regression",
    accepts_dataframes=False,
    cpu_only=True,
)
