"""RB hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers (training pipeline, serving, tests) read from the
exported ``POSITION_CONFIG`` exclusively. ``CONFIG_TINY`` and
``ATTN_STATIC_CATEGORIES`` remain at module level — the former is used by
direct-import e2e tests, the latter by the attention-static whitelist test.
"""

from src.shared.position_config import (
    DEFAULT_ATTN_STATIC_CATEGORIES,
    DEFAULT_ENET_L1_RATIOS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

_TARGETS = [
    "rushing_tds",
    "receiving_tds",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fumbles_lost",
]

# Dropped weighted_opportunities_L3 (r=0.940 with opportunity_index_L3).
_SPECIFIC_FEATURES = [
    "yards_per_carry_L3",
    "reception_rate_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "rushing_first_down_rate_L3",
    "receiving_first_down_rate_L3",
    "yac_per_reception_L3",
    "receiving_epa_per_target_L3",
    "air_yards_per_target_L3",
    "career_carries",
    "team_rb_carry_hhi_L3",
    "team_rb_target_hhi_L3",
    "opportunity_index_L3",
]

# Per-cell prior_season drops from the multicollinearity audit
# (analysis_output/rb_feature_audit.json, docs/rb_feature_history.md):
#   * receptions/{mean,std,max}: r=0.937–0.982 with the matching targets
#     aggregates (catch rate stable across seasons). YPC residual restored
#     via `prior_season_mean_yards_per_carry` (see rb/features.py).
#   * rushing_yards/{mean,std,max}: r=0.943–0.963 with carries aggregates.
#   * std_{receiving_yards, fantasy_points}: r=0.91–0.94 with the max.
#   * mean_/max_fantasy_points (PR #191): per-target prior-season aggregates
#     (carries, targets, snap_pct, plus new total_touchdowns) plus the
#     volume×rate features above carry the FP signal that PR #190 was
#     standing in for.
_PRIOR_SEASON_DROPS = {
    "prior_season_mean_receptions",
    "prior_season_std_receptions",
    "prior_season_max_receptions",
    "prior_season_mean_rushing_yards",
    "prior_season_std_rushing_yards",
    "prior_season_max_rushing_yards",
    "prior_season_std_receiving_yards",
    "prior_season_std_fantasy_points",
    "prior_season_mean_fantasy_points",
    "prior_season_max_fantasy_points",
}

_ROLLING_STATS = [
    "fantasy_points",
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

_INCLUDE_FEATURES = {
    # L5 rolling means/stds/maxes are excluded from the ``rolling`` category
    # by design: every L5 ``mean``/``std``/``max`` lands at r=0.96-0.98 with
    # the matching L3/L8 variants (and ``carry_share_L5`` r=0.984 with the L3
    # equivalent — see the multicollinearity audit at
    # analysis_output/rb_feature_audit.json and docs/rb_feature_history.md).
    # Keeping only L3+L8 spans the short / mid horizon without inflating the
    # condition number. ``rolling_min_fantasy_points_L5`` survives because the
    # ``min`` aggregate is decorrelated from mean/std/max (it tracks worst-case
    # rather than central tendency) and adds floor signal the others miss.
    "rolling": [
        col
        for stat in _ROLLING_STATS
        for w in [3, 8]
        for col in (
            [f"rolling_{a}_{stat}_L{w}" for a in ["mean", "std", "max"]]
            + ([f"rolling_min_{stat}_L{w}"] if stat == "fantasy_points" else [])
        )
    ]
    + ["rolling_min_fantasy_points_L5"],
    "prior_season": [
        f"prior_season_{a}_{stat}"
        for stat in _ROLLING_STATS
        for a in ["mean", "std", "max"]
        if f"prior_season_{a}_{stat}" not in _PRIOR_SEASON_DROPS
    ]
    + [
        # prior_season_total_touchdowns: full-season sum of rushing_tds +
        # receiving_tds. Built in src.features.engineer.build_features.
        "prior_season_total_touchdowns",
        # Derived ratios that restore the player-specific rate signal lost
        # when PR #190 dropped the receptions / rushing_yards aggregates.
        # Built in src.rb.features._compute_features.
        "prior_season_mean_catch_rate",
        "prior_season_mean_yards_per_carry",
        # Decomposed restoration of the fantasy-points signal PR #191
        # dropped. total_yards is the ~90%-of-FP volume×efficiency
        # component; games_played gives the model the count needed to
        # convert per-game means into season totals; mean_fumbles_lost is
        # the negative-FP component absent from every other prior_season
        # aggregate.
        "prior_season_total_yards",
        "prior_season_games_played",
        "prior_season_mean_fumbles_lost",
        # Red-zone usage carried into season S from S-1. Encodes goal-line
        # opportunity (carries inside the 20 + targets inside the 20)
        # which the volume-only aggregates can't separate from open-field
        # production. Sourced from PBP via src.data.redzone_pbp; targets
        # the rushing_tds (+0.080 vs Ridge) and receiving_tds (+0.035)
        # gaps the rejected hurdle_poisson fix would have closed.
        "prior_season_total_redzone_touches",
        "prior_season_mean_redzone_touches_per_game",
        # ff_opportunity expected-production prior (static "opportunity" signal,
        # leakage-safe S-1 mean). Built in src.features.engineer from the
        # per-game *_exp columns merged by src.data.external_sources.
        "prior_season_mean_total_fantasy_points_exp",
        "prior_season_mean_rush_yards_gained_exp",
        "prior_season_mean_receptions_exp",
    ],
    "ewma": [],
    "trend": ["trend_fantasy_points", "trend_targets", "trend_carries", "trend_snap_pct"],
    # Audit drops: target_share_L5 (r=0.966 with L3), carry_share_L5
    # (r=0.984 with L3), carry_share_L3 (r=0.982 with team_rb_carry_share_L3).
    "share": [
        "target_share_L3",
        "snap_pct",
        "air_yards_share",
    ],
    # Audit drops:
    #   * opp_fantasy_pts_allowed_to_pos (VIF 193) — sum ≈ rush + recv;
    #     components carry directional info the sum collapses.
    #   * opp_def_rank_vs_pos — rank() of the dropped sum (Spearman 1.0).
    "matchup": [
        "opp_rush_pts_allowed_to_pos",
        "opp_recv_pts_allowed_to_pos",
    ],
    "defense": [
        "opp_def_sacks_L5",
        "opp_def_pass_yds_allowed_L5",
        "opp_def_pass_td_allowed_L5",
        "opp_def_ints_L5",
        "opp_def_rush_yds_allowed_L5",
        "opp_def_pts_allowed_L5",
    ],
    # is_returning_from_absence (r=0.934 with days_rest) dropped — the
    # indicator is essentially `days_rest > 13`; days_rest carries the
    # magnitude.
    "contextual": [
        "is_home",
        "week",
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
        # Role-inheritance (next-man-up): current-week vacancy when a higher-snap
        # RB is Out/Doubtful. Built in src.features.engineer._build_inheritance_features;
        # validated A/B src/tuning/ab_history_token.py (strong on the inheritor subgroup,
        # all models). "contextual" is an ATTN_STATIC category, so these feed the NN static
        # branch too (NOT attn_history_stats — redundant there, tested-rejected).
        "is_top_available",
        "inherited_opportunity",
    ],
    # implied_team + implied_opp encodes both game total and spread direction
    # without the perfect collinearity of keeping total_line alongside.
    # is_dome: dome premium on receiving (r=0.023 receiving_floor).
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome", "rest_advantage"],
    "specific": _SPECIFIC_FEATURES,
}

ATTN_STATIC_CATEGORIES = DEFAULT_ATTN_STATIC_CATEGORIES


# === Tiny config for end-to-end smoke tests ===
# Shrunk overrides layered on top of ``build_pipeline_config("RB",
# POSITION_CONFIG)`` by ``tests/_pipeline_e2e_utils.py::build_tiny_config``.
# The generic NN/scheduler knobs (``nn_backbone_layers``, ``nn_head_hidden``,
# ``nn_epochs``, ``nn_batch_size``, ``nn_patience``, etc.) live in
# ``tests/_pipeline_e2e_utils.py::_TINY_OVERRIDES`` so they're shared across
# every position — duplicating them here would risk drift if a future tightening
# of the test budget bumps one and not the other. Only the RB-specific shrinks
# (per-target Ridge alpha grids, equalized loss weights, Huber deltas) stay
# here.
CONFIG_TINY = {
    "targets": _TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in _TARGETS},
    # Yard heads are MSE (1/delta) to match production; others equalized at 1.0.
    "loss_weights": {
        **{t: 1.0 for t in _TARGETS},
        "rushing_yards": 0.0667,
        "receiving_yards": 0.0667,
    },
    "huber_deltas": {"rushing_yards": 15.0, "receiving_yards": 15.0},
    # Exercise the receptions head-width override plumbing in the e2e shard:
    # production sizes this head wider for the hurdle-NegBin outputs; tiny just
    # needs it non-None so the override path runs. _TINY_OVERRIDES forces None
    # and CONFIG_TINY is merged last, so this restores a non-None override. (#576)
    "nn_head_hidden_overrides": {"receptions": 8},
}


# Single source of truth for downstream consumers (registry / pipeline / serving).
POSITION_CONFIG = PositionConfig(
    name="RB",
    # Relaxed from the global 6 → 1: the train-only filter was stripping low-volume
    # RB seasons the model is still scored on; relaxing lowers cold-start MAE (LGBM
    # would_filter +0.175±0.021, 8-seed) with no kept regression. See TODO.md.
    min_games_per_season=1,
    targets=_TARGETS,
    specific_features=_SPECIFIC_FEATURES,
    include_features=_INCLUDE_FEATURES,
    # Raw-stat grids — yards need broader high end (large variance vs counts);
    # receptions get a finer 20-point grid because Ridge cares more.
    ridge_alpha_grids={
        "rushing_tds": alpha_grid(-1, 4, 15),
        "receiving_tds": alpha_grid(-1, 4, 15),
        "rushing_yards": alpha_grid(-2, 3, 15),
        "receiving_yards": alpha_grid(-2, 3, 15),
        "receptions": alpha_grid(-2, 2.5, 20),
        "fumbles_lost": alpha_grid(-1, 4, 15),
    },
    # PCR: 80 components retains 99.8% variance, drops condition number from
    # 1.8e8 to 49.8. Both yard targets improve ~0.002 MAE.
    ridge_pca_components=80,
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    # === Neural Net ===
    # [128, 64] two-layer backbone — single [128] was underfitting (early stop
    # epoch 54, flat val loss from epoch 3). Added depth + larger heads + less
    # regularization.
    nn_backbone_layers=[128, 64],
    nn_head_hidden=48,
    nn_dropout=0.15,
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=1e-3,
    nn_weight_decay=5e-5,
    nn_epochs=300,
    nn_batch_size=256,
    nn_patience=30,
    # Larger head for the hurdle-NegBin reception head (two value outputs:
    # mu + log_alpha). TD heads moved to plain Poisson NLL (dispersion
    # ~1.03-1.17, no zero-excess) and no longer need the extra capacity
    # the Huber+gate setup did.
    nn_head_hidden_overrides={"receptions": 64},
    # === Per-head loss families ===
    # TDs + fumbles: plain Poisson NLL (dispersion 1.03-1.17, no zero-excess).
    # Receptions: zero-truncated NegBin-2 hurdle (overdispersed + zero-excess).
    # Variant E (hurdle_poisson on TDs+fumbles_lost) was tested+rejected —
    # wins per-target MAE but regresses FP MAE +0.163 vs current. See
    # TODO.md "[TESTED, REJECTED] RB sparse-count hurdle_poisson" and
    # docs/ARCHITECTURE.md D-entry.
    head_losses={
        "rushing_tds": "poisson_nll",
        "receiving_tds": "poisson_nll",
        "rushing_yards": "mse",
        "receiving_yards": "mse",
        "receptions": "hurdle_negbin",
        "fumbles_lost": "poisson_nll",
    },
    # Yards heads switched Huber -> MSE to stop under-projecting elite RBs (the
    # biggest tail bias measured, ~-2 FP). Weight = 1/delta (gradient-matched to
    # the old 2.0/delta Huber weighting at e~delta; half the Huber weight).
    # Poisson NLL heads stay 1.0; hurdle_negbin value loss scaled internally.
    loss_weights={
        "rushing_tds": 1.0,
        "receiving_tds": 1.0,
        "rushing_yards": 0.0667,  # 1 / 15 (MSE)
        "receiving_yards": 0.0667,  # 1 / 15 (MSE)
        "receptions": 1.0,
        "fumbles_lost": 1.0,
    },
    # Characteristic error scale the MSE weights (1/delta) derive from; ignored
    # at loss time by the MSE heads.
    huber_deltas={
        "rushing_yards": 15.0,
        "receiving_yards": 15.0,
    },
    # Hurdle gate on receptions + BCE gate on each TD head — Variant C from
    # src/tuning/ablate_rb_gate.py. Gate brings rushing_tds MAE 0.329→0.304
    # without giving up the hurdle-NegBin reception win.
    gated_targets=["receptions", "rushing_tds", "receiving_tds"],
    scheduler_type="cosine_warm_restarts",
    cosine_t0=40,
    cosine_t_mult=2,
    cosine_eta_min=1e-5,
    # === Attention NN (game history variant) ===
    # d_model=32 (proven baseline), n_heads=2 (larger overfits on 15K samples).
    train_attention_nn=True,
    attn_d_model=32,
    attn_n_heads=2,
    # 2-layer nonlinear game encoder (Linear→ReLU→LayerNorm→Linear→ReLU)
    # so each game is a richer event embedding before attention.
    attn_encoder_hidden_dim=32,
    attn_max_seq_len=17,
    # K/V projections disabled — at d_model=32 the 2K extra params hurt
    # optimization more than they help (tested: 4.330 vs 4.228 without).
    attn_project_kv=False,
    attn_positional_encoding=True,
    attn_gated_fusion=False,
    # Season openers (empty in-season history) otherwise get a zero/constant
    # pooled history vector and over-predict off the static prior-season branch
    # (npg==0 bias +1.13). Give that case a learned per-target embedding instead.
    attn_no_history_embedding=True,
    attn_dropout=0.05,
    attn_lr=1.4142135623730952e-3,
    attn_weight_decay=5e-5,
    attn_batch_size=512,
    attn_patience=20,
    attn_cosine_eta_min=1.4142135623730952e-5,
    # Per-game stats fed into the attention sequence. fantasy_points
    # intentionally excluded — its scoring components are already in the
    # sequence. Game-script context + team box score are per-historical-
    # game raw context (not rolling/EWMA/trend); the per-game values land
    # on the history tokens here, while the *current-game* counterparts
    # (is_home, days_rest, implied_team_total, implied_opp_total) flow
    # into attn_static_features via the `contextual` / `weather_vegas`
    # INCLUDE_FEATURES categories.
    attn_history_stats=[
        "rushing_yards",
        "receiving_yards",
        "rushing_tds",
        "receiving_tds",
        "carries",
        "targets",
        "receptions",
        "fumbles_lost",
        "snap_pct_raw",
        "rushing_first_downs",
        "receiving_first_downs",
        "game_carry_share",
        "game_target_share",
        "game_carry_hhi",
        "game_target_hhi",
        # Game-script context (already merged onto every player-week row).
        "implied_team_total",
        "implied_opp_total",
        "is_home",
        "days_rest",
        # Team box score for the historical game (merged by
        # src.shared.team_box_score.merge_team_box_score_features).
        "team_pass_attempts",
        "team_completions",
        "team_passing_yards",
        "team_rush_attempts",
        "team_rushing_yards",
        "team_points_scored",
        "team_turnovers",
        "opp_team_points_scored",
        # Per-game red-zone & goal-line touch counts. Sourced from PBP via
        # src.data.redzone_pbp.reconstruct_redzone_from_pbp. Targets
        # rushing_tds (+0.080 MAE vs Ridge) and receiving_tds (+0.035)
        # gaps the rejected hurdle_poisson loss-family fix couldn't close
        # without regressing aggregate FP MAE.
        "redzone_carries",
        "redzone_targets",
        "inside10_carries",
        "inside5_carries",
        "redzone_target_share",
        # Per-game ff_opportunity expected stats (rushing + receiving),
        # merged by src.data.external_sources. Leakage-safe via
        # build_game_history_arrays (prior in-season games only).
        "rush_yards_gained_exp",
        "rush_touchdown_exp",
        "rec_yards_gained_exp",
        "rec_touchdown_exp",
        "receptions_exp",
    ],
    attn_static_features=derive_attn_static_features(_INCLUDE_FEATURES, ATTN_STATIC_CATEGORIES),
    # opp-defense attention branch disabled 2026-06-15: a 24-seed stacked Batch
    # A/B (src.tuning.ab_opp_def) showed it does not improve the attention NN
    # (hurts QB/RB/TE, WR within noise) — see ADR-0004 changelog. Empty ⇒ no opp
    # tensors ⇒ single-branch NN (pre-#123); static opp_def_*_L5 still feeds
    # Ridge/LGBM via INCLUDE_FEATURES["defense"].
    opp_attn_history_stats=[],
    attn_gated=True,
    attn_gate_hidden=16,
    attn_gate_weight=1.0,
    # === LightGBM (Optuna retune, 50 trials, CV MAE 4.5244) ===
    # Switched "huber" → "regression" (L2/MSE) to chase the elite tail; was
    # flipped from "fair" → "huber" in PR 3 LGBM unification. Holdout vs old
    # fair: Total MAE 4.479 → 4.155 (-0.325), big yards improvements
    # dominate small TD regression.
    train_lightgbm=True,
    lgbm_n_estimators=2000,
    lgbm_learning_rate=0.0918744,
    lgbm_num_leaves=16,
    lgbm_max_depth=-1,
    lgbm_subsample=0.547198,
    lgbm_colsample_bytree=0.783541,
    lgbm_reg_lambda=8.2532,
    lgbm_reg_alpha=4.91934,
    lgbm_min_child_samples=20,
    lgbm_min_split_gain=0.315457,
    lgbm_objective="regression",
    # === RB-only TD model variants ===
    # Production uses ``td_model_type="gated_ordinal"`` below — only
    # ``gated_ordinal_targets`` is read by ``_rb_classification_targets`` in
    # ``src/shared/position_pipeline.py``. ``two_stage_targets`` and
    # ``ordinal_targets`` are kept (rather than deleted) for three reasons:
    #   1. They preserve tuned hyperparameters for the alternate
    #      ``td_model_type`` values (``two_stage`` / ``ordinal``) so switching
    #      the selector for a manual Ridge-side comparison doesn't require
    #      re-deriving them. (Note: ``src/tuning/ablate_rb_gate.py`` does NOT
    #      touch ``td_model_type`` or these blocks — it ablates the attention
    #      NN's sparse-count heads by flipping ``gated_targets`` / ``head_losses``
    #      / ``loss_weights``, an orthogonal axis.)
    #   2. ``tests/rb/test_models.py`` reads ``POSITION_CONFIG.ordinal_targets``
    #      directly (``test_uses_config_from_rb_ordinal_targets``).
    #   3. Future ablations should not have to re-derive the hyperparameters.
    # Drift risk: a model-only change here that doesn't re-tune the alternate
    # variants will leave the dormant blocks stale relative to
    # ``gated_ordinal_targets``.
    # Two-stage zero-inflated TD models: both rushing_tds and receiving_tds.
    two_stage_targets={
        "rushing_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
        "receiving_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
    },
    # Ordinal classification over raw TD counts {0,1,2,3+} per TD target.
    ordinal_targets={
        "rushing_tds": {"type": "ordinal", "class_values": [0, 1, 2, 3], "alpha": 1.0},
        "receiving_tds": {"type": "ordinal", "class_values": [0, 1, 2, 3], "alpha": 1.0},
    },
    # Gated ordinal: binary gate + ordinal on positives, per TD target.
    gated_ordinal_targets={
        "rushing_tds": {
            "type": "gated_ordinal",
            "class_values": [0, 1, 2, 3],
            "alpha": 1.0,
            "clf_C": 0.001,
            "threshold": 0.5,
        },
        "receiving_tds": {
            "type": "gated_ordinal",
            "class_values": [0, 1, 2, 3],
            "alpha": 1.0,
            "clf_C": 0.001,
            "threshold": 0.5,
        },
    },
    td_model_type="gated_ordinal",
    accepts_dataframes=True,
    cpu_only=False,
)


# === td_model_type validation ===
# The TD-model dispatcher in ``src/shared/position_pipeline.py``
# (``_rb_classification_targets`` / ``_rb_two_stage_targets``) falls through
# to an empty dict when ``td_model_type`` doesn't match any of the recognised
# variants. That fall-through silently degrades the model to plain ridge on
# the TD targets — a typo like ``gated_oridinal`` produces no warning at
# import time and is only detectable by inspecting the per-target backtest
# breakdown. Re-validate at module import so the typo raises immediately.
_VALID_TD_MODEL_TYPES = frozenset({"ridge", "two_stage", "ordinal", "gated_ordinal"})
if POSITION_CONFIG.td_model_type not in _VALID_TD_MODEL_TYPES:
    raise ValueError(
        f"src.rb.config: td_model_type={POSITION_CONFIG.td_model_type!r} is not one of "
        f"{sorted(_VALID_TD_MODEL_TYPES)}. The dispatcher in src/shared/position_pipeline.py "
        f"silently degrades to plain ridge on unknown variants; re-check for typos "
        f"(e.g. 'gated_oridinal' -> 'gated_ordinal')."
    )
