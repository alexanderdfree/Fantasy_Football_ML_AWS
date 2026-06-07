"""QB hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers (training pipeline, serving, tests) read from the
exported ``POSITION_CONFIG`` exclusively. ``CONFIG_TINY`` and
``ATTN_STATIC_CATEGORIES`` remain at module level — the former is picked up
by ``tests/_pipeline_e2e_utils.build_tiny_config('QB')`` (and by
``src/shared/run_pipeline_factory.py``'s ``--tiny`` CLI path) via a
``getattr(config_mod, "CONFIG_TINY", None)`` lookup; the latter by the
attention-static whitelist test. Private helper variables stay scoped to
construction.
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

_ROOKIE_PHASE_FEATURES = ["rookie_early"]

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
    ]
    + [
        # ff_opportunity expected-production prior (static "opportunity" signal,
        # leakage-safe S-1 mean). Built in src.features.engineer from the
        # per-game *_exp columns merged by src.data.external_sources.
        "prior_season_mean_total_fantasy_points_exp",
        "prior_season_mean_pass_yards_gained_exp",
        "prior_season_mean_pass_touchdown_exp",
        # ESPN QBR prior (static "QB quality" signal). Career-cumulative is
        # deliberately kept in the history domain (see tests/test_attn_static_columns.py);
        # the prior-season mean is the static-legal shape.
        "prior_season_mean_qbr_total",
        "prior_season_mean_pts_added",
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
        # Contract value — team-investment / expected-role prior. Static
        # player-season state (active contract as-of season), merged by
        # src.data.external_sources. apy_cap_pct is cross-era-normalized.
        "contract_apy_cap_pct",
        "contract_guaranteed",
        "contract_years_remaining",
        "contract_age",
        # Roster/data-availability phase indicators. These are non-temporal
        # static context, used to cancel the measured early-rookie QB
        # overprediction without reintroducing draft-capital pedigree features.
        *_ROOKIE_PHASE_FEATURES,
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


# === Tiny config for end-to-end smoke tests ===
# Shrunk overrides layered on top of ``build_pipeline_config("QB",
# POSITION_CONFIG)`` by ``tests/_pipeline_e2e_utils.py::build_tiny_config``.
# The generic NN/scheduler knobs (``nn_backbone_layers``, ``nn_head_hidden``,
# ``nn_epochs``, ``nn_batch_size``, ``nn_patience``, etc.) live in
# ``tests/_pipeline_e2e_utils.py::_TINY_OVERRIDES`` so they're shared across
# every position. Only the QB-specific shrinks stay here:
#   * per-target Ridge alpha grids collapsed to one alpha,
#   * loss weights mirroring the production 2.0/delta rebalance on Huber
#     heads + 1.0 on Poisson NLL heads (without these the count heads
#     collapse to mean under yards-dominated gradients — see
#     ``loss_weights`` comment lower in this file),
#   * Huber deltas matching production so the test exercises the same
#     loss-head shape as the real run.
# Attention + LightGBM disabled (they're already disabled by ``_TINY_OVERRIDES``
# but spelling them here makes the override explicit when reading config_tiny
# in isolation).
CONFIG_TINY = {
    "targets": _TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in _TARGETS},
    "loss_weights": {
        "passing_yards": 0.04,  # 1 / 25 (MSE)
        "rushing_yards": 0.0667,  # 1 / 15 (MSE)
        "passing_tds": 1.0,  # Poisson NLL
        "rushing_tds": 1.0,  # Poisson NLL
        "interceptions": 1.0,  # Poisson NLL
        "fumbles_lost": 1.0,  # Poisson NLL
    },
    "huber_deltas": {"passing_yards": 25.0, "rushing_yards": 15.0},
    "train_attention_nn": False,
    "train_lightgbm": False,
}


# Single source of truth for downstream consumers (registry / pipeline / serving).
# Read from POSITION_CONFIG exclusively; no module-level UPPERCASE constants.
POSITION_CONFIG = PositionConfig(
    name="QB",
    targets=_TARGETS,
    # ``specific_features`` is the set add_specific_features owns/fills and
    # keys into the feature cache. Rookie phase is categorized as contextual
    # above so the attention static branch may consume it.
    specific_features=_INCLUDE_FEATURES["specific"] + _ROOKIE_PHASE_FEATURES,
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
        "passing_yards": "mse",
        "rushing_yards": "mse",
        "passing_tds": "poisson_nll",
        "rushing_tds": "poisson_nll",
        "interceptions": "poisson_nll",
        "fumbles_lost": "poisson_nll",
    },
    # Yards heads switched Huber -> MSE to stop discounting the elite upper tail
    # (Huber's flat large-error gradient systematically under-projected top QBs).
    # Weight = 1/delta (gradient-matched to the old 2.0/delta Huber weighting at
    # the characteristic error e~delta; half the Huber weight since MSE's 2e
    # gradient replaces Huber's flat delta). Poisson NLL heads stay at 1.0.
    loss_weights={
        "passing_yards": 0.04,  # 1 / 25  (MSE)
        "rushing_yards": 0.0667,  # 1 / 15  (MSE)
        "passing_tds": 1.0,  # Poisson NLL
        "rushing_tds": 1.0,  # Poisson NLL
        "interceptions": 1.0,  # Poisson NLL
        "fumbles_lost": 1.0,  # Poisson NLL
    },
    # Retained as the characteristic error scale the MSE weights (1/delta) are
    # derived from; MSE heads ignore the delta at loss time (count heads are
    # Poisson NLL).
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
    # Learned embedding for empty-history (season-opener) rows — reduces the
    # attention NN's opener over-prediction (player branch only; see
    # src/shared/neural_net.py and src/analysis/rb_lgbm_disagreement_findings.md).
    attn_no_history_embedding=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=0,
    attn_max_seq_len=17,
    attn_positional_encoding=True,
    attn_dropout=0.05,
    attn_lr=2e-3,
    attn_weight_decay=5e-5,
    attn_batch_size=512,
    attn_patience=20,
    attn_cosine_eta_min=2e-5,
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
        "snap_pct_raw",
        "sacks",
        "sack_yards",
        # Per-game ff_opportunity expected stats (modeled from in-game
        # opportunity) + per-game ESPN QBR. Merged by src.data.external_sources;
        # leakage-safe via build_game_history_arrays (prior in-season games only).
        "pass_yards_gained_exp",
        "pass_touchdown_exp",
        "pass_interception_exp",
        "rush_yards_gained_exp",
        "rush_touchdown_exp",
        "qbr_total",
        "pts_added",
        # Per-game game-script context + realized final score, merged onto
        # every player-week row by build_position_features (schedule +
        # team_box_score merges). Lets attention learn this QB's conditional
        # response to matchup/script — the boom/bust signal the player-only
        # history was blind to. team_pass_attempts/team_passing_yards are
        # omitted (near-collinear with the QB's own attempts/passing_yards
        # already in this sequence); team_rush_* carry the run/pass game-script
        # the QB's own line doesn't. Mirrors RB; current-week counterparts live
        # on the static branch. Leakage-safe via build_game_history_arrays.
        "implied_team_total",
        "implied_opp_total",
        "is_home",
        "days_rest",
        "team_points_scored",
        "opp_team_points_scored",
        "team_rush_attempts",
        "team_rushing_yards",
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
    # PR #870 switched all six positions (QB included) to MSE (`regression`,
    # see lgbm_objective below). Historical note: QB previously kept LightGBM
    # `fair` — a 50-trial Huber retune had regressed QB total MAE 6.269 -> 6.479
    # (+0.210 pts/game, passing_yards 66.1 -> 71.2) because LightGBM's Huber
    # alpha=0.9 quantile puts ~90% of QB passing_yards residuals in the
    # quadratic zone, while Fair's smooth log-curvature downweighted the heavy
    # tail. The #870 unification to MSE supersedes that Fair-vs-Huber tuning.
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
    # Switched fair -> regression (L2/MSE): chase the elite upper tail instead of
    # the robust/outlier-tolerant fair loss that under-projected top scorers.
    lgbm_objective="regression",
    accepts_dataframes=True,
    cpu_only=False,
)
