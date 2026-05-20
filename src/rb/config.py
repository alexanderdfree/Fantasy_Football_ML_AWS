from src.shared.position_config import (
    DEFAULT_ENET_L1_RATIOS,
    DEFAULT_OPP_ATTN_MAX_SEQ_LEN,
    DEFAULT_OPP_DEF_HISTORY_STATS,
    PositionConfig,
    alpha_grid,
    derive_attn_static_features,
)

# === RB Raw-Stat Targets ===
TARGETS = [
    "rushing_tds",
    "receiving_tds",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fumbles_lost",
]

# === RB-Specific Features ===
# Dropped in the audit (analysis_output/rb_feature_audit.json,
# docs/rb_feature_history.md):
#   - weighted_opportunities_L3: r=0.940 with opportunity_index_L3 (the share
#     normalises out team-level usage; the raw count duplicates that signal
#     scaled by team usage). Compute block also removed from rb/features.py.
SPECIFIC_FEATURES = [
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

# === RB Feature Whitelist ===
# Explicit include list — new columns must be opted in, preventing silent leakage.
_ROLLING_STATS = [
    "fantasy_points",
    "targets",
    "receptions",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "snap_pct",
]

# Per-cell drops from the multicollinearity audit
# (analysis_output/rb_feature_audit.json, docs/rb_feature_history.md):
#   * receptions/{mean,std,max}: r=0.937–0.982 with the matching targets
#     aggregates (catch rate is stable across seasons, so prior-season
#     receptions ≈ targets × constant). The catch-rate residual is restored
#     via the derived `prior_season_mean_catch_rate` (see rb/features.py).
#   * rushing_yards/{mean,std,max}: r=0.943–0.963 with the matching carries
#     aggregates (yards/carry varies < 30% YoY). YPC residual is restored
#     via `prior_season_mean_yards_per_carry`.
#   * std_{receiving_yards, fantasy_points}: r=0.91–0.94 with the matching
#     max aggregate; max is the more interpretable shape stat for skewed
#     yardage distributions.
#   * mean_/max_fantasy_points (PR #191): per-target prior-season aggregates
#     (carries, targets, snap_pct, plus the new total_touchdowns) plus the
#     volume×rate features above carry the signal that prior_season_fp was
#     standing in for. fp is a position-scoring sum; the components are
#     more directly informative for prediction.
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

INCLUDE_FEATURES = {
    # L3/L8 only — all L5 dropped (>0.97 corr with L3/L8).
    # min variant only exists for fantasy_points (kept at all windows).
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
    # Standard mean/std/max aggregates over _ROLLING_STATS, minus the cells
    # filtered by _PRIOR_SEASON_DROPS, plus appended custom features:
    #   * prior_season_total_touchdowns: full-season sum of rushing_tds +
    #     receiving_tds. Built in src.features.engineer.build_features.
    #   * prior_season_mean_catch_rate / mean_yards_per_carry: derived
    #     ratios that restore the player-specific rate signal lost when
    #     PR #190 dropped the receptions / rushing_yards aggregates. Built
    #     in src.rb.features._compute_features.
    #   * prior_season_total_yards / games_played / mean_fumbles_lost
    #     (PR #192): decomposed restoration of the fantasy-points signal
    #     PR #191 dropped. The post-#191 EC2 retrain showed a +0.072 MAE
    #     Attention NN regression because the FP aggregates were carrying
    #     signal the decomposed features (catch_rate, YPC, total_touchdowns)
    #     didn't fully recover. These three close that gap orthogonally:
    #     total_yards is the ~90%-of-FP volume×efficiency component;
    #     games_played gives the model the count needed to convert per-game
    #     means into season totals; mean_fumbles_lost is the negative-FP
    #     component absent from every other prior_season aggregate.
    "prior_season": [
        f"prior_season_{a}_{stat}"
        for stat in _ROLLING_STATS
        for a in ["mean", "std", "max"]
        if f"prior_season_{a}_{stat}" not in _PRIOR_SEASON_DROPS
    ]
    + [
        "prior_season_total_touchdowns",
        "prior_season_mean_catch_rate",
        "prior_season_mean_yards_per_carry",
        "prior_season_total_yards",
        "prior_season_games_played",
        "prior_season_mean_fumbles_lost",
        # Red-zone usage carried into season S from S-1. Encodes goal-line
        # opportunity (carries inside the 20 + targets inside the 20) which
        # the volume-only aggregates above can't separate from open-field
        # production. Sourced from PBP via src.data.redzone_pbp and merged
        # onto every weekly row in src.data.loader. Motivation: RB attn-NN
        # weak on sparse TD heads vs Ridge (TODO.md hurdle_poisson archive
        # entry). Auto-flows into ATTN_STATIC_FEATURES via the prior_season
        # category derivation below.
        "prior_season_total_redzone_touches",
        "prior_season_mean_redzone_touches_per_game",
    ],
    # All EWMA dropped (>0.98 corr with rolling means)
    "ewma": [],
    "trend": ["trend_fantasy_points", "trend_targets", "trend_carries", "trend_snap_pct"],
    # Audit drops:
    #   * target_share_L5 (r=0.966 with L3) and carry_share_L5 (r=0.984 with L3)
    #     — match the documented >0.97 L5-drop rule applied to all rolling_*_L5;
    #     share-L5 was never re-audited under that rule until now.
    #   * carry_share_L3 (r=0.982 with team_rb_carry_share_L3) — the two
    #     denominators differ only by QB scrambles; the RB-specific
    #     denominator (in SPECIFIC_FEATURES) is the cleaner signal for an
    #     RB model.
    "share": [
        "target_share_L3",
        "snap_pct",
        "air_yards_share",
    ],
    # Audit drop:
    #   * opp_fantasy_pts_allowed_to_pos (VIF 193) — sum ≈ rush + recv
    #     components by construction; the components carry directional info
    #     (rush-friendly vs pass-funnel defenses) the sum collapses.
    #   * opp_def_rank_vs_pos — by construction
    #     rank(opp_fantasy_pts_allowed_to_pos) per week (engineer.py:480-482),
    #     Spearman = 1.0 with the value being ranked. The rank carries
    #     strictly less info than either the raw value or its components.
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
    # Audit drop:
    #   * is_returning_from_absence (r=0.934 with days_rest) — the indicator
    #     is essentially `days_rest > 13`; days_rest carries the magnitude.
    "contextual": [
        "is_home",
        "week",
        "days_rest",
        "practice_status",
        "game_status",
        "depth_chart_rank",
    ],
    # implied_team + implied_opp encodes both game total and spread direction
    # without the perfect collinearity of keeping total_line alongside either.
    # is_dome: dome premium on receiving (r=0.023 receiving_floor).
    "weather_vegas": ["implied_team_total", "implied_opp_total", "is_dome", "rest_advantage"],
    "specific": SPECIFIC_FEATURES,
}

# === Ridge ===
# Raw-stat grids — yards need broader high end (large variance vs counts).
_ALPHA_YARDS = alpha_grid(-2, 3, 15)
_ALPHA_COUNTS = alpha_grid(-1, 4, 15)
_ALPHA_RECEPTIONS = alpha_grid(-2, 2.5, 20)

RIDGE_ALPHA_GRIDS = {
    "rushing_tds": _ALPHA_COUNTS,
    "receiving_tds": _ALPHA_COUNTS,
    "rushing_yards": _ALPHA_YARDS,
    "receiving_yards": _ALPHA_YARDS,
    "receptions": _ALPHA_RECEPTIONS,
    "fumbles_lost": _ALPHA_COUNTS,
}

# Two-stage zero-inflated models: both rushing_tds and receiving_tds.
# Threshold + hyperparams preserved from the pre-migration td_points config;
# rebuilding per-TD afterwards gives two parallel classify-then-regress stacks.
TWO_STAGE_TARGETS = {
    "rushing_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
    "receiving_tds": {"clf_C": 0.001, "ridge_alpha": 0.01, "threshold": 0.5},
}

# Ordinal classification over raw TD counts {0,1,2,3+} per TD target.
ORDINAL_TARGETS = {
    "rushing_tds": {
        "type": "ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
    },
    "receiving_tds": {
        "type": "ordinal",
        "class_values": [0, 1, 2, 3],
        "alpha": 1.0,
    },
}

# Gated ordinal: binary gate + ordinal on positives, per TD target.
GATED_ORDINAL_TARGETS = {
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
}

# Which TD model variant to use: "ridge" | "two_stage" | "ordinal" | "gated_ordinal"
TD_MODEL_TYPE = "gated_ordinal"

# PCR: 80 components retains 99.8% variance, drops condition number from 1.8e8
# (after is_home removal) to 49.8.  Both yard targets improve by ~0.002 MAE.
RIDGE_PCA_COMPONENTS = 80

# === ElasticNet (optional parallel linear baseline, L1+L2) ===
# Off by default. Reuses RIDGE_ALPHA_GRIDS and searches over ENET_L1_RATIOS.
# Intentionally skips PCA — L1 on a rotated basis doesn't zero original features.
TRAIN_ELASTICNET = False
ENET_L1_RATIOS = list(DEFAULT_ENET_L1_RATIOS)

# === Neural Net ===
# [128, 64] two-layer backbone — single [128] was underfitting (early stop epoch 54,
# flat val loss from epoch 3). Added depth + larger heads + less regularization.
NN_BACKBONE_LAYERS = [128, 64]
NN_HEAD_HIDDEN = 48
NN_DROPOUT = 0.15
# All 6 RB heads are non-negative raw stats (yards, TDs, receptions, fumbles).
NN_NON_NEGATIVE_TARGETS = set(TARGETS)
NN_LR = 1e-3
NN_WEIGHT_DECAY = 5e-5
NN_EPOCHS = 300
NN_BATCH_SIZE = 256
NN_PATIENCE = 30
# Larger head for the hurdle-NegBin reception head (two value outputs: mu +
# log_alpha). TD heads moved to plain Poisson NLL (dispersion ~1.03-1.17, no
# zero-excess) and no longer need the extra capacity the Huber+gate setup did.
NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 64}

# === Per-Head Loss Families ===
# TDs + fumbles: plain Poisson NLL. Empirical dispersion 1.03-1.17 with
# negligible zero-excess — plain Poisson fits; the old BCE gate on (TD>0) was
# unmotivated and comes off here.
# Receptions: zero-truncated NegBin-2 hurdle. Variance/mean ~2.0 (overdispersed)
# with zero-excess up to +0.13 — textbook hurdle fit. Gate BCE is added via
# GATED_TARGETS below; the ZTNB NLL trains on positive samples only, scaled
# by fraction-positive inside the batch.
# Variant E (hurdle_poisson on TDs + fumbles_lost) was tested and rejected —
# wins per-target MAE on all three count heads but regresses FP MAE +0.163
# vs the current Variant C config. See TODO.md archive entry "RB sparse-count
# hurdle_poisson ablation" and docs/ARCHITECTURE.md D-entry for evidence.
HEAD_LOSSES = {
    "rushing_tds": "poisson_nll",
    "receiving_tds": "poisson_nll",
    "rushing_yards": "huber",
    "receiving_yards": "huber",
    "receptions": "hurdle_negbin",
    "fumbles_lost": "poisson_nll",
}

# === Loss Weights ===
# Yards heads: keep the 2.0/delta rebalance that stops yards gradients from
# dominating (without this, fantasy-point MAE regressed 4.23 -> 5.21; see the
# pre-PR-1 archive entry).
# Poisson NLL heads: picked so the expected weighted per-sample loss sits near
# the Huber contributions. At mean-TD-rate ~0.3, Poisson NLL ~ O(0.5); weight
# 1.0 keeps the contribution in the same 0.5-1.0 band as weighted yards Huber.
# hurdle_negbin reception head: weight 1.0. Value loss is already scaled by
# fraction-positive inside hurdle_negbin_value_loss, so no further rescaling.
LOSS_WEIGHTS = {
    "rushing_tds": 1.0,  # Poisson NLL
    "receiving_tds": 1.0,  # Poisson NLL
    "rushing_yards": 0.133,  # 2.0 / 15  (Huber)
    "receiving_yards": 0.133,
    "receptions": 1.0,  # hurdle_negbin, fraction-scaled internally
    "fumbles_lost": 1.0,  # Poisson NLL
}

# === Huber Deltas (per-target, raw-stat units) ===
# Only Huber heads need a delta. TD / fumble / reception heads use Poisson /
# hurdle-NegBin, which don't consume a delta.
HUBER_DELTAS = {
    "rushing_yards": 15.0,
    "receiving_yards": 15.0,
}

# === LR Scheduler ===
SCHEDULER_TYPE = "cosine_warm_restarts"
COSINE_T0 = 40
COSINE_T_MULT = 2
COSINE_ETA_MIN = 1e-5

# === Attention NN (game history variant) ===
TRAIN_ATTENTION_NN = True
# Keep d_model=32 (proven baseline) and n_heads=2 (larger values overfit on 15K samples).
ATTN_D_MODEL = 32
ATTN_N_HEADS = 2
# 2-layer nonlinear game encoder (Linear→ReLU→LayerNorm→Linear→ReLU) so each
# game is represented as a richer event embedding before attention, instead of
# a near-linear projection of raw stats.
ATTN_ENCODER_HIDDEN_DIM = 32
ATTN_MAX_SEQ_LEN = 17
# K/V projections disabled — at d_model=32 the 2K extra params hurt optimization
# more than they help (tested: 4.330 MAE with vs 4.228 without).
ATTN_PROJECT_KV = False
# Positional encoding: lightweight (17×32=544 params) temporal ordering signal so
# attention can distinguish recent games from older ones.
ATTN_POSITIONAL_ENCODING = True
ATTN_GATED_FUSION = False
# Very light attention dropout for regularization.
ATTN_DROPOUT = 0.05
# Standard training params match the base NN.
ATTN_LR = 1e-3
ATTN_WEIGHT_DECAY = 5e-5
ATTN_BATCH_SIZE = 256
ATTN_PATIENCE = 35
# Per-game stats fed into the attention sequence. fantasy_points is
# intentionally excluded — its scoring components (rushing/receiving
# yards + TDs + receptions + fumbles_lost) are all represented in the
# sequence, so it would be a redundant linear combination. Lagged
# fantasy_points signal still reaches Ridge / LightGBM / the base NN
# through the rolling / prior_season / trend categories of
# INCLUDE_FEATURES.
#
# Game-script context + team box score (last 12 columns) are per-historical-
# game raw context — not rolling/EWMA/trend — so they belong on the history
# tokens, not in ATTN_STATIC_FEATURES (which would violate D6). Without them
# the attention encoder treats every historical game as if its environment
# matched the prediction week, which is obviously wrong for RB usage (game
# script drives carry / target split). team_points_scored + opp_team_points_scored
# together encode in-game score margin, the dominant game-script signal.
ATTN_HISTORY_STATS = [
    "rushing_yards",
    "receiving_yards",
    "rushing_tds",
    "receiving_tds",
    "carries",
    "targets",
    "receptions",
    "fumbles_lost",
    "snap_pct",
    "rushing_first_downs",
    "receiving_first_downs",
    "game_carry_share",
    "game_target_share",
    "game_carry_hhi",
    "game_target_hhi",
    # Game-script context (already merged onto every player-week row by
    # merge_schedule_features / _build_contextual_features).
    "implied_team_total",
    "implied_opp_total",
    "is_home",
    "days_rest",
    # Team box score for that historical game (merged by
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
    # src.data.redzone_pbp.reconstruct_redzone_from_pbp and merged onto each
    # weekly row in src.data.loader. The 24 columns above can't separate a
    # goal-line back from a between-the-tackles workhorse — both can run for
    # 70 yards on 18 carries — but their TD output diverges by yardline_100
    # distribution. Targets the rushing_tds (+0.080 MAE vs Ridge) and
    # receiving_tds (+0.035) gaps the hurdle_poisson loss-family fix
    # couldn't close without regressing aggregate FP MAE (see TODO.md
    # "[TESTED, REJECTED] RB hurdle_poisson" archive entry).
    "redzone_carries",
    "redzone_targets",
    "inside10_carries",
    "inside5_carries",
    "redzone_target_share",
]
# Categories of INCLUDE_FEATURES that flow into the attention NN's static
# branch. The attention branch learns its own temporal representation from
# ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
# categories are intentionally excluded to avoid duplicating that signal.
# ``defense`` is also excluded: OPP_ATTN_HISTORY_STATS feeds the opposing
# defense's trailing form through a parallel attention branch, which makes
# the L5 static aggregates redundant for the NN. (They stay in
# INCLUDE_FEATURES["defense"] so Ridge / LightGBM still see them.) Mirrors
# the same exclusion already in place for QB and WR.
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
# RB-specific motivation: rushing yards are highly matchup-elastic against
# the opposing run defense, and the fixed L5 aggregate forces one recency
# curve for every player-week. The parallel branch lets attention learn its
# own recency emphasis (e.g. weighting the most recent 3 games when a
# defense just lost a starting DT) and decomposes the L5 sum back into its
# component games so per-target queries can focus on the matchup signal each
# stat actually cares about.
OPP_ATTN_HISTORY_STATS = list(DEFAULT_OPP_DEF_HISTORY_STATS)
OPP_ATTN_MAX_SEQ_LEN = DEFAULT_OPP_ATTN_MAX_SEQ_LEN
# Hurdle gate on receptions + BCE gate on each TD head. Variant C from the
# RB TD-gate ablation (src/tuning/ablate_rb_gate.py → run 24813558434):
#
#   Variant          FP MAE   Rush TD MAE   Rec TD MAE   Rec MAE
#   A (huber+gate)    4.453        0.277        0.077     1.034
#   B (Poisson/none)  4.258        0.329        0.064     0.989
#   C (Poisson+gate)  4.239        0.304        0.099     0.983
#
# C edges B by 0.019 pt/game — below the 0.05 pt decision threshold so the
# ablation script reported "drop TD gate", but the gate pulls the rushing_tds
# per-target MAE back from 0.329 to 0.304 (~the halfway point to variant A's
# 0.277 baseline) at no cost to FP MAE. Shipping with the gate because the
# per-target regression on rushing_tds under B was the one red flag in the
# PR #96 benchmark review, and variant C addresses it without giving up
# the hurdle-NegBin reception win. ``head_losses`` stays as PR #96 shipped
# — TDs on ``poisson_nll`` with BCE gate loss added via ``gated_targets``.
#
# Variants D (hurdle_poisson on TDs), E (D + fumbles_lost), and Bf (C + gate
# on fumbles_lost) were re-tested on 2026-05-20 (run fdcba72_rb_td_gate.json)
# under a new per-target-MAE decision rule. E wins per-target (CntSum 0.353
# vs C's 0.487 and Ridge's 0.369) but regresses FP MAE by +0.163; held back
# from shipping. See TODO.md archive for the full table and lesson.
ATTN_GATED = True
GATED_TARGETS = ["receptions", "rushing_tds", "receiving_tds"]
ATTN_GATE_HIDDEN = 16
ATTN_GATE_WEIGHT = 1.0

# === LightGBM (Optuna retune, 50 trials, CV MAE 4.5244) ===
# Flipped from ``"fair"`` to ``"huber"`` as part of the PR 3 LGBM unification
# (QB is the one exception — see LGBM_OBJECTIVE). Retuned on the huber
# objective, holdout comparison vs the old fair config:
#   Total MAE        4.479 -> 4.155  (-0.325)
#   Rushing Yards    21.3  -> 17.6   (-3.7)
#   Receiving Yards   9.15 ->  8.68  (-0.47)
#   Rushing Tds      0.299 -> 0.321  (+0.022)
#   Receiving Tds    0.093 -> 0.108  (+0.015)
#   Top-12 hit rate  0.476 -> 0.508  (+0.032)
#   Spearman rho     0.577 -> 0.733  (+0.156)
# Big yards improvements dominate a small TD regression. The CV MAE moved
# only +0.0095 vs the old fair (4.5149 → 4.5244) — within the plan's ±0.05
# tolerance — so the holdout -0.325 comes from the better hyperparams, not
# a lucky CV→holdout draw. Full tune_lgbm_results.json in retune run
# 24823926033 artifacts.
TRAIN_LIGHTGBM = True
LGBM_N_ESTIMATORS = 2000
LGBM_LEARNING_RATE = 0.0918744
LGBM_NUM_LEAVES = 16
LGBM_MAX_DEPTH = -1
LGBM_SUBSAMPLE = 0.547198
LGBM_COLSAMPLE_BYTREE = 0.783541
LGBM_REG_LAMBDA = 8.2532
LGBM_REG_ALPHA = 4.91934
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_MIN_SPLIT_GAIN = 0.315457
LGBM_OBJECTIVE = "huber"

# === Tiny config for end-to-end smoke tests ===
# Shrunk copy of the production config: 1 epoch, 2-layer x 8-unit backbone,
# attention and LightGBM disabled to keep the E2E smoke under 20s.
# Keeps every behavior toggle identical to CONFIG so test coverage
# exercises the same code paths.
NN_BACKBONE_LAYERS_TINY = [8, 8]
NN_HEAD_HIDDEN_TINY = 4
NN_EPOCHS_TINY = 1
NN_BATCH_SIZE_TINY = 64
NN_PATIENCE_TINY = 1

# Tiny configs for test fixtures: single-alpha grids and a flattened loss
# weight map keyed to the new targets.
CONFIG_TINY = {
    "targets": TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in TARGETS},
    "loss_weights": {t: 1.0 for t in TARGETS},
    "huber_deltas": HUBER_DELTAS,
    "nn_backbone_layers": NN_BACKBONE_LAYERS_TINY,
    "nn_head_hidden": NN_HEAD_HIDDEN_TINY,
    "nn_epochs": NN_EPOCHS_TINY,
    "nn_batch_size": NN_BATCH_SIZE_TINY,
    "nn_patience": NN_PATIENCE_TINY,
}


# === Bundled config object ===
# Single source of truth for downstream consumers (registry / pipeline / serving).
POSITION_CONFIG = PositionConfig(
    name="RB",
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
    attn_lr=ATTN_LR,
    attn_weight_decay=ATTN_WEIGHT_DECAY,
    attn_batch_size=ATTN_BATCH_SIZE,
    attn_patience=ATTN_PATIENCE,
    attn_history_stats=ATTN_HISTORY_STATS,
    attn_static_features=ATTN_STATIC_FEATURES,
    attn_project_kv=ATTN_PROJECT_KV,
    attn_gated_fusion=ATTN_GATED_FUSION,
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
    td_model_type=TD_MODEL_TYPE,
    two_stage_targets=TWO_STAGE_TARGETS,
    ordinal_targets=ORDINAL_TARGETS,
    gated_ordinal_targets=GATED_ORDINAL_TARGETS,
    accepts_dataframes=True,
    cpu_only=False,
    has_cv_runner=True,
)
