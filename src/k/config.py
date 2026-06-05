"""K (kicker) hyperparameters bundled in a single :class:`PositionConfig`.

Downstream consumers read from the exported ``POSITION_CONFIG`` exclusively.
``CONFIG_TINY`` and ``CONFIG_TINY_ATTN`` remain at module level — e2e and
attention-pipeline tests import them by name.
"""

from src.shared.position_config import (
    DEFAULT_ENET_L1_RATIOS,
    PositionConfig,
    alpha_grid,
)

# === K Seasons (post-PAT rule change: 2015+) ===
_SEASONS = list(range(2015, 2026))  # 2015-2025

# 4 non-negative raw-value heads. Total fantasy points = sum with signs
# [+1, +1, -1, -1] applied at inference (target_signs below).
_TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]

_SPECIFIC_FEATURES = [
    # Rolling performance
    "fg_attempts_L3",
    "fg_accuracy_L5",
    "pat_volume_L3",
    "total_k_pts_L3",
    "long_fg_rate_L3",
    "k_pts_trend",
    "k_pts_std_L3",
    # PBP: distance & difficulty
    "avg_fg_distance_L3",
    "avg_fg_prob_L3",
    # PBP: situational accuracy
    "fg_pct_40plus_L5",
    "q4_fg_rate_L5",
    "xp_accuracy_L5",
]

_CONTEXTUAL_FEATURES = [
    "is_home",
    "week",
    "implied_team_total",
    "total_line",
    # PBP Tier 1: game-level weather/venue
    "is_dome",
    "game_wind",
    "game_temp",
]

_ALL_FEATURES = _SPECIFIC_FEATURES + _CONTEXTUAL_FEATURES

_ATTN_HISTORY_STATS = [
    # FG / PAT volume + accuracy.
    "fg_att",
    "fg_made",
    "pat_att",
    "pat_made",
    "fg_yards_made",
    # Distance / difficulty.
    "avg_fg_distance",
    "long_fg_att",
    "long_fg_made",
    # Situational accuracy.
    "q4_fg_att",
    "q4_fg_made",
    # Per-game fantasy total (signed scoring sum from the four target
    # heads; attention learns its own recency weighting on it).
    "fantasy_points",
    # Game-level context — these ALSO appear in the static whitelist (via
    # ``_CONTEXTUAL_FEATURES`` → ``attn_static_features`` below), feeding
    # the attention NN through both branches. The static branch provides
    # the current-game value; the per-game history branch lets attention
    # condition recency weighting on the recent-game weather/venue trend.
    "is_home",
    "is_dome",
    "implied_team_total",
    "game_wind",
    # Realized team score of each prior game, merged onto every row by
    # build_position_features (the team_box_score merge runs for K — its
    # _schedule_merged sentinel only blocks the schedule merge, not this one).
    # K's kicking opportunity scales with its own team's scoring, so the
    # per-game team score lets attention condition on game environment.
    # Only team_points_scored is added: opp_team_points_scored is omitted
    # (K's frame carries no schedule-matching opponent_team, so the opp-side
    # merge leaves it constant-zero / dead), and implied_opp_total / days_rest
    # are absent because K short-circuits the schedule merge (src/k/data.py).
    "team_points_scored",
]

_ATTN_KICK_STATS = [
    "is_fg",
    "is_xp",
    "kick_distance",
    "kick_made",
    "fg_prob",
    "is_q4",
    "score_diff",
    "game_wind",
    "is_home",
]


# === Tiny config for E2E smoke tests ===
# Shrunk to 1 epoch with a 2-layer × 8-unit NN so the full pipeline runs
# in well under 20s on CPU. Used by tests/k/test_pipeline_e2e.py.
CONFIG_TINY = {
    "targets": _TARGETS,
    "ridge_alpha_grids": {t: [1.0] for t in _TARGETS},
    "ridge_cv_folds": 2,
    "cv_split_column": "season",
    "ridge_refine_points": 0,
    # ALL_FEATURES (specific + contextual), mirroring the production K config
    # — the contextual block carries the PBP-derived ``game_wind`` /
    # ``game_temp`` columns that the catch-all ``.fillna(0)`` would
    # otherwise collapse to 0 (off the train scale).
    "specific_features": _ALL_FEATURES,
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 32,
    "nn_patience": 1,
    "nn_log_every": 1,
    "loss_weights": {t: 1.0 for t in _TARGETS},
    "huber_deltas": {t: 2.0 for t in _TARGETS},
    "scheduler_type": "onecycle",
    "onecycle_max_lr": 1e-3,
    "onecycle_pct_start": 0.3,
    "train_lightgbm": False,
}

# Attention tiny config — reuses CONFIG_TINY base plus a shrunk attention
# branch. Used by the E2E attention test. ``attn_history_builder_fn`` must
# be supplied by the caller (it captures a kicks_df in its closure).
CONFIG_TINY_ATTN = {
    **CONFIG_TINY,
    "train_attention_nn": True,
    "attn_history_structure": "nested",
    "attn_static_from_df": True,
    "attn_static_features": list(_CONTEXTUAL_FEATURES),
    "attn_history_stats": _ATTN_HISTORY_STATS,
    "attn_d_model": 8,
    "attn_n_heads": 1,
    "attn_kick_dim": 4,
    "attn_encoder_hidden_dim": 0,
    "attn_project_kv": False,
    "attn_positional_encoding": True,
    "attn_dropout": 0.0,
    "attn_lr": 1e-3,
    "attn_weight_decay": 0.0,
    "attn_batch_size": 32,
    "attn_patience": 1,
    # Explicit nested-attention sequence caps — historically relied on the
    # builder closure's fallback defaults, fragile if
    # ``MultiHeadNetWithNestedHistory`` ever required them from cfg.
    "attn_max_games": 17,
    "attn_max_kicks_per_game": 10,
}


# Single source of truth for downstream consumers (registry / pipeline / serving).
# K's attention is nested (per-kick inner pool + per-game outer attention), so
# attn_max_seq_len stays None — the outer sequence length lives in attn_max_games.
# The consuming gate at ``src/shared/position_pipeline.py`` (search
# "attn_max_seq_len") skips plumbing this key when it's None; this is the
# only position that intentionally relies on that branch.
POSITION_CONFIG = PositionConfig(
    name="K",
    targets=_TARGETS,
    specific_features=_SPECIFIC_FEATURES,
    contextual_features=_CONTEXTUAL_FEATURES,
    all_features=_ALL_FEATURES,
    # No general features — kickers bypass the player-level feature pipeline.
    drop_features=set(),
    ridge_alpha_grids={t: alpha_grid(-1, 4, 15) for t in _TARGETS},
    ridge_cv_folds=3,
    ridge_refine_points=0,
    cv_split_column="season",
    train_elasticnet=False,
    enet_l1_ratios=list(DEFAULT_ENET_L1_RATIOS),
    # 2015-2025 dataset: more data allows a larger model.
    nn_backbone_layers=[64, 32],
    nn_head_hidden=16,
    nn_dropout=0.25,
    # All 4 K heads are non-negative raw counts/points; signs are applied
    # only in the final fantasy total aggregation, not in per-head outputs.
    nn_non_negative_targets=set(_TARGETS),
    nn_lr=3e-4,
    nn_weight_decay=2e-4,
    nn_epochs=250,
    nn_batch_size=128,
    nn_patience=30,
    # All heads on "huber" — no behavior change vs the default.
    head_losses={t: "huber" for t in _TARGETS},
    # Equal per-target weights — Huber delta harmonized to 2.0 across targets.
    loss_weights={t: 1.0 for t in _TARGETS},
    huber_deltas={t: 2.0 for t in _TARGETS},
    scheduler_type="onecycle",
    onecycle_max_lr=1e-3,
    onecycle_pct_start=0.3,
    # === Attention NN (nested: per-kick inner pool, per-game outer) ===
    # Outer attention mirrors RB's proven d_model=32 / n_heads=2.
    train_attention_nn=True,
    attn_d_model=32,
    attn_n_heads=2,
    attn_encoder_hidden_dim=32,
    attn_max_seq_len=None,
    attn_positional_encoding=True,
    attn_dropout=0.05,
    attn_lr=2e-3,
    attn_weight_decay=5e-5,
    attn_batch_size=512,
    attn_patience=20,
    attn_onecycle_max_lr=2e-3,
    attn_static_features=list(_CONTEXTUAL_FEATURES),
    attn_history_stats=_ATTN_HISTORY_STATS,
    attn_project_kv=False,
    attn_max_games=17,
    attn_kick_dim=16,
    attn_max_kicks_per_game=10,
    attn_kick_stats=_ATTN_KICK_STATS,
    opp_attn_max_seq_len=None,
    train_lightgbm=True,
    lgbm_n_estimators=300,
    lgbm_learning_rate=0.05,
    lgbm_num_leaves=15,
    lgbm_max_depth=-1,
    lgbm_subsample=0.8,
    lgbm_colsample_bytree=0.8,
    lgbm_reg_lambda=2.0,
    lgbm_reg_alpha=0.1,
    lgbm_min_child_samples=30,
    lgbm_min_split_gain=0.0,
    lgbm_objective="huber",
    seasons=_SEASONS,
    # Cross-season split, matching other positions.
    min_games=4,
    # K's serving aggregator (app.py) uses ``target_signs`` to combine the
    # four raw-count heads into fantasy points: positive for scoring heads,
    # negative for miss penalties. Other positions go through the shared
    # ``predictions_to_fantasy_points`` aggregator instead.
    target_signs={
        "fg_yard_points": 1.0,
        "pat_points": 1.0,
        "fg_misses": -1.0,
        "xp_misses": -1.0,
    },
    accepts_dataframes=False,
    cpu_only=True,
    has_cv_runner=True,
)
