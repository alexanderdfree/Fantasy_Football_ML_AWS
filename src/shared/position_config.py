"""Shared per-position configuration schema.

Each ``src/{pos}/config.py`` exports exactly one ``POSITION_CONFIG:
PositionConfig`` instance with every hyperparameter inlined into the
constructor call. Downstream consumers (training pipeline, serving, tests)
read attributes off ``POSITION_CONFIG`` — there are no module-level
UPPERCASE constants mirroring the dataclass fields.

The two narrow exceptions kept at module level are ``CONFIG_TINY`` /
``CONFIG_TINY_ATTN`` (e2e tests import them by name) and
``ATTN_STATIC_CATEGORIES`` (the attention-static whitelist test reads it
on QB/RB/WR/TE to confirm the category list excludes rolling buckets).

Field defaults capture the most-common value across the six positions, so
each position config only has to spell out where it deviates. Adding a new
position becomes a single ``PositionConfig(...)`` constructor call.

Use :func:`alpha_grid` and the ``DEFAULT_OPP_DEF_HISTORY_STATS`` /
``derive_attn_static_features`` helpers below to remove the obvious
structural duplication across configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.position import Position

# ---------------------------------------------------------------------------
# Shared constants and helpers used by per-position config files.
# ---------------------------------------------------------------------------


def alpha_grid(low: float, high: float, n: int = 15) -> list[float]:
    """Log-spaced Ridge alpha grid, rounded to 4 decimal places."""
    return [round(x, 4) for x in np.logspace(low, high, n)]


# Per-game opposing-defense stats fed to the parallel attention branch for
# QB / RB / WR / TE. DST overrides with offense-side stats; K omits the
# branch entirely. Centralised here so adding a new defensive aggregate is a
# single-line edit.
DEFAULT_OPP_DEF_HISTORY_STATS: list[str] = [
    "def_sacks",
    "def_pass_yds_allowed",
    "def_pass_td_allowed",
    "def_ints",
    "def_rush_yds_allowed",
    "def_pts_allowed",
]

DEFAULT_OPP_ATTN_MAX_SEQ_LEN: int = 17
DEFAULT_ENET_L1_RATIOS: tuple[float, ...] = (0.3, 0.5, 0.7)

# Default INCLUDE_FEATURES category whitelist for the attention NN's static
# branch on QB / RB / WR / TE. Rolling / ewma / trend / share / specific
# categories are intentionally excluded — the attention branch learns its own
# temporal representation from ATTN_HISTORY_STATS, so double-feeding leaks
# signal. ``defense`` is excluded because the parallel opp-defense attention
# branch already feeds those aggregates per-game. K and DST use different /
# explicit allowlists (no INCLUDE_FEATURES category structure), so they don't
# read this default. Re-exported at module level on each skill-position config
# so tests/test_attn_static_columns.py can read ``cfg.ATTN_STATIC_CATEGORIES``.
DEFAULT_ATTN_STATIC_CATEGORIES: list[str] = [
    "prior_season",
    "matchup",
    "contextual",
    "weather_vegas",
]


def derive_attn_static_features(
    include_features: dict[str, list[str]],
    categories: list[str],
) -> list[str]:
    """Flatten the named INCLUDE_FEATURES categories into the attention NN
    static branch's feature list.

    The attention branch learns its own temporal representation from
    ATTN_HISTORY_STATS, so rolling / ewma / trend / share / specific
    categories are intentionally excluded — this helper takes the categories
    that *do* belong (prior_season / matchup / contextual / weather_vegas
    in the canonical four skill positions) and concatenates their entries.
    """
    return [c for cat in categories for c in include_features[cat]]


# ---------------------------------------------------------------------------
# PositionConfig dataclass — bundles per-position hyperparameters.
# ---------------------------------------------------------------------------


@dataclass
class PositionConfig:
    """Per-position training/inference hyperparameters.

    Required fields (no default): identity, targets, ridge grid, neural-net
    architecture, scheduler choice, per-head loss specification. Everything
    else has a default that matches the most-common value across the six
    positions, so each ``PositionConfig(...)`` call only spells out where
    that position deviates from the consensus.

    Snake-case attribute names mirror the dataclass field names directly
    (``targets``, ``nn_backbone_layers``, ``attn_d_model``, ...); the old
    UPPER_CASE module-level constants have been retired.
    """

    # === Identity ===
    name: str

    # === Targets and feature whitelist ===
    targets: list[str]
    specific_features: list[str]

    # Skill positions (QB/RB/WR/TE) use the include_features category dict;
    # K and DST use the flat all_features list. Both shapes are tracked here
    # so downstream consumers can pick whichever fits.
    include_features: dict[str, list[str]] = field(default_factory=dict)
    contextual_features: list[str] = field(default_factory=list)
    all_features: list[str] = field(default_factory=list)
    drop_features: set[str] = field(default_factory=set)

    # === Ridge / ElasticNet ===
    ridge_alpha_grids: dict[str, list[float]] = field(default_factory=dict)
    ridge_pca_components: int | None = None
    ridge_cv_folds: int | None = None
    ridge_refine_points: int | None = None
    cv_split_column: str | None = None
    train_elasticnet: bool = False
    enet_l1_ratios: list[float] = field(default_factory=lambda: list(DEFAULT_ENET_L1_RATIOS))

    # === Neural Net (core architecture) ===
    nn_backbone_layers: list[int] = field(default_factory=list)
    nn_head_hidden: int = 32
    nn_dropout: float = 0.25
    nn_non_negative_targets: set[str] = field(default_factory=set)
    nn_lr: float = 1e-3
    nn_weight_decay: float = 5e-5
    nn_epochs: int = 250
    nn_batch_size: int = 128
    nn_patience: int = 30
    nn_head_hidden_overrides: dict[str, int] = field(default_factory=dict)
    # FP16 autocast + GradScaler on the NN forward + loss path (both base
    # and attention branches). True by default so every position picks it
    # up on T4 (sm_75 has native FP16 Tensor Cores); flip to False per-
    # position if a benchmark diff shows a per-target MAE regression beyond
    # the project's ±2% tolerance. No-op on non-CUDA devices (GradScaler
    # short-circuits via `enabled=False`). Was BF16 in PR #293 — replaced
    # with FP16 because T4 has no native BF16 Tensor Cores (Ampere+); BF16
    # autocast hung production for ~1 hour before this fix.
    nn_use_amp: bool = True

    # === Per-head loss families ===
    head_losses: dict[str, str] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)
    huber_deltas: dict[str, float] = field(default_factory=dict)
    poisson_targets: list[str] = field(default_factory=list)
    gated_targets: list[str] = field(default_factory=list)

    # === LR scheduler (cosine_warm_restarts | onecycle) ===
    scheduler_type: str = "cosine_warm_restarts"
    cosine_t0: int | None = None
    cosine_t_mult: int | None = None
    cosine_eta_min: float | None = None
    onecycle_max_lr: float | None = None
    onecycle_pct_start: float | None = None

    # === Attention NN — flat history (QB/RB/WR/TE/DST) ===
    train_attention_nn: bool = True
    attn_d_model: int = 32
    attn_n_heads: int = 2
    attn_encoder_hidden_dim: int = 0
    attn_max_seq_len: int | None = DEFAULT_OPP_ATTN_MAX_SEQ_LEN
    attn_positional_encoding: bool = True
    attn_dropout: float = 0.0
    attn_lr: float = 1e-3
    attn_weight_decay: float = 5e-5
    attn_batch_size: int = 256
    attn_patience: int = 35
    attn_history_stats: list[str] = field(default_factory=list)
    attn_static_features: list[str] = field(default_factory=list)
    attn_project_kv: bool = False
    attn_gated_fusion: bool = False
    attn_gated: bool = False
    attn_gate_hidden: int = 16
    attn_gate_weight: float = 1.0

    # === Opposing-side attention branch ===
    opp_attn_history_stats: list[str] = field(default_factory=list)
    opp_attn_max_seq_len: int | None = DEFAULT_OPP_ATTN_MAX_SEQ_LEN
    opp_attn_kind: str = "defense"  # DST overrides to "offense"

    # === Nested kick attention (K only) ===
    attn_max_games: int | None = None
    attn_kick_dim: int | None = None
    attn_max_kicks_per_game: int | None = None
    attn_kick_stats: list[str] = field(default_factory=list)

    # === LightGBM ===
    train_lightgbm: bool = True
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_max_depth: int = -1
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 2.0
    lgbm_reg_alpha: float = 0.1
    lgbm_min_child_samples: int = 30
    lgbm_min_split_gain: float = 0.0
    lgbm_objective: str = "huber"

    # === RB-only TD model variants ===
    td_model_type: str | None = None
    two_stage_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    ordinal_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    gated_ordinal_targets: dict[str, dict[str, Any]] = field(default_factory=dict)

    # === K-only ===
    seasons: list[int] | None = None
    min_games: int | None = None
    # K's serving aggregator (app.py) uses ``target_signs`` to combine the four
    # raw-count heads into fantasy points: positive for the scoring heads,
    # negative for the miss penalties. Other positions use the shared
    # ``predictions_to_fantasy_points`` aggregator instead and leave this None.
    target_signs: dict[str, float] | None = None

    # === Pipeline orchestration metadata ===
    # Read by src/shared/registry.py to decide which runner / inference
    # contract to use; PR #244 migrated the registry to read from here.
    accepts_dataframes: bool = True
    cpu_only: bool = False
    has_cv_runner: bool = False

    def __post_init__(self) -> None:
        # Validate ``name`` against the canonical Position enum so typos in a
        # per-position ``POSITION_CONFIG = PositionConfig(name=...)`` call
        # fail at import time with a clear error rather than landing in a
        # downstream dict lookup as a silent miss.
        Position(self.name)
