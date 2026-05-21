"""Factory that turns a :class:`PositionConfig` into the CONFIG dict /
``run()`` callable each ``src/{pos}/run_pipeline.py`` used to assemble by
hand.

Six near-identical ``CONFIG = {...}`` dicts (~50 keys, ~150 LOC each) get
replaced by a single ``build_pipeline_config(pos, POSITION_CONFIG)`` call,
plus ``make_run(pos, ...)`` / ``make_run_cv(pos, ...)`` for the wrapper
functions. The per-position callables (``filter_to_position``,
``compute_targets``, ``add_specific_features``, ``fill_nans``,
``get_feature_columns``) are resolved by convention via
``importlib.import_module(f"src.{pos.lower()}.{data|features|targets}")``.

K and DST keep their own ``run()`` orchestration (data loading happens
inside the runner, not in the shared pipeline), but they too consume this
factory to build the CONFIG base before injecting runtime closures like K's
``attn_history_builder_fn``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.position_config import PositionConfig

# ---------------------------------------------------------------------------
# Pipeline config schema — required vs optional keys
#
# ``src/shared/pipeline.py::run_pipeline`` consumes a ``cfg`` dict with ~79
# keys (mix of ``cfg["..."]`` required and ``cfg.get("...")`` optional). The
# constants below document the contract so :func:`validate_pipeline_config`
# can catch missing required keys at *construction* time (where the failure
# points back at the caller's ``PositionConfig`` or override list) rather
# than during training (where the trace points into ``pipeline.py``).
# ---------------------------------------------------------------------------

# Required cfg keys that ``run_pipeline`` accesses unconditionally via
# ``cfg["..."]`` — missing these crashes deep inside the pipeline.
REQUIRED_PIPELINE_CFG_KEYS: frozenset[str] = frozenset(
    {
        # Targets + callables
        "targets",
        "filter_fn",
        "compute_targets_fn",
        "get_feature_columns_fn",
        # Ridge
        "ridge_alpha_grids",
        # Per-head loss family
        "loss_weights",
        "huber_deltas",
        # Neural net
        "nn_backbone_layers",
        "nn_head_hidden",
        "nn_dropout",
        "nn_lr",
        "nn_weight_decay",
        "nn_epochs",
        "nn_batch_size",
        "nn_patience",
        # Scheduler — the type field is always read; the variant-specific
        # fields (cosine_t0 / onecycle_max_lr / etc.) are only required if
        # the chosen scheduler_type uses them, so they stay optional below.
        "scheduler_type",
    }
)


class PipelineConfigError(ValueError):
    """Raised by :func:`validate_pipeline_config` when required keys are missing."""


def validate_pipeline_config(cfg: dict[str, Any], *, context: str = "") -> None:
    """Assert every key in :data:`REQUIRED_PIPELINE_CFG_KEYS` is present.

    Raises :class:`PipelineConfigError` with the full list of missing keys
    in one shot, so callers see all of their mistakes at once instead of
    fixing them one-at-a-time as pipeline.py crashes on each.
    """
    missing = sorted(REQUIRED_PIPELINE_CFG_KEYS - cfg.keys())
    if missing:
        context_str = f" for {context}" if context else ""
        raise PipelineConfigError(
            f"Pipeline config{context_str} missing required keys: {missing}. "
            f"build_pipeline_config(pos, POSITION_CONFIG) is the canonical "
            f"constructor; if you've built the dict by hand, mirror the "
            f"REQUIRED_PIPELINE_CFG_KEYS set in src/shared/position_pipeline.py."
        )


# Standard import paths for the position-specific callables every position
# module exposes. By convention each position module declares these symbols at
# the same paths; the factory simply walks the convention rather than naming
# six separate import sites.
def build_position_callables(pos: str) -> dict[str, Callable]:
    """Resolve a position's data/features/targets callables by convention."""
    pos_lower = pos.lower()
    data_mod = importlib.import_module(f"src.{pos_lower}.data")
    features_mod = importlib.import_module(f"src.{pos_lower}.features")
    targets_mod = importlib.import_module(f"src.{pos_lower}.targets")
    return {
        "filter_fn": data_mod.filter_to_position,
        "compute_targets_fn": targets_mod.compute_targets,
        "add_features_fn": features_mod.add_specific_features,
        "fill_nans_fn": features_mod.fill_nans,
        "get_feature_columns_fn": features_mod.get_feature_columns,
    }


def _rb_classification_targets(pc: PositionConfig) -> dict[str, Any]:
    """RB picks one of three TD-model variants by ``TD_MODEL_TYPE``."""
    if pc.td_model_type == "ordinal":
        return pc.ordinal_targets
    if pc.td_model_type == "gated_ordinal":
        return pc.gated_ordinal_targets
    return {}


def _rb_two_stage_targets(pc: PositionConfig) -> dict[str, Any]:
    """RB's two_stage_targets are only active under TD_MODEL_TYPE=='two_stage'."""
    return pc.two_stage_targets if pc.td_model_type == "two_stage" else {}


def build_pipeline_config(
    pos: str,
    position_config: PositionConfig,
    **overrides: Any,
) -> dict[str, Any]:
    """Translate a :class:`PositionConfig` into the runtime CONFIG dict that
    ``src/shared/pipeline.py::run_pipeline`` consumes.

    ``overrides`` are merged last so callers (notably K) can inject runtime
    closures like ``attn_history_builder_fn`` after the base config is built.
    """
    pc = position_config

    cfg: dict[str, Any] = {
        # === Targets and feature whitelist ===
        "targets": pc.targets,
        # K is the lone exception: pass ``ALL_FEATURES`` (specific + contextual)
        # so ``fill_nans`` train-mean-fills the PBP-derived ``game_wind`` /
        # ``game_temp`` columns. For everyone else ``specific_features``
        # already isolates the position-specific block.
        "specific_features": (pc.all_features if pos == "K" else pc.specific_features),
        # === Ridge ===
        "ridge_alpha_grids": pc.ridge_alpha_grids,
        # === Per-position callables ===
        **build_position_callables(pos),
        # === Loss + scheduler ===
        "loss_weights": pc.loss_weights,
        "huber_deltas": pc.huber_deltas,
        "scheduler_type": pc.scheduler_type,
        # === Neural net ===
        "nn_backbone_layers": pc.nn_backbone_layers,
        "nn_head_hidden": pc.nn_head_hidden,
        "nn_dropout": pc.nn_dropout,
        "nn_head_hidden_overrides": pc.nn_head_hidden_overrides,
        "nn_non_negative_targets": pc.nn_non_negative_targets,
        "nn_lr": pc.nn_lr,
        "nn_weight_decay": pc.nn_weight_decay,
        "nn_epochs": pc.nn_epochs,
        "nn_batch_size": pc.nn_batch_size,
        "nn_patience": pc.nn_patience,
        "nn_use_amp": pc.nn_use_amp,
        # === Attention ===
        "train_attention_nn": pc.train_attention_nn,
        "attn_d_model": pc.attn_d_model,
        "attn_n_heads": pc.attn_n_heads,
        "attn_encoder_hidden_dim": pc.attn_encoder_hidden_dim,
        "attn_positional_encoding": pc.attn_positional_encoding,
        "attn_dropout": pc.attn_dropout,
        "attn_static_features": pc.attn_static_features,
        # === ElasticNet ===
        "train_elasticnet": pc.train_elasticnet,
        "enet_l1_ratios": pc.enet_l1_ratios,
        # === LightGBM ===
        "train_lightgbm": pc.train_lightgbm,
        "lgbm_n_estimators": pc.lgbm_n_estimators,
        "lgbm_learning_rate": pc.lgbm_learning_rate,
        "lgbm_num_leaves": pc.lgbm_num_leaves,
        "lgbm_subsample": pc.lgbm_subsample,
        "lgbm_colsample_bytree": pc.lgbm_colsample_bytree,
        "lgbm_reg_lambda": pc.lgbm_reg_lambda,
        "lgbm_reg_alpha": pc.lgbm_reg_alpha,
        "lgbm_min_child_samples": pc.lgbm_min_child_samples,
        "lgbm_min_split_gain": pc.lgbm_min_split_gain,
        "lgbm_objective": pc.lgbm_objective,
    }

    # === Optional scheduler-family keys ===
    # cosine fields default to None on positions using onecycle and vice versa;
    # only emit the ones the position actually set so pipeline.py's getter
    # doesn't see a None where the original CONFIG omitted the key.
    if pc.cosine_t0 is not None:
        cfg["cosine_t0"] = pc.cosine_t0
        cfg["cosine_t_mult"] = pc.cosine_t_mult
        cfg["cosine_eta_min"] = pc.cosine_eta_min
    if pc.onecycle_max_lr is not None:
        cfg["onecycle_max_lr"] = pc.onecycle_max_lr
        cfg["onecycle_pct_start"] = pc.onecycle_pct_start

    # === Optional attention extras ===
    # ``attn_max_seq_len`` is omitted for K (it uses ``attn_max_games`` for its
    # nested attention instead) and for positions that didn't define it; the
    # opp-attention max-seq-len next door follows the same rule.
    if pc.attn_max_seq_len is not None:
        cfg["attn_max_seq_len"] = pc.attn_max_seq_len
    # The four attention training-loop knobs (lr, weight_decay, batch_size,
    # patience) get plumbed through whenever the position trains attention —
    # WR/TE leave them at defaults and aren't included historically.
    if pos in ("QB", "RB", "K", "DST"):
        cfg["attn_lr"] = pc.attn_lr
        cfg["attn_weight_decay"] = pc.attn_weight_decay
        cfg["attn_batch_size"] = pc.attn_batch_size
        cfg["attn_patience"] = pc.attn_patience
    # attn_project_kv is plumbed by RB/K/DST (matches original CONFIG dicts);
    # the skill positions skip it because their flat attention path defaults False.
    if pos in ("RB", "K", "DST"):
        cfg["attn_project_kv"] = pc.attn_project_kv
    # attn_gated_fusion plumbed by RB/DST; K's nested path doesn't consume it.
    if pos in ("RB", "DST"):
        cfg["attn_gated_fusion"] = pc.attn_gated_fusion
    # attn_history_stats / head_losses / attn_gated / gate_* — flat attention
    # consumers only on the non-K side. K opts in to ``attn_history_stats`` via
    # the nested+history branch (per-game aggregates feed alongside the inner
    # kick pool); head_losses / attn_gated are still flat-only.
    if pc.attn_history_stats:
        cfg["attn_history_stats"] = pc.attn_history_stats
    if pos != "K":
        cfg["head_losses"] = pc.head_losses
        cfg["attn_gated"] = pc.attn_gated
    # attn_gate_hidden / attn_gate_weight only ride along on the skill positions
    # whose ATTN_GATED can be True (RB/WR/TE) plus QB which sets them to False
    # but historically plumbed them. K and DST omit the pair.
    if pos in ("QB", "RB", "WR", "TE"):
        cfg["attn_gate_hidden"] = pc.attn_gate_hidden
        cfg["attn_gate_weight"] = pc.attn_gate_weight
    if pc.gated_targets:
        cfg["gated_targets"] = pc.gated_targets

    # === Opp-attention branch (skipped for K; DST uses offense) ===
    if pc.opp_attn_history_stats:
        cfg["opp_attn_history_stats"] = pc.opp_attn_history_stats
        cfg["opp_attn_max_seq_len"] = pc.opp_attn_max_seq_len
        if pc.opp_attn_kind != "defense":
            cfg["opp_attn_kind"] = pc.opp_attn_kind

    # === Ridge extras ===
    if pc.ridge_pca_components is not None:
        cfg["ridge_pca_components"] = pc.ridge_pca_components
    if pc.ridge_cv_folds is not None:
        cfg["ridge_cv_folds"] = pc.ridge_cv_folds
    if pc.ridge_refine_points is not None:
        cfg["ridge_refine_points"] = pc.ridge_refine_points
    if pc.cv_split_column is not None:
        cfg["cv_split_column"] = pc.cv_split_column

    # === LGBM max_depth (omitted for QB) ===
    if pos != "QB":
        cfg["lgbm_max_depth"] = pc.lgbm_max_depth

    # === Aggregator (used at training/eval time to compute fantasy-point
    # totals from raw-stat predictions; K's aggregator subtracts miss
    # penalties via ``target_signs`` so we route through it just like
    # other positions) ===
    cfg["aggregate_fn"] = aggregate_fn_for(pos)

    # === K/DST shared: explicit None for the adjustment slot ===
    if pos in ("K", "DST"):
        cfg["compute_adjustment_fn"] = None

    # === K-only: nested attention structure flags ===
    if pos == "K":
        cfg["attn_history_structure"] = "nested"
        cfg["attn_static_from_df"] = True
        cfg["attn_kick_dim"] = pc.attn_kick_dim

    # === DST-only: very-rare-count Poisson targets ===
    if pos == "DST":
        cfg["poisson_targets"] = pc.poisson_targets

    # === RB-only: TD model variants ===
    if pos == "RB":
        cfg["two_stage_targets"] = _rb_two_stage_targets(pc)
        cfg["classification_targets"] = _rb_classification_targets(pc)

    # === Caller overrides (e.g. K injects attn_history_builder_fn) ===
    cfg.update(overrides)

    # === Schema validation ===
    # Catch missing required keys at construction time. Optional keys (cosine_*,
    # onecycle_*, ridge_pca_components, attn_*, etc.) intentionally bypass this
    # — pipeline.py reads them via .get() with sensible defaults.
    validate_pipeline_config(cfg, context=pos)
    return cfg


# Note: per-position ``run`` / ``run_cv`` functions are defined inside each
# ``src/{pos}/run_pipeline.py`` rather than generated here so that
# ``src.shared.pipeline.run_pipeline`` is bound at the position-module level
# (existing tests monkeypatch it via ``setattr(qb_pipe, "run_pipeline", ...)``).
# The factory's job is purely to assemble the CONFIG dict.
