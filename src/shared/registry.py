"""Single source of truth for per-position dispatch.

Consumers: app.py (inference), src/batch/train.py (training), benchmark.py
(local benchmark), tune_lgbm.py (LightGBM tuning), tests/_pipeline_e2e_utils.py.

Everything is lazily imported so loading this module is cheap.

Position metadata (``runner_module``, ``accepts_dataframes``, ``cpu_only``,
``has_cv_runner``) lives on each position's ``POSITION_CONFIG`` instance.
The legacy ``_POSITION_META`` dict is gone; we read the bundled config via a
small lazy importer.
"""

import importlib
from functools import cache

from src.shared.position import Position
from src.shared.position_config import PositionConfig

ALL_POSITIONS = Position.values()


@cache
def _import_config_module(pos: str):
    """Lazy-import a position's config module (cheap; cached)."""
    if pos not in ALL_POSITIONS:
        raise ValueError(f"Unknown position: {pos}")
    return importlib.import_module(f"src.{pos.lower()}.config")


def _position_config(pos: str) -> PositionConfig:
    return _import_config_module(pos).POSITION_CONFIG


@cache
def _import_runner_module(pos: str):
    if pos not in ALL_POSITIONS:
        raise ValueError(f"Unknown position: {pos}")
    return importlib.import_module(f"src.{pos.lower()}.run_pipeline")


def get_runner(pos: str):
    return _import_runner_module(pos).run


def get_config(pos: str) -> dict:
    """Return the position runner's CONFIG dict (the one passed into
    ``src.shared.pipeline.run_pipeline``). Use ``POSITION_CONFIG`` directly if
    you want the dataclass form instead.
    """
    return _import_runner_module(pos).CONFIG


def accepts_dataframes(pos: str) -> bool:
    return _position_config(pos).accepts_dataframes


def is_cpu_only(pos: str) -> bool:
    return _position_config(pos).cpu_only


CPU_ONLY_POSITIONS = {p for p in ALL_POSITIONS if _position_config(p).cpu_only}


# ---------------------------------------------------------------------------
# Inference spec — used by app.py to apply position-specific models.
# Lazy-loaded once per position; per-position modules are only imported when
# the position is first requested.
# ---------------------------------------------------------------------------


def _flat_attn_kwargs_static(pc: PositionConfig) -> dict:
    """Pull all ``MultiHeadNetWithHistory`` kwargs from a ``PositionConfig``
    EXCEPT the runtime-dependent ``static_dim`` / ``game_dim`` / ``target_names`` —
    those are filled in by the caller at inference time once the feature list
    is known.
    """
    kwargs = dict(
        backbone_layers=list(pc.nn_backbone_layers),
        d_model=pc.attn_d_model,
        n_attn_heads=pc.attn_n_heads,
        head_hidden=pc.nn_head_hidden,
        dropout=pc.nn_dropout,
        project_kv=pc.attn_project_kv,
        use_positional_encoding=pc.attn_positional_encoding,
        max_seq_len=pc.attn_max_seq_len if pc.attn_max_seq_len is not None else 17,
        use_gated_fusion=pc.attn_gated_fusion,
        attn_dropout=pc.attn_dropout,
        encoder_hidden_dim=pc.attn_encoder_hidden_dim,
        gated=pc.attn_gated,
        gate_hidden=pc.attn_gate_hidden,
        gated_targets=list(pc.gated_targets) if pc.gated_targets else None,
        # Architecture knob (adds an nn.Parameter when on) — must be in the
        # served kwargs so app.py / smoke_test rebuild the matching state_dict.
        no_history_embedding=pc.attn_no_history_embedding,
    )
    if pc.nn_head_hidden_overrides:
        kwargs["head_hidden_overrides"] = dict(pc.nn_head_hidden_overrides)
    # ``nn_non_negative_targets`` is ``field(default_factory=set)`` on
    # :class:`PositionConfig` — never ``None``. All six positions set it
    # explicitly to ``set(_TARGETS)`` (clamp every head). Plumb unconditionally;
    # an empty set would mean "no head is clamped", which is the intended
    # escape hatch for a future signed-output head.
    kwargs["non_negative_targets"] = set(pc.nn_non_negative_targets)
    return kwargs


def _nested_attn_kwargs_static(pc: PositionConfig) -> dict:
    """``MultiHeadNetWithNestedHistory`` (K) takes a different shape — inner
    per-kick attention is parametrised by ``d_kick`` and the outer per-target
    attention by ``d_model`` / ``max_games`` instead of ``max_seq_len``.

    ``game_dim`` flows from the length of ``attn_history_stats``: 0 = legacy
    nested-only path, >0 = per-game aggregates fed alongside the inner pool.
    """
    return dict(
        backbone_layers=list(pc.nn_backbone_layers),
        d_kick=pc.attn_kick_dim,
        d_model=pc.attn_d_model,
        n_attn_heads=pc.attn_n_heads,
        head_hidden=pc.nn_head_hidden,
        dropout=pc.nn_dropout,
        non_negative_targets=set(pc.nn_non_negative_targets),
        project_kv=pc.attn_project_kv,
        use_positional_encoding=pc.attn_positional_encoding,
        max_games=pc.attn_max_games,
        attn_dropout=pc.attn_dropout,
        encoder_hidden_dim=pc.attn_encoder_hidden_dim,
        game_dim=len(pc.attn_history_stats),
    )


def _position_modules(pos: str):
    """Lazy-import the per-position data/features/targets modules."""
    pos_lower = pos.lower()
    return (
        importlib.import_module(f"src.{pos_lower}.data"),
        importlib.import_module(f"src.{pos_lower}.features"),
        importlib.import_module(f"src.{pos_lower}.targets"),
    )


def _flat_nn_kwargs(pc: PositionConfig) -> dict:
    """Per-position ``MultiHeadNet`` kwargs (no attention).

    QB sets ``NN_HEAD_HIDDEN_OVERRIDES = {}`` and historically passed neither
    the kwarg nor an empty dict — the K branch did the same. Anywhere else we
    pass the dict iff it actually has entries.
    """
    kwargs = dict(
        backbone_layers=pc.nn_backbone_layers,
        head_hidden=pc.nn_head_hidden,
        dropout=pc.nn_dropout,
        non_negative_targets=pc.nn_non_negative_targets,
    )
    if pc.nn_head_hidden_overrides:
        kwargs["head_hidden_overrides"] = pc.nn_head_hidden_overrides
    return kwargs


@cache
def get_inference_spec(pos: str) -> dict:
    """Build the inference-time spec dict consumed by app.py.

    All field values flow from ``POSITION_CONFIG``; K's nested attention path
    and its ``target_signs`` aggregator are the only branches.
    """
    pc = _position_config(pos)
    pos_lower = pos.lower()
    data_mod, features_mod, targets_mod = _position_modules(pos)

    spec = {
        # === Targets and per-position callables ===
        "targets": pc.targets,
        # K is the lone exception: pass ``all_features`` (specific + contextual)
        # so serving's ``fill_nans`` train-mean-fills the PBP-derived
        # ``game_wind`` / ``game_temp`` columns exactly as training does. For
        # everyone else ``specific_features`` already isolates the position
        # block. Mirrors ``build_pipeline_config`` in position_pipeline.py.
        "specific_features": (pc.all_features if pos == Position.K else pc.specific_features),
        # Per-position min-games threshold (None → global MIN_GAMES_PER_SEASON,
        # applied by the caller). Serving must replicate training's train-frame
        # filter or fill_nans means + the StandardScaler drift from the models.
        "min_games_per_season": pc.min_games_per_season,
        "filter_fn": data_mod.filter_to_position,
        "compute_targets_fn": targets_mod.compute_targets,
        "add_features_fn": features_mod.add_specific_features,
        "fill_nans_fn": features_mod.fill_nans,
        "get_feature_columns_fn": features_mod.get_feature_columns,
        # === Model artifact paths ===
        "model_dir": f"src/{pos_lower}/outputs/models",
        "nn_file": f"{pos_lower}_multihead_nn.pt",
        "nn_kwargs": _flat_nn_kwargs(pc),
        # === Attention NN flags + static features (universal) ===
        "train_attention_nn": pc.train_attention_nn,
        "attn_nn_file": f"{pos_lower}_attention_nn.pt",
        "attn_static_features": list(pc.attn_static_features),
        # === LightGBM ===
        "train_lightgbm": pc.train_lightgbm,
    }

    if pos == "K":
        # K is the only nested-attention position. Its serving aggregator uses
        # ``target_signs`` (positive scoring heads, negative miss penalties)
        # instead of the shared ``predictions_to_fantasy_points``.
        spec["compute_adjustment_fn"] = None
        spec["target_signs"] = dict(pc.target_signs) if pc.target_signs else None
        spec["attn_history_structure"] = "nested"
        spec["attn_static_from_df"] = True
        spec["attn_kick_stats"] = list(pc.attn_kick_stats)
        spec["attn_max_games"] = pc.attn_max_games
        spec["attn_max_kicks_per_game"] = pc.attn_max_kicks_per_game
        spec["attn_history_stats"] = list(pc.attn_history_stats)
        spec["attn_nn_kwargs_static"] = _nested_attn_kwargs_static(pc)
    else:
        # Flat-attention positions share the same shape — including DST, whose
        # offense-side opp-attn branch is selected by ``opp_attn_kind``.
        # ``aggregate_fn`` stays present as a None-valued compatibility slot for
        # older serving mocks and any external code still reading the key.
        # app.py no longer invokes it; scoring goes through
        # predictions_to_fantasy_points unless K's target_signs branch applies.
        spec["aggregate_fn"] = None
        spec["attn_history_stats"] = list(pc.attn_history_stats)
        spec["attn_max_seq_len"] = pc.attn_max_seq_len or 17
        spec["opp_attn_history_stats"] = list(pc.opp_attn_history_stats)
        spec["opp_attn_max_seq_len"] = pc.opp_attn_max_seq_len or 17
        spec["opp_attn_kind"] = pc.opp_attn_kind
        spec["attn_nn_kwargs_static"] = _flat_attn_kwargs_static(pc)
        if pos == "DST":
            # DST adds an explicit None for the adjustment-fn slot (mirrors
            # the K branch above so app.py's degraded-mode path can read the
            # key without a KeyError).
            spec["compute_adjustment_fn"] = None
    return spec


class _LazyInferenceRegistry:
    """Dict-like view over ``get_inference_spec`` — ``registry[pos]`` triggers lazy import."""

    def __getitem__(self, pos):
        return get_inference_spec(pos)

    def __contains__(self, pos):
        return pos in ALL_POSITIONS


INFERENCE_REGISTRY = _LazyInferenceRegistry()
