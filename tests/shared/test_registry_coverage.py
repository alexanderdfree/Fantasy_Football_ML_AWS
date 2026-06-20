"""Coverage tests for ``src/shared/registry.py``.

PR 3 of the consolidation series collapsed the six per-position branches in
``get_inference_spec`` into a single generic dispatcher that reads from
``POSITION_CONFIG``. The legacy ``_POSITION_META`` dict and the per-cfg-module
``_attn_kwargs_static`` helper are gone; metadata now flows from the
dataclass on each position's config module.
"""

from __future__ import annotations

import inspect

import pytest

from src.shared.neural_net import (
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)
from src.shared.position_config import PositionConfig
from src.shared.registry import (
    ALL_POSITIONS,
    CPU_ONLY_POSITIONS,
    INFERENCE_REGISTRY,
    _flat_attn_kwargs_static,
    _nested_attn_kwargs_static,
    accepts_dataframes,
    get_config,
    get_inference_spec,
    get_runner,
    is_cpu_only,
)

# --------------------------------------------------------------------------
# Position metadata + lightweight lookups
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_all_positions_match_expected_set():
    """Registry lists exactly the six position codes, in canonical order."""
    assert ALL_POSITIONS == ["QB", "RB", "WR", "TE", "K", "DST"]


@pytest.mark.unit
def test_cpu_only_positions_is_k_and_dst():
    """Only K and DST run on CPU in Batch; the rest need GPU."""
    assert {"K", "DST"} == CPU_ONLY_POSITIONS


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_is_cpu_only_flag_matches_set(pos):
    assert is_cpu_only(pos) == (pos in CPU_ONLY_POSITIONS)


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_standard_positions_accept_dataframes(pos):
    assert accepts_dataframes(pos) is True


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["K", "DST"])
def test_special_positions_do_not_accept_dataframes(pos):
    assert accepts_dataframes(pos) is False


@pytest.mark.unit
def test_is_cpu_only_raises_on_unknown_position():
    with pytest.raises(ValueError, match="Unknown position"):
        is_cpu_only("FOO")


# --------------------------------------------------------------------------
# Runner / config lookups — triggers lazy import per position
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_get_runner_returns_callable(pos):
    fn = get_runner(pos)
    assert callable(fn)
    assert fn.__name__ == "run"


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_get_config_returns_dict(pos):
    cfg = get_config(pos)
    assert isinstance(cfg, dict)
    assert "targets" in cfg


# --------------------------------------------------------------------------
# get_inference_spec — universal + position-specific shape
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_get_inference_spec_returns_all_required_keys(pos):
    spec = get_inference_spec(pos)
    # Core keys every branch must return
    for key in (
        "targets",
        "specific_features",
        "filter_fn",
        "compute_targets_fn",
        "add_features_fn",
        "fill_nans_fn",
        "get_feature_columns_fn",
        "model_dir",
        "nn_file",
        "nn_kwargs",
        "train_attention_nn",
        "attn_nn_file",
        "attn_nn_kwargs_static",
    ):
        assert key in spec, f"{pos}: spec missing {key!r}"


@pytest.mark.unit
def test_get_inference_spec_k_has_nested_attention_and_target_signs():
    """K's spec is the only one with nested-history attention + target_signs."""
    spec = get_inference_spec("K")
    assert spec["attn_history_structure"] == "nested"
    assert spec["attn_static_from_df"] is True
    assert "target_signs" in spec
    assert spec["target_signs"] == {
        "fg_yard_points": 1.0,
        "pat_points": 1.0,
        "fg_misses": -1.0,
        "xp_misses": -1.0,
    }


@pytest.mark.unit
def test_get_inference_spec_dst_uses_offense_opp_attn_kind():
    spec = get_inference_spec("DST")
    assert spec["opp_attn_kind"] == "offense"


@pytest.mark.unit
def test_get_inference_spec_raises_on_unknown_position():
    with pytest.raises(ValueError, match="Unknown position"):
        get_inference_spec("ZZZ")


# --------------------------------------------------------------------------
# INFERENCE_REGISTRY — dict-like view
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_inference_registry_getitem_delegates_to_get_inference_spec():
    spec = INFERENCE_REGISTRY["QB"]
    assert isinstance(spec, dict)
    assert "targets" in spec


@pytest.mark.unit
def test_inference_registry_contains_all_positions():
    for pos in ALL_POSITIONS:
        assert pos in INFERENCE_REGISTRY
    assert "FOO" not in INFERENCE_REGISTRY


# --------------------------------------------------------------------------
# _flat_attn_kwargs_static — helper used by get_inference_spec for the
# five flat-attention positions (QB/RB/WR/TE/DST).
# --------------------------------------------------------------------------


def _make_pc(**overrides) -> PositionConfig:
    """Build a minimal PositionConfig for kwargs-extraction tests."""
    base = dict(
        # PR 6 added Position-enum validation on PositionConfig.name, so we
        # piggyback on QB rather than an arbitrary placeholder string.
        name="QB",
        targets=["a"],
        specific_features=[],
        ridge_alpha_grids={"a": [1.0]},
        nn_backbone_layers=[8],
        loss_weights={"a": 1.0},
        head_losses={"a": "huber"},
        huber_deltas={"a": 1.0},
    )
    base.update(overrides)
    return PositionConfig(**base)


@pytest.mark.unit
def test_flat_attn_kwargs_static_uses_dataclass_defaults():
    kwargs = _flat_attn_kwargs_static(_make_pc())
    assert kwargs["d_model"] == 32
    assert kwargs["n_attn_heads"] == 2
    assert kwargs["head_hidden"] == 32
    assert kwargs["gated_targets"] is None


@pytest.mark.unit
def test_flat_attn_kwargs_static_populates_head_hidden_overrides_when_set():
    overrides = {"a": 8, "b": 16}
    kwargs = _flat_attn_kwargs_static(_make_pc(nn_head_hidden_overrides=overrides))
    assert kwargs["head_hidden_overrides"] == overrides


@pytest.mark.unit
def test_flat_attn_kwargs_static_populates_gated_targets_when_set():
    kwargs = _flat_attn_kwargs_static(_make_pc(gated_targets=["a", "b"]))
    assert kwargs["gated_targets"] == ["a", "b"]


@pytest.mark.unit
def test_flat_attn_kwargs_static_threads_non_negative_targets():
    nn = {"a", "b"}
    kwargs = _flat_attn_kwargs_static(_make_pc(nn_non_negative_targets=nn))
    assert kwargs["non_negative_targets"] == nn


# --------------------------------------------------------------------------
# Factory <-> registry-builder parity (audit #362 F9/F10).
#
# ``get_inference_spec`` rebuilds each served attention NN from
# ``POSITION_CONFIG`` via ``_flat_attn_kwargs_static`` /
# ``_nested_attn_kwargs_static``. The constructors those kwargs feed
# (``MultiHeadNetWithHistory`` / ``MultiHeadNetWithNestedHistory`` — the
# classes the ``build_multihead_net*`` factories forward to) accept knobs like
# ``learn_attn_temperature``/``history_dropout``/``use_swiglu_encoder``/
# ``attn_entropy_coeff``/``use_alibi_bias``/``self_attn_*`` that are currently
# latent (no PositionConfig field drives them, so they sit at their constructor
# defaults). ``condition_queries_on_static`` is no longer latent — #1198 added
# the ``attn_condition_queries_on_static`` field (enabled for RB/WR/TE).
#
# The drift these tests guard: someone promotes one of those latent knobs to a
# PositionConfig field but forgets to forward it through the registry builder,
# so production training (which reads the cfg dict in ``build_multihead_net*``)
# diverges from serving (which reads the registry builder output) — the served
# state_dict would no longer match. The assertion: every PositionConfig field
# that maps to a constructor parameter MUST appear in the builder's output
# keys. Both spellings are checked — the direct name overlap the task names
# (``set(fields) & set(ctor_params)``) AND the ``attn_``-prefixed convention
# (PositionConfig ``attn_d_model`` -> constructor ``d_model``), so the guard
# covers the attention kwargs rather than only the two names that happen to be
# spelled identically.
# --------------------------------------------------------------------------


def _ctor_param_names(ctor) -> set[str]:
    """Keyword parameter names accepted by a net constructor (drops ``self``)."""
    return set(inspect.signature(ctor.__init__).parameters) - {"self"}


def _config_fields_mapped_to_ctor(ctor) -> set[str]:
    """PositionConfig fields that map to a parameter of ``ctor``.

    A field maps either by identical name (``gated_targets`` ->
    ``gated_targets``) or by stripping the ``attn_`` prefix the config uses for
    attention knobs (``attn_d_model`` -> ``d_model``). Returns the
    *PositionConfig field names* so the failure message names the field a
    contributor would have just added.

    Constructor params ending in ``_dim`` (``kick_dim``/``opp_dim``/
    ``history_dim``/``static_dim``/``input_dim``) are excluded: those are the
    net's data-derived feature dimensions, injected at build time from the
    actual array shapes (see ``build_multihead_net*``), NOT static knobs the
    ``_*_attn_kwargs_static`` builders forward from config. K's config carries a
    cached ``attn_kick_dim``, so without this exclusion the heuristic would
    false-positive on ``attn_kick_dim`` -> ``kick_dim``.
    """
    params = {p for p in _ctor_param_names(ctor) if not p.endswith("_dim")}
    fields = set(PositionConfig.__dataclass_fields__)
    mapped = set()
    for fld in fields:
        if fld in params or fld.startswith("attn_") and fld[len("attn_") :] in params:
            mapped.add(fld)
    return mapped


def _ctor_param_for_field(fld: str, ctor_params: set[str]) -> str:
    """Constructor parameter name a PositionConfig field forwards to."""
    if fld in ctor_params:
        return fld
    return fld[len("attn_") :]


@pytest.mark.unit
def test_flat_factory_params_overlap_is_non_empty():
    """Sanity guard on the introspection itself: if the flat constructor or the
    config schema is refactored such that NOTHING overlaps, the parity test
    below would silently pass vacuously. Pin a non-empty mapped set so that
    failure surfaces instead."""
    mapped = _config_fields_mapped_to_ctor(MultiHeadNetWithHistory)
    assert mapped, "no PositionConfig field maps to MultiHeadNetWithHistory params"


@pytest.mark.unit
def test_flat_attn_kwargs_static_forwards_every_mapped_config_field():
    """Every PositionConfig field that maps to a ``MultiHeadNetWithHistory``
    constructor parameter must be forwarded by ``_flat_attn_kwargs_static``.

    Passes today; FAILS if someone adds a PositionConfig field the flat
    registry builder forgets to forward."""
    builder_keys = set(_flat_attn_kwargs_static(_make_pc()))
    ctor_params = _ctor_param_names(MultiHeadNetWithHistory)
    forwarded = {_ctor_param_for_field(f, ctor_params) for f in builder_keys & ctor_params}
    # Account for the ``attn_``-prefix mapping on both sides: a field
    # ``attn_d_model`` is "forwarded" iff the builder emits ``d_model``.
    forwarded |= builder_keys

    missing = []
    for fld in _config_fields_mapped_to_ctor(MultiHeadNetWithHistory):
        param = _ctor_param_for_field(fld, ctor_params)
        if param not in builder_keys:
            missing.append((fld, param))
    assert not missing, (
        "PositionConfig fields not forwarded by _flat_attn_kwargs_static "
        f"(field -> expected builder key): {missing}"
    )


@pytest.mark.unit
def test_nested_attn_kwargs_static_forwards_every_mapped_config_field():
    """Same parity guard for K's nested-history builder /
    ``MultiHeadNetWithNestedHistory`` constructor."""
    builder_keys = set(_nested_attn_kwargs_static(_make_pc()))
    ctor_params = _ctor_param_names(MultiHeadNetWithNestedHistory)

    missing = []
    for fld in _config_fields_mapped_to_ctor(MultiHeadNetWithNestedHistory):
        param = _ctor_param_for_field(fld, ctor_params)
        if param not in builder_keys:
            missing.append((fld, param))
    assert not missing, (
        "PositionConfig fields not forwarded by _nested_attn_kwargs_static "
        f"(field -> expected builder key): {missing}"
    )
