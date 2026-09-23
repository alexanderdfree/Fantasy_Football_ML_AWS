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
def test_flat_attn_kwargs_static_populates_gated_targets_when_set():
    kwargs = _flat_attn_kwargs_static(_make_pc(gated_targets=["a", "b"]))
    assert kwargs["gated_targets"] == ["a", "b"]


@pytest.mark.unit
def test_flat_attn_kwargs_static_threads_non_negative_targets():
    nn = {"a", "b"}
    kwargs = _flat_attn_kwargs_static(_make_pc(nn_non_negative_targets=nn))
    assert kwargs["non_negative_targets"] == nn


# --------------------------------------------------------------------------
# ``nn_head_hidden_overrides`` -> ``head_hidden_overrides`` on BOTH builders.
# #1503: K's nested-history builder forwarded every other nn_/attn_ field the
# training factory consumes but dropped this one, so a K per-head override
# would have trained one head shape and served another. Each builder emits the
# key iff the config sets an override, and forwards a COPY — never the config's
# own dict — so a served-kwargs mutation can't leak back into POSITION_CONFIG.
# --------------------------------------------------------------------------

_ATTN_KWARGS_BUILDERS = [
    pytest.param(_flat_attn_kwargs_static, id="flat"),
    pytest.param(_nested_attn_kwargs_static, id="nested"),
]


@pytest.mark.unit
@pytest.mark.parametrize("builder", _ATTN_KWARGS_BUILDERS)
def test_attn_kwargs_static_forwards_head_hidden_overrides_as_a_copy(builder):
    pc = _make_pc(nn_head_hidden_overrides={"a": 8, "b": 16})
    kwargs = builder(pc)
    assert kwargs["head_hidden_overrides"] == pc.nn_head_hidden_overrides
    assert kwargs["head_hidden_overrides"] is not pc.nn_head_hidden_overrides


@pytest.mark.unit
@pytest.mark.parametrize("builder", _ATTN_KWARGS_BUILDERS)
def test_attn_kwargs_static_omits_head_hidden_overrides_when_empty(builder):
    """No override configured -> key absent, so an override-free position's
    served kwargs are unchanged by the conditional forward."""
    assert "head_hidden_overrides" not in builder(_make_pc())


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
# keys. A field resolves to a constructor parameter in this order — an
# explicit alias for the RENAMED knobs (``attn_positional_encoding`` ->
# ``use_positional_encoding``, ``attn_kick_dim`` -> ``d_kick``, ...;
# ``_CONFIG_FIELD_ALIASES``), then the identical name (``gated_targets``),
# then the ``attn_`` prefix convention (``attn_d_model`` -> ``d_model``), then
# the ``nn_`` prefix the shared-NN knobs use (``nn_head_hidden_overrides`` ->
# ``head_hidden_overrides`` — the K/nested drop #1503 caught, invisible to the
# guard until this prefix was mapped). The alias tier exists because a prefix
# strip cannot see a rename: without it the guard silently skipped five
# state_dict- or numerics-affecting forwards (``use_positional_encoding``,
# ``use_gated_fusion``, ``d_kick``, ``n_attn_heads``, and
# ``encoder_hidden_dim`` — the last hidden by a blanket "ends in ``_dim``"
# exemption, since narrowed to the enumerated runtime dims).
#
# Fixture caveat: ``head_hidden_overrides`` is emitted CONDITIONALLY by both
# builders (only when ``nn_head_hidden_overrides`` is non-empty), so the parity
# guards build their fixture with that knob POPULATED
# (``_POPULATED_OPTIONAL_KNOBS``). On an empty fixture the key is legitimately
# absent and the guard would false-positive on a builder that forwards it.
# --------------------------------------------------------------------------

# PositionConfig prefixes a constructor parameter drops: ``attn_`` for the
# attention knobs (``attn_d_model`` -> ``d_model``) and ``nn_`` for the shared
# NN knobs (``nn_head_hidden_overrides`` -> ``head_hidden_overrides``). Identity
# matches win over prefix stripping, so ``attn_dropout`` (a real ctor param) is
# NOT read as ``dropout``.
_CONFIG_FIELD_PREFIXES = ("attn_", "nn_")

# PositionConfig fields whose constructor parameter is a RENAME rather than a
# prefix strip — consulted before the identity / prefix tiers. Without these
# the guard is blind to the forward (``attn_positional_encoding`` strips to
# ``positional_encoding``, which names no parameter). Every value must be a
# real parameter of at least one served constructor (pinned below), else a
# constructor rename would silently retire the alias.
_CONFIG_FIELD_ALIASES = {
    "attn_positional_encoding": "use_positional_encoding",
    "attn_gated_fusion": "use_gated_fusion",
    "attn_kick_dim": "d_kick",
    "attn_encoder_hidden_dim": "encoder_hidden_dim",
    "attn_n_heads": "n_attn_heads",
}

# Constructor parameters that are the net's data-derived feature dimensions,
# injected at build time from the actual array shapes (see
# ``build_multihead_net*``), NOT static knobs the ``_*_attn_kwargs_static``
# builders forward from config — excluded from the mapping. Enumerated rather
# than "ends in ``_dim``": ``encoder_hidden_dim`` (a sizing knob that adds
# encoder layers when > 0) and ``self_attn_ffn_dim`` are real architecture
# parameters a suffix rule would wrongly exempt. K's config carries a cached
# ``attn_kick_dim``, which the alias tier routes to ``d_kick`` (the inner
# attention width) — not to the runtime ``kick_dim`` (= len(attn_kick_stats)).
_RUNTIME_DIM_PARAMS = frozenset({"static_dim", "kick_dim", "game_dim", "opp_game_dim", "input_dim"})

# Optional knobs the parity fixtures populate so the ONE conditionally-emitted
# builder key (``head_hidden_overrides``) is actually exercised. ``gated_targets``
# needs no populating: the flat builder emits that key unconditionally
# (``None`` when unset), and pairing a non-empty list with the default
# ``attn_gated=False`` is a combination ``MultiHeadNetWithHistory`` rejects.
_POPULATED_OPTIONAL_KNOBS = dict(nn_head_hidden_overrides={"a": 8})

_PARITY_CASES = [
    pytest.param(_flat_attn_kwargs_static, MultiHeadNetWithHistory, id="flat"),
    pytest.param(_nested_attn_kwargs_static, MultiHeadNetWithNestedHistory, id="nested"),
]


def _ctor_param_names(ctor) -> set[str]:
    """Keyword parameter names accepted by a net constructor (drops ``self``)."""
    return set(inspect.signature(ctor.__init__).parameters) - {"self"}


def _ctor_param_for_field(fld: str, ctor_params: set[str]) -> str | None:
    """Constructor parameter name a PositionConfig field forwards to, or
    ``None`` when the field maps to no parameter of the constructor.

    Resolution order: explicit alias (``_CONFIG_FIELD_ALIASES``), identical
    name, then one stripped ``_CONFIG_FIELD_PREFIXES`` prefix."""
    alias = _CONFIG_FIELD_ALIASES.get(fld)
    if alias in ctor_params:
        return alias
    if fld in ctor_params:
        return fld
    for prefix in _CONFIG_FIELD_PREFIXES:
        if fld.startswith(prefix) and fld[len(prefix) :] in ctor_params:
            return fld[len(prefix) :]
    return None


def _mappable_ctor_params(ctor) -> set[str]:
    """``ctor``'s parameters minus the runtime feature dims (see
    ``_RUNTIME_DIM_PARAMS``) — the set a config field can legitimately map to."""
    return _ctor_param_names(ctor) - _RUNTIME_DIM_PARAMS


def _config_fields_mapped_to_ctor(ctor) -> set[str]:
    """PositionConfig fields that map to a (non-runtime-dim) parameter of
    ``ctor``. Returns the *PositionConfig field names* so the failure message
    names the field a contributor would have just added."""
    params = _mappable_ctor_params(ctor)
    fields = set(PositionConfig.__dataclass_fields__)
    return {fld for fld in fields if _ctor_param_for_field(fld, params) is not None}


def _unforwarded_fields(ctor, builder_keys: set[str]) -> list[tuple[str, str]]:
    """``(field, expected builder key)`` for every PositionConfig field that
    maps to a ``ctor`` parameter but is absent from ``builder_keys`` — the
    parity guard's verdict, factored out so a simulated drop can be pinned on a
    plain key-set without monkeypatching the builders."""
    params = _mappable_ctor_params(ctor)
    return sorted(
        (fld, _ctor_param_for_field(fld, params))
        for fld in _config_fields_mapped_to_ctor(ctor)
        if _ctor_param_for_field(fld, params) not in builder_keys
    )


@pytest.mark.unit
def test_flat_factory_params_overlap_is_non_empty():
    """Sanity guard on the introspection itself: if the flat constructor or the
    config schema is refactored such that NOTHING overlaps, the parity test
    below would silently pass vacuously. Pin a non-empty mapped set so that
    failure surfaces instead."""
    mapped = _config_fields_mapped_to_ctor(MultiHeadNetWithHistory)
    assert mapped, "no PositionConfig field maps to MultiHeadNetWithHistory params"


@pytest.mark.unit
@pytest.mark.parametrize("ctor", [MultiHeadNetWithHistory, MultiHeadNetWithNestedHistory])
def test_nn_prefixed_override_field_is_visible_to_the_parity_guard(ctor):
    """Pin the ``nn_`` prefix mapping itself: ``nn_head_hidden_overrides`` must
    be recognised as feeding ``head_hidden_overrides`` on BOTH constructors,
    else the parity guards go blind to exactly the drop #1503 found."""
    assert "nn_head_hidden_overrides" in _config_fields_mapped_to_ctor(ctor)
    param = _ctor_param_for_field("nn_head_hidden_overrides", _ctor_param_names(ctor))
    assert param == "head_hidden_overrides"


@pytest.mark.unit
def test_alias_targets_are_real_constructor_parameters():
    """Every alias must name a parameter of at least one served constructor
    (a constructor rename would otherwise silently retire it), and on each
    constructor that takes the parameter the aliased field must resolve to it
    — the alias tier is load-bearing, not shadowed by a prefix-strip miss."""
    ctors = (MultiHeadNetWithHistory, MultiHeadNetWithNestedHistory)
    all_params = set().union(*(_ctor_param_names(c) for c in ctors))
    stale = {f: p for f, p in _CONFIG_FIELD_ALIASES.items() if p not in all_params}
    assert not stale, f"aliases naming no constructor parameter: {stale}"
    for fld, param in _CONFIG_FIELD_ALIASES.items():
        for ctor in ctors:
            params = _mappable_ctor_params(ctor)
            if param in params:
                assert _ctor_param_for_field(fld, params) == param, (fld, ctor.__name__)
                assert fld in _config_fields_mapped_to_ctor(ctor), (fld, ctor.__name__)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "ctor", "dropped_key", "field"),
    [
        pytest.param(
            _flat_attn_kwargs_static,
            MultiHeadNetWithHistory,
            "use_positional_encoding",
            "attn_positional_encoding",
            id="flat-use_positional_encoding",
        ),
        pytest.param(
            _flat_attn_kwargs_static,
            MultiHeadNetWithHistory,
            "use_gated_fusion",
            "attn_gated_fusion",
            id="flat-use_gated_fusion",
        ),
        pytest.param(
            _flat_attn_kwargs_static,
            MultiHeadNetWithHistory,
            "encoder_hidden_dim",
            "attn_encoder_hidden_dim",
            id="flat-encoder_hidden_dim",
        ),
        pytest.param(
            _flat_attn_kwargs_static,
            MultiHeadNetWithHistory,
            "n_attn_heads",
            "attn_n_heads",
            id="flat-n_attn_heads",
        ),
        pytest.param(
            _nested_attn_kwargs_static,
            MultiHeadNetWithNestedHistory,
            "use_positional_encoding",
            "attn_positional_encoding",
            id="nested-use_positional_encoding",
        ),
        pytest.param(
            _nested_attn_kwargs_static,
            MultiHeadNetWithNestedHistory,
            "d_kick",
            "attn_kick_dim",
            id="nested-d_kick",
        ),
        pytest.param(
            _nested_attn_kwargs_static,
            MultiHeadNetWithNestedHistory,
            "encoder_hidden_dim",
            "attn_encoder_hidden_dim",
            id="nested-encoder_hidden_dim",
        ),
    ],
)
def test_parity_guard_names_a_dropped_renamed_forward(builder, ctor, dropped_key, field):
    """Red-side pin for the alias tier: delete one renamed forward from the
    builder's output (on a copy of the key-set — no monkeypatching) and the
    guard's missing-list must name the PositionConfig field behind it. Before
    the alias map + runtime-dim-only exemption every one of these drops was
    invisible to the guard."""
    keys = set(builder(_make_pc(**_POPULATED_OPTIONAL_KNOBS)))
    assert dropped_key in keys, f"{builder.__name__} no longer forwards {dropped_key!r}"
    assert _unforwarded_fields(ctor, keys) == []
    assert _unforwarded_fields(ctor, keys - {dropped_key}) == [(field, dropped_key)]


@pytest.mark.unit
@pytest.mark.parametrize(("builder", "ctor"), _PARITY_CASES)
def test_attn_kwargs_static_forwards_every_mapped_config_field(builder, ctor):
    """Every PositionConfig field that maps to the served constructor's
    parameters must be forwarded by the registry builder that feeds it.

    Passes today; FAILS if someone adds (or renames) a PositionConfig field the
    builder forgets to forward. The nested arm was red on the pre-#1503 builder
    (``nn_head_hidden_overrides`` -> ``head_hidden_overrides`` was dropped)."""
    builder_keys = set(builder(_make_pc(**_POPULATED_OPTIONAL_KNOBS)))
    missing = _unforwarded_fields(ctor, builder_keys)
    assert not missing, (
        f"PositionConfig fields not forwarded by {builder.__name__} "
        f"(field -> expected builder key): {missing}"
    )
