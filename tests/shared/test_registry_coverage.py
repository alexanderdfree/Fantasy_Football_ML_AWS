"""Coverage tests for ``src/shared/registry.py``.

PR 3 of the consolidation series collapsed the six per-position branches in
``get_inference_spec`` into a single generic dispatcher that reads from
``POSITION_CONFIG``. The legacy ``_POSITION_META`` dict and the per-cfg-module
``_attn_kwargs_static`` helper are gone; metadata now flows from the
dataclass on each position's config module.
"""

from __future__ import annotations

import pytest

from src.shared.position_config import PositionConfig
from src.shared.registry import (
    ALL_POSITIONS,
    CPU_ONLY_POSITIONS,
    INFERENCE_REGISTRY,
    _flat_attn_kwargs_static,
    accepts_dataframes,
    get_config,
    get_cv_runner,
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
# Runner / CV runner / config lookups — triggers lazy import per position
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_get_runner_returns_callable(pos):
    fn = get_runner(pos)
    assert callable(fn)
    assert fn.__name__ == "run"


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])  # all six now have cv runners
def test_get_cv_runner_returns_callable(pos):
    fn = get_cv_runner(pos)
    assert callable(fn)


@pytest.mark.unit
def test_get_cv_runner_raises_on_unknown_position():
    with pytest.raises(ValueError, match="Unknown position"):
        get_cv_runner("XYZ")


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
