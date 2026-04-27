"""Coverage tests for ``src/shared/registry.py``.

The registry is a lazy dispatch table: each position has metadata in
``_POSITION_META`` + a bespoke branch in ``get_inference_spec`` that wires
up the position's filter/feature/target callables for app.py to consume
at inference time. These tests exercise every lookup + every branch of
``get_inference_spec`` so the serving path's model loading stays green.
"""

from __future__ import annotations

import pytest

from src.shared.registry import (
    ALL_POSITIONS,
    CPU_ONLY_POSITIONS,
    INFERENCE_REGISTRY,
    _attn_kwargs_static,
    _meta,
    accepts_dataframes,
    get_config,
    get_cv_runner,
    get_inference_spec,
    get_runner,
    is_cpu_only,
)

# --------------------------------------------------------------------------
# Meta + lightweight lookups
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
def test_meta_returns_position_dict(pos):
    m = _meta(pos)
    assert m["runner_module"].endswith("run_pipeline")
    assert m["runner_fn"] == "run"
    assert m["config_var"] == "CONFIG"


@pytest.mark.unit
def test_meta_raises_on_unknown_position():
    with pytest.raises(ValueError, match="Unknown position"):
        _meta("FOO")


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
@pytest.mark.parametrize("pos", ["QB", "RB", "WR"])  # only these three have cv runners
def test_get_cv_runner_returns_callable(pos):
    fn = get_cv_runner(pos)
    assert callable(fn)


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["TE", "K", "DST"])
def test_get_cv_runner_raises_for_positions_without_cv(pos):
    with pytest.raises(ValueError, match="CV pipeline not implemented"):
        get_cv_runner(pos)


@pytest.mark.unit
@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_get_config_returns_dict(pos):
    cfg = get_config(pos)
    assert isinstance(cfg, dict)
    assert "targets" in cfg


# --------------------------------------------------------------------------
# get_inference_spec — per-position branches
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
# _attn_kwargs_static — helper used by get_inference_spec
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_attn_kwargs_static_falls_back_to_defaults_for_missing_attrs():
    """When a cfg module doesn't define an attr, the constructor default lands."""

    class _EmptyCfg:
        pass

    kwargs = _attn_kwargs_static(_EmptyCfg())
    assert kwargs["d_model"] == 32  # default
    assert kwargs["n_attn_heads"] == 2  # default
    assert kwargs["head_hidden"] == 32  # default
    assert kwargs["dropout"] == 0.3  # default
    assert kwargs["gated_targets"] is None


@pytest.mark.unit
def test_attn_kwargs_static_populates_head_hidden_overrides_when_set():
    """If ``NN_HEAD_HIDDEN_OVERRIDES`` is truthy it lands as a dict."""

    class _Cfg:
        NN_HEAD_HIDDEN_OVERRIDES = {"passing_yards": 8, "rushing_yards": 16}
        NN_NON_NEGATIVE_TARGETS = {"passing_yards", "rushing_yards"}

    kwargs = _attn_kwargs_static(_Cfg())
    assert kwargs["head_hidden_overrides"] == {"passing_yards": 8, "rushing_yards": 16}
    assert kwargs["non_negative_targets"] == {"passing_yards", "rushing_yards"}


@pytest.mark.unit
def test_attn_kwargs_static_populates_gated_targets_when_set():
    class _Cfg:
        GATED_TARGETS = ["passing_tds", "rushing_tds"]

    kwargs = _attn_kwargs_static(_Cfg())
    assert kwargs["gated_targets"] == ["passing_tds", "rushing_tds"]
