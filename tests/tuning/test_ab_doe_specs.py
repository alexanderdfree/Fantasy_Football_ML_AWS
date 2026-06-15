"""Unit tests for the stacked DoE specs (ab_feature_screen + ab_knob_doe).

No training here (that's the GPU-fleet harness run; local training SIGSEGVs on
the macOS libomp triple-load anyway). Coverage is the pure logic these specs
add on top of the shared ``ab_harness`` execution + #720 PB engine:

  * spec resolution + design shape (1 baseline + 12 Plackett-Burman rows; the
    correct ``expect_ridge_identical`` per spec — feature drops MUST move Ridge,
    attention-only knob arms must keep Ridge byte-identical),
  * the cfg mutators against the REAL runtime cfg shape (no ``include_features``
    key — the #1172 null-result bug): the feature drop still removes columns via
    ``_FAMILY_COLS``; knob arms apply overrides and run the full pipeline,
  * the main-effects estimators recover a single planted effect and leave the
    orthogonal factors at zero (the PB design is balanced 6/6, so the contrast
    is clean).
"""

from __future__ import annotations

import pytest

from src.tuning import ab_feature_screen as F
from src.tuning import ab_harness as H
from src.tuning import ab_knob_doe as K
from src.tuning.attn_knob_experiments import ATTN_KNOBS, KNOB_NAMES, plackett_burman_design

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# spec resolution + design shape
# --------------------------------------------------------------------------- #
def test_feature_screen_resolves_with_pb_rows():
    spec = H.resolve_spec("src.tuning.ab_feature_screen")
    assert spec.dotted == "src.tuning.ab_feature_screen"
    assert spec.positions == ["RB"]
    assert spec.baseline == "baseline"
    # 1 baseline + 12 PB rows
    assert len(spec.variants) == 1 + len(plackett_burman_design(len(F.SCREENED_FAMILIES)))
    assert spec.variants["baseline"].is_baseline_shape  # identity = keep all families
    for name, v in spec.variants.items():
        if name == "baseline":
            continue
        assert v.cfg_mutator is not None
        # MUST move Ridge — a Δ=0 (drop didn't take) fails loud (#1172 fix).
        assert v.expect_ridge_identical is False


def test_knob_doe_resolves_attention_only_must_keep_ridge():
    spec = H.resolve_spec("src.tuning.ab_knob_doe")
    assert spec.positions == ["RB"]
    assert spec.baseline == "baseline"
    assert len(spec.variants) == 1 + len(plackett_burman_design(len(ATTN_KNOBS)))
    # baseline is the identity (production knobs, full pipeline); the PB arms
    # apply knob overrides and are attention-only → Ridge must stay identical.
    assert spec.variants["baseline"].is_baseline_shape
    for name, v in spec.variants.items():
        if name == "baseline":
            continue
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is True


# --------------------------------------------------------------------------- #
# feature-family drop mutator
# --------------------------------------------------------------------------- #
def _runtime_like_cfg() -> dict:
    """A cfg shaped like the REAL runtime pipeline cfg: NO ``include_features``
    key (build_pipeline_config flattens it into get_feature_columns_fn), just the
    two feature paths populated with real RB columns. This is the shape the #1172
    bug silently no-opped on."""
    from src.rb.config import POSITION_CONFIG

    inc = POSITION_CONFIG.include_features
    flat = [c for fam in inc for c in inc[fam]]
    static = [
        c
        for fam in ("prior_season", "matchup", "contextual", "weather_vegas")
        for c in inc.get(fam, [])
    ]
    return {
        "get_feature_columns_fn": (lambda f=list(flat): list(f)),
        "attn_static_features": list(static),
    }


def test_drop_family_removes_cols_without_cfg_include_features():
    """#1172 regression: the runtime cfg has NO ``include_features``, yet dropping
    a family must still remove its columns (sourced from _FAMILY_COLS, not the
    cfg). The buggy version no-opped → this asserts something WAS removed."""
    cfg = _runtime_like_cfg()
    assert "include_features" not in cfg  # the real runtime shape
    before = set(cfg["get_feature_columns_fn"]())
    F._drop_families(cfg, frozenset({"rolling"}))
    after = set(cfg["get_feature_columns_fn"]())
    rolling = F._FAMILY_COLS["rolling"]
    assert rolling  # rolling is populated for RB
    assert before - after  # something was actually removed (a no-op would fail here)
    assert after == before - rolling  # exactly rolling's columns removed


def test_drop_static_family_also_filters_attn_static():
    cfg = _runtime_like_cfg()
    pri = F._FAMILY_COLS["prior_season"]
    assert set(cfg["attn_static_features"]) & pri  # prior_season feeds the static branch
    F._drop_families(cfg, frozenset({"prior_season"}))
    assert not (set(cfg["attn_static_features"]) & pri)  # gone from the static branch
    assert not (set(cfg["get_feature_columns_fn"]()) & pri)  # and the linear/tree path


def test_drop_nonstatic_family_leaves_attn_static_untouched():
    cfg = _runtime_like_cfg()
    static_before = list(cfg["attn_static_features"])
    F._drop_families(cfg, frozenset({"rolling"}))  # rolling is not a static category
    assert cfg["attn_static_features"] == static_before


def test_drop_unknown_family_is_safe_noop():
    """An unscreened/empty family contributes no columns → mutator is a no-op."""
    cfg = {"get_feature_columns_fn": lambda: ["a", "b"], "attn_static_features": ["a"]}
    F._drop_families(cfg, frozenset({"ewma"}))  # ewma is not in SCREENED_FAMILIES
    assert cfg["get_feature_columns_fn"]() == ["a", "b"]
    assert cfg["attn_static_features"] == ["a"]


# --------------------------------------------------------------------------- #
# knob override mutator
# --------------------------------------------------------------------------- #
def test_knob_baseline_is_identity():
    """Baseline = production knobs + FULL pipeline (no model-disabling), so the
    stacked Phase-A reaches the test_df return (#1172 fix)."""
    baseline = K.VARIANTS[0]
    assert baseline.name == "baseline"
    assert baseline.cfg_mutator is None  # identity
    assert baseline.is_baseline_shape


def test_knob_pb_arm_applies_overrides_without_disabling_models():
    pb01 = K.VARIANTS[1]
    assert pb01.name == "pb01"
    cfg = pb01.cfg_mutator({})
    # every screened knob is set to one of its two bounds
    for knob in ATTN_KNOBS:
        assert cfg[knob.name] in (knob.low, knob.high)
    # the #1172 fix: PB arms must NOT disable the non-attn models (that dropped
    # the stacked Phase-A into the early-return path that omits test_df).
    assert "train_lightgbm" not in cfg
    assert "train_base_nn" not in cfg


# --------------------------------------------------------------------------- #
# main-effects estimators recover a planted effect on the orthogonal design
# --------------------------------------------------------------------------- #
def test_feature_main_effects_recovers_single_planted_effect():
    design = plackett_burman_design(len(F.SCREENED_FAMILIES))
    target = "rolling"
    # synthetic: dropping `target` adds +0.1 MAE; all else flat. Same drop logic
    # the estimator reconstructs internally, evaluated on two identical seeds.
    variant_seed_mae: dict[str, dict[int, float]] = {}
    for idx, signs in enumerate(design, start=1):
        drop = {fam for fam, s in zip(F.SCREENED_FAMILIES, signs, strict=True) if s < 0}
        mae = 1.0 + (0.1 if target in drop else 0.0)
        variant_seed_mae[f"pb{idx:02d}"] = {42: mae, 43: mae}

    effects = F.feature_main_effects(variant_seed_mae)
    assert effects[target]["mean_effect"] == pytest.approx(0.1)
    assert effects[target]["std_effect"] == pytest.approx(0.0)  # identical seeds
    # the orthogonal PB columns leave every other family's contrast at zero
    for fam in F.SCREENED_FAMILIES:
        if fam != target:
            assert effects[fam]["mean_effect"] == pytest.approx(0.0, abs=1e-9)


def test_knob_main_effects_recovers_single_planted_effect():
    design = plackett_burman_design(len(ATTN_KNOBS))
    target = "attn_lr"
    target_idx = KNOB_NAMES.index(target)
    variant_seed_mae: dict[str, dict[int, float]] = {}
    for idx, signs in enumerate(design, start=1):
        # plant the effect on the HIGH bound of the target knob
        mae = 1.0 + (0.1 if signs[target_idx] > 0 else 0.0)
        variant_seed_mae[f"pb{idx:02d}"] = {42: mae, 43: mae}

    effects = K.knob_main_effects(variant_seed_mae)
    assert effects[target]["mean_effect"] == pytest.approx(0.1)
    for name in KNOB_NAMES:
        if name != target:
            assert effects[name]["mean_effect"] == pytest.approx(0.0, abs=1e-9)
