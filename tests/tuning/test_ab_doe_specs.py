"""Unit tests for the stacked DoE specs (ab_feature_screen + ab_knob_doe).

No training here (that's the GPU-fleet harness run; local training SIGSEGVs on
the macOS libomp triple-load anyway). Coverage is the pure logic these specs
add on top of the shared ``ab_harness`` execution + #720 PB engine:

  * spec resolution + design shape (1 baseline + 12 Plackett-Burman rows; the
    correct ``expect_ridge_identical`` per spec — feature drops are report-only,
    attention-only knob arms must keep Ridge byte-identical),
  * the cfg mutators (feature-family drop filters BOTH model paths and no-ops a
    family absent for the position; knob arms apply overrides + disable the
    non-attention models),
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
def test_feature_screen_resolves_with_pb_rows_report_only():
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
        # report-only: an all-empty-family drop for some position would false-trip
        # a "must differ" assertion (see module docstring).
        assert v.expect_ridge_identical is None


def test_knob_doe_resolves_attention_only_must_keep_ridge():
    spec = H.resolve_spec("src.tuning.ab_knob_doe")
    assert spec.positions == ["RB"]
    assert spec.baseline == "baseline"
    assert len(spec.variants) == 1 + len(plackett_burman_design(len(ATTN_KNOBS)))
    # every arm (baseline included) disables non-attn models, so none is identity-shaped;
    # all are attention-only → Ridge must stay byte-identical.
    for v in spec.variants.values():
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is True


# --------------------------------------------------------------------------- #
# feature-family drop mutator
# --------------------------------------------------------------------------- #
def _skill_cfg() -> dict:
    """A skill-position-shaped cfg: category dict + both model feature paths."""
    return {
        "include_features": {
            "rolling": ["rolling_x", "rolling_y"],
            "prior_season": ["prior_x"],  # a DEFAULT_ATTN_STATIC category
            "share": ["share_x"],
            "ewma": [],  # empty for this position
        },
        "get_feature_columns_fn": lambda: ["rolling_x", "rolling_y", "prior_x", "share_x"],
        "attn_static_features": ["prior_x"],  # only static categories feed this branch
    }


def test_drop_nonstatic_family_moves_only_linear_path():
    cfg = F._drop_families(_skill_cfg(), frozenset({"rolling"}))
    assert cfg["get_feature_columns_fn"]() == ["prior_x", "share_x"]  # rolling cols gone
    assert cfg["attn_static_features"] == ["prior_x"]  # rolling isn't static → unchanged
    assert cfg["include_features"]["rolling"] == []


def test_drop_static_family_moves_both_paths():
    cfg = F._drop_families(_skill_cfg(), frozenset({"prior_season"}))
    assert cfg["get_feature_columns_fn"]() == ["rolling_x", "rolling_y", "share_x"]
    assert cfg["attn_static_features"] == []  # prior_x is a static feature → dropped


def test_drop_empty_family_is_a_noop():
    cfg = F._drop_families(_skill_cfg(), frozenset({"ewma"}))
    assert cfg["get_feature_columns_fn"]() == ["rolling_x", "rolling_y", "prior_x", "share_x"]
    assert cfg["attn_static_features"] == ["prior_x"]


def test_drop_on_flat_position_cfg_is_safe():
    """K/DST carry no include_features dict → mutator is a no-op, not a crash."""
    cfg = {"get_feature_columns_fn": lambda: ["a", "b"], "attn_static_features": ["a"]}
    out = F._drop_families(cfg, frozenset({"rolling"}))
    assert out["get_feature_columns_fn"]() == ["a", "b"]
    assert out["attn_static_features"] == ["a"]


# --------------------------------------------------------------------------- #
# knob override mutator
# --------------------------------------------------------------------------- #
def test_knob_baseline_disables_non_attn_without_overriding_knobs():
    baseline = K.VARIANTS[0]
    assert baseline.name == "baseline"
    cfg = baseline.cfg_mutator({"attn_d_model": 32})
    assert cfg["train_lightgbm"] is False
    assert cfg["train_base_nn"] is False
    assert cfg["train_elasticnet"] is False
    assert cfg["train_ridge"] is True
    assert cfg["attn_d_model"] == 32  # production value untouched on the baseline arm


def test_knob_pb_arm_applies_overrides_and_disables_non_attn():
    pb01 = K.VARIANTS[1]
    assert pb01.name == "pb01"
    cfg = pb01.cfg_mutator({})
    assert cfg["train_lightgbm"] is False
    # every screened knob is set to one of its two bounds
    for knob in ATTN_KNOBS:
        assert cfg[knob.name] in (knob.low, knob.high)


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
