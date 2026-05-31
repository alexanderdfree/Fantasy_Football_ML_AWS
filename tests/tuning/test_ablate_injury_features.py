"""Smoke + unit tests for src/tuning/ablate_injury_features.py.

``ablate_position`` / ``run_variant`` run the full per-position pipeline twice
(slow, data-dependent) and are out of scope for unit tests. These cover the pure
config-surgery helpers — the feature drop and the drop-summary — against a
synthetic cfg, locking in that the injury/return features are removed from BOTH
model paths and that the original cfg is left untouched. Also asserts importing
the module fires no pipeline.
"""

from __future__ import annotations

import pytest

from src.tuning import ablate_injury_features as aif

pytestmark = pytest.mark.unit


def _fake_cfg() -> dict:
    # ``a``/``b`` are non-injury features that must survive; the rest are the
    # features under test. Mirrors the real cfg's two relevant keys.
    return {
        "get_feature_columns_fn": lambda: ["a", "game_status", "days_rest", "b"],
        "attn_static_features": ["game_status", "c", "practice_status"],
    }


def test_module_imports_without_running_pipeline():
    assert hasattr(aif, "ablate_position")
    assert hasattr(aif, "_drop_injury_features")


def test_drop_removes_injury_features_from_both_paths():
    cfg = _fake_cfg()
    drop = aif._drop_injury_features(cfg)

    # linear/tree path: injury features gone, others preserved in order.
    assert drop["get_feature_columns_fn"]() == ["a", "b"]
    # attention static path: injury features gone, others preserved.
    assert drop["attn_static_features"] == ["c"]


def test_drop_does_not_mutate_original_cfg():
    cfg = _fake_cfg()
    aif._drop_injury_features(cfg)
    # Original must be untouched (deep-copy), else a baseline run would train on
    # the ablated feature set — the silent MAE Δ=0 failure mode.
    assert cfg["get_feature_columns_fn"]() == ["a", "game_status", "days_rest", "b"]
    assert cfg["attn_static_features"] == ["game_status", "c", "practice_status"]


def test_dropped_summary_reports_each_path():
    cfg = _fake_cfg()
    drop = aif._drop_injury_features(cfg)
    summary = aif._dropped_summary(cfg, drop)
    assert set(summary["linear_tree_dropped"]) == {"game_status", "days_rest"}
    assert set(summary["attn_static_dropped"]) == {"game_status", "practice_status"}


def test_injury_features_constant_covers_all_four():
    assert set(aif.INJURY_FEATURES) == {
        "game_status",
        "practice_status",
        "days_rest",
        "is_returning_from_absence",
    }
