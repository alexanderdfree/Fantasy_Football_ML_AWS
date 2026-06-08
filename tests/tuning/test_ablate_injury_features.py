"""Unit tests for src/tuning/ablate_injury_features.py.

``_execute_injury_job`` / ``main`` run the full per-position pipeline (slow,
data-dependent) and are out of scope for unit tests. These cover the pure
config-surgery helpers — the feature drop and the drop-summary — against a
synthetic cfg, locking in that the injury/return features are removed from BOTH
model paths and that the original cfg is left untouched. Also asserts importing
the module fires no pipeline.

New-shape specifics tested here (vs. the old sequential script):
- VARIANTS dict contains exactly "with" and "without" keys.
- ``_build_jobs`` produces one AblationJob per (position × seed × variant) and
  sets run_fn to the module's executor — without running any pipeline.
- ``_dropped_summary`` warning path: a cfg with no injury features produces
  empty lists in both paths.
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


def _fake_cfg_no_injury() -> dict:
    """Config that carries no injury features in either path."""
    return {
        "get_feature_columns_fn": lambda: ["a", "b"],
        "attn_static_features": ["c", "d"],
    }


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------


def test_module_imports_without_running_pipeline():
    # Old entry-points gone; new harness symbols present.
    assert hasattr(aif, "_drop_injury_features")
    assert hasattr(aif, "_execute_injury_job")
    assert hasattr(aif, "_build_jobs")
    assert hasattr(aif, "main")


# ---------------------------------------------------------------------------
# INJURY_FEATURES constant
# ---------------------------------------------------------------------------


def test_injury_features_constant_covers_all_four():
    assert set(aif.INJURY_FEATURES) == {
        "game_status",
        "practice_status",
        "days_rest",
        "is_returning_from_absence",
    }


# ---------------------------------------------------------------------------
# _drop_injury_features
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _dropped_summary
# ---------------------------------------------------------------------------


def test_dropped_summary_reports_each_path():
    cfg = _fake_cfg()
    drop = aif._drop_injury_features(cfg)
    summary = aif._dropped_summary(cfg, drop)
    assert set(summary["linear_tree_dropped"]) == {"game_status", "days_rest"}
    assert set(summary["attn_static_dropped"]) == {"game_status", "practice_status"}


def test_dropped_summary_empty_when_no_injury_features_present():
    """A cfg with no injury features → both lists are empty (the NO-OP warning case)."""
    cfg = _fake_cfg_no_injury()
    drop = aif._drop_injury_features(cfg)
    summary = aif._dropped_summary(cfg, drop)
    assert summary["linear_tree_dropped"] == []
    assert summary["attn_static_dropped"] == []


# ---------------------------------------------------------------------------
# VARIANTS dict + variant key constants
# ---------------------------------------------------------------------------


def test_variants_dict_has_with_and_without():
    assert aif.VARIANT_WITH in aif.VARIANTS
    assert aif.VARIANT_WITHOUT in aif.VARIANTS
    assert set(aif.VARIANTS) == {aif.VARIANT_WITH, aif.VARIANT_WITHOUT}


def test_default_variants_tuple_matches_variant_keys():
    assert set(aif.DEFAULT_VARIANTS) == set(aif.VARIANTS)


# ---------------------------------------------------------------------------
# _build_jobs — structure only, no pipeline execution
# ---------------------------------------------------------------------------


def test_build_jobs_produces_correct_count(monkeypatch):
    """_build_jobs(positions=['QB'], seeds=[42,43], variants=['with','without'])
    should yield 1 position × 2 seeds × 2 variants = 4 jobs."""

    # Patch get_config so no real import happens.
    fake_base = _fake_cfg()
    monkeypatch.setattr(aif, "get_config", lambda pos: fake_base)

    jobs = aif._build_jobs(
        positions=["QB"],
        seeds=[42, 43],
        variants=[aif.VARIANT_WITH, aif.VARIANT_WITHOUT],
    )
    assert len(jobs) == 4  # 1 pos × 2 seeds × 2 variants


def test_build_jobs_run_fn_is_execute_injury_job(monkeypatch):
    """Every job's run_fn must point at _execute_injury_job (not an old helper)."""
    fake_base = _fake_cfg()
    monkeypatch.setattr(aif, "get_config", lambda pos: fake_base)

    jobs = aif._build_jobs(
        positions=["RB"],
        seeds=[42],
        variants=[aif.VARIANT_WITH, aif.VARIANT_WITHOUT],
    )
    for job in jobs:
        assert job.run_fn is aif._execute_injury_job


def test_build_jobs_covers_both_variants(monkeypatch):
    """Both 'with' and 'without' variants appear in the job list."""
    fake_base = _fake_cfg()
    monkeypatch.setattr(aif, "get_config", lambda pos: fake_base)

    jobs = aif._build_jobs(
        positions=["WR"],
        seeds=[42],
        variants=[aif.VARIANT_WITH, aif.VARIANT_WITHOUT],
    )
    variant_set = {job.variant for job in jobs}
    assert variant_set == {aif.VARIANT_WITH, aif.VARIANT_WITHOUT}


def test_build_jobs_base_cfg_is_untouched_original(monkeypatch):
    """job.base_cfg must be the raw production config — not the mutated drop_cfg —
    so that _execute_injury_job can apply the drop itself per-variant."""
    fake_base = _fake_cfg()
    monkeypatch.setattr(aif, "get_config", lambda pos: fake_base)

    jobs = aif._build_jobs(
        positions=["TE"],
        seeds=[42],
        variants=[aif.VARIANT_WITH, aif.VARIANT_WITHOUT],
    )
    for job in jobs:
        # The base_cfg's get_feature_columns_fn must still return the injury features —
        # confirming it was not mutated by the drop computation inside _build_jobs.
        cols = job.base_cfg["get_feature_columns_fn"]()
        assert "game_status" in cols
        assert "days_rest" in cols
