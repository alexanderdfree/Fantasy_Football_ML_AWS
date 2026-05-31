"""Unit tests for src/analysis/analysis_feature_audit.py.

Pure-function coverage on synthetic frames with *known* collinearity — no
data splits, no nflverse loader, no position imports (those live behind
``_resolve_position`` and are exercised only by the real CLI run). Mirrors the
direct-import pattern of ``test_covariate_shift.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_feature_audit as fa

pytestmark = pytest.mark.unit


def _collinear_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """x1 ~ N(0,1); x2 = 3*x1 + tiny noise (near-collinear); x3 independent."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = 3.0 * x1 + rng.normal(0, 0.01, n)
    x3 = rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def test_module_imports_cleanly():
    for name in (
        "_present_numeric",
        "_high_corr_pairs",
        "_vif",
        "_condition_number",
        "_spearman_matrix",
        "_classify_condition_number",
        "_decide_drop",
        "_drop_candidates",
        "_save_static_heatmap",
        "_resolve_position",
        "_audit_position",
        "main",
    ):
        assert hasattr(fa, name), f"missing {name}"
    assert {"QB", "RB", "WR", "TE"} == fa.SPLIT_POSITIONS
    assert {"DST", "K"} == fa.DEDICATED_POSITIONS
    assert fa.SIGNAL_TARGETS["DST"] == "def_sacks"  # raw stat, not fantasy_points


def test_vif_flags_collinear_and_spares_independent():
    df = _collinear_frame()
    vif = fa._vif(df, ["x1", "x2", "x3"])
    assert vif["x1"] > fa.DROP_VIF  # both sides of the collinear pair inflate
    assert vif["x2"] > fa.DROP_VIF
    assert vif["x3"] < 2.0  # independent column near the ideal 1.0


def test_vif_small_sample_returns_nan():
    df = _collinear_frame(n=10)
    vif = fa._vif(df, ["x1", "x2", "x3"])
    assert all(np.isnan(v) for v in vif.values())


def test_condition_number_no_pca_returns_none_post():
    df = _collinear_frame()
    pre, post = fa._condition_number(df, ["x1", "x2", "x3"], None)
    assert np.isfinite(pre)
    assert pre > 1.0
    assert post is None


def test_condition_number_pca_improves_conditioning():
    df = _collinear_frame()
    pre, post = fa._condition_number(df, ["x1", "x2", "x3"], 2)
    assert np.isfinite(pre) and post is not None
    assert post < pre  # dropping the near-null direction conditions the matrix


def test_high_corr_pairs_ranks_planted_pair_first():
    df = _collinear_frame()
    pairs = fa._high_corr_pairs(df.corr(), threshold=0.85)
    assert pairs, "expected at least the planted collinear pair"
    a, b, r = pairs[0]
    assert {a, b} == {"x1", "x2"}
    assert abs(r) > 0.99


def test_high_corr_pairs_threshold_excludes_independent():
    df = _collinear_frame()
    pairs = fa._high_corr_pairs(df.corr(), threshold=0.85)
    assert all({a, b} != {"x1", "x3"} and {a, b} != {"x2", "x3"} for a, b, _ in pairs)


def test_high_corr_pairs_skips_nan():
    corr = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert fa._high_corr_pairs(corr, threshold=0.5) == []


def test_drop_candidates_thresholds():
    df = _collinear_frame()
    corr = df.corr()
    vif = fa._vif(df, ["x1", "x2", "x3"])
    cands = fa._drop_candidates(corr, vif, target_signal={}, target_col=None)
    assert len(cands) == 1
    assert {cands[0]["drop"], cands[0]["keep"]} == {"x1", "x2"}


def test_drop_candidates_requires_high_vif():
    # Correlation high but VIF below threshold -> no candidate.
    corr = pd.DataFrame([[1.0, 0.99], [0.99, 1.0]], index=["a", "b"], columns=["a", "b"])
    cands = fa._drop_candidates(corr, {"a": 2.0, "b": 2.0}, target_signal={}, target_col=None)
    assert cands == []


def test_decide_drop_prefers_higher_target_corr():
    drop, keep, reason = fa._decide_drop("a", "b", {"a": 0.5, "b": 0.1}, "yards")
    assert drop == "b" and keep == "a"
    assert "corr-with-yards" in reason


def test_decide_drop_prefers_l3_over_l5():
    drop, keep, _ = fa._decide_drop("foo_L3", "foo_L5", {}, None)
    assert drop == "foo_L5" and keep == "foo_L3"


def test_decide_drop_falls_back_to_longer_name():
    drop, keep, _ = fa._decide_drop("x", "xxxx", {}, None)
    assert drop == "xxxx" and keep == "x"


def test_present_numeric_filters_missing_constant_and_nonnumeric():
    df = pd.DataFrame(
        {
            "good": np.arange(100.0),
            "const": np.ones(100),
            "text": ["a"] * 100,
        }
    )
    present = fa._present_numeric(df, ["good", "const", "text", "absent"])
    assert present == ["good"]


def test_classify_condition_number_boundaries():
    assert fa._classify_condition_number(50.0) == "well-conditioned"
    assert fa._classify_condition_number(5e3) == "moderate"
    assert fa._classify_condition_number(5e5) == "suspect"
    assert fa._classify_condition_number(2e8) == "multicollinear"
    assert fa._classify_condition_number(float("nan")) == "n/a"


def test_spearman_matrix_two_columns():
    df = _collinear_frame()
    m = fa._spearman_matrix(df, ["x1", "x2"])
    assert m.shape == (2, 2)
    assert abs(m.loc["x1", "x2"]) > 0.99


def test_vif_output_is_json_serializable():
    df = _collinear_frame()
    vif = fa._vif(df, ["x1", "x2", "x3"])
    json.dumps(vif)  # native floats only, must not raise
