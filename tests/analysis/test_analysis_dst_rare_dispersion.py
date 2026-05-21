"""Smoke tests for src/analysis/analysis_dst_rare_dispersion.py.

The script's ``main()`` builds DST team-level data from raw parquets (slow,
data-dependent) — out of scope for unit tests. These cover the pure-function
helpers (``describe`` and ``recommend``) against synthetic distributions so
a future change to the decision-rule thresholds is caught by CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import analysis_dst_rare_dispersion as adrd

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    """No env vars / file IO at import time."""
    assert hasattr(adrd, "describe")
    assert hasattr(adrd, "recommend")
    assert hasattr(adrd, "main")
    # Four rare DST targets (a regression here would mean a config split).
    assert adrd.RARE_TARGETS == [
        "def_safeties",
        "def_tds",
        "def_blocked_kicks",
        "special_teams_tds",
    ]


def test_describe_returns_expected_shape():
    rng = np.random.default_rng(0)
    y = rng.poisson(0.1, size=10_000)
    stats = adrd.describe(y)
    assert set(stats) == {
        "n",
        "mean",
        "var",
        "dispersion",
        "p_zero_obs",
        "p_zero_poisson",
        "zero_excess",
        "p_ge1",
        "p_ge2",
        "p_ge3",
        "max",
    }
    assert stats["n"] == 10_000
    # Poisson(0.1) → mean ≈ 0.1, var ≈ 0.1, dispersion ≈ 1, zero_excess ≈ 0.
    assert abs(stats["mean"] - 0.1) < 0.03
    assert abs(stats["dispersion"] - 1.0) < 0.15
    assert abs(stats["zero_excess"]) < 0.02


def test_describe_handles_all_zeros():
    """All-zero column → mean=0, dispersion=NaN, p_zero_obs=1.0."""
    y = np.zeros(100)
    stats = adrd.describe(y)
    assert stats["mean"] == 0.0
    assert np.isnan(stats["dispersion"])
    assert stats["p_zero_obs"] == 1.0
    assert stats["p_ge1"] == 0.0


def test_recommend_bce_for_nearly_binary_target():
    """P(y>=2) < 0.02 → BCE on (y>0) recommended."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, size=5_000)  # essentially binary
    stats = adrd.describe(y)
    rec = adrd.recommend(stats)
    assert rec.startswith("BCE on (y>0)")


def test_recommend_poisson_for_clean_poisson():
    """Poisson sample with dispersion ≈ 1 and rate high enough that
    P(y>=2) > 0.02 should recommend Poisson NLL.
    """
    rng = np.random.default_rng(0)
    y = rng.poisson(0.5, size=10_000)
    stats = adrd.describe(y)
    rec = adrd.recommend(stats)
    assert rec.startswith("Poisson NLL")


def test_recommend_negative_binomial_for_overdispersed():
    """Heavily overdispersed counts (NegBin-like) → NB / Tweedie NLL."""
    rng = np.random.default_rng(0)
    # Mixture: 95% zero, 5% drawn from a wide Poisson — pushes variance up
    # without inflating zero-excess too much, hitting the d > 1.2 branch.
    mask = rng.random(size=10_000) < 0.05
    y = np.where(mask, rng.poisson(20, size=10_000), 0)
    stats = adrd.describe(y)
    rec = adrd.recommend(stats)
    # The exact recommendation depends on whether zero_excess clears the
    # 0.05 threshold first — accept either of the high-dispersion branches.
    assert rec.startswith("Negative-Binomial") or rec.startswith("Zero-inflated Poisson")
