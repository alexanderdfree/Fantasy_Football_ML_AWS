"""Unit tests for src/analysis/significance.py.

Pure-stats helpers, so these run fast with no data/network. Cover the decision
the analysis relies on: "model strictly better" must come out significant, and
"identical errors" must come out a tie / CI-contains-zero. Plus determinism,
clustering, and input validation. Import smoke guards against a signature break
surfacing only at PR-review time.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import significance as sig

pytestmark = pytest.mark.unit


def test_module_imports_cleanly() -> None:
    assert hasattr(sig, "diebold_mariano_test")
    assert hasattr(sig, "paired_bootstrap_metric_ci")


# ---------- Diebold-Mariano --------------------------------------------------


def test_dm_model_strictly_better_is_significant() -> None:
    rng = np.random.default_rng(0)
    e_expert = rng.normal(0.0, 5.0, size=300)
    e_model = 0.3 * e_expert  # same sign, smaller magnitude everywhere
    res = sig.diebold_mariano_test(e_model, e_expert, power=1)
    assert res["mean_loss_diff"] < 0  # |e_model| - |e_expert| < 0
    assert res["dm_stat"] < 0
    assert res["p_value"] < 0.01
    assert res["favored"] == "model"
    assert res["n"] == 300


def test_dm_identical_errors_is_tie() -> None:
    e = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    res = sig.diebold_mariano_test(e, e.copy(), power=1)
    assert res["favored"] == "tie"
    assert res["p_value"] == 1.0
    assert res["dm_stat"] == 0.0


def test_dm_power_two_matches_rmse_loss_direction() -> None:
    rng = np.random.default_rng(1)
    e_expert = rng.normal(0.0, 4.0, size=200)
    e_model = 0.5 * e_expert
    res = sig.diebold_mariano_test(e_model, e_expert, power=2)
    assert res["favored"] == "model"
    assert res["p_value"] < 0.05


def test_dm_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sig.diebold_mariano_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_dm_too_few_obs_raises() -> None:
    with pytest.raises(ValueError):
        sig.diebold_mariano_test([1.0], [2.0])


# ---------- Paired bootstrap -------------------------------------------------


def test_bootstrap_model_better_ci_excludes_zero() -> None:
    rng = np.random.default_rng(2)
    e_expert = rng.normal(0.0, 5.0, size=400)
    e_model = 0.3 * e_expert
    res = sig.paired_bootstrap_metric_ci(e_model, e_expert, metric="mae", n_boot=300, seed=7)
    assert res["delta"] < 0  # model MAE < expert MAE
    assert res["hi"] < 0  # whole CI below zero ⇒ significant
    assert res["p_value"] < 0.05
    assert res["metric"] == "mae"


def test_bootstrap_identical_errors_ci_contains_zero() -> None:
    e = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 6.0])
    res = sig.paired_bootstrap_metric_ci(e, e.copy(), metric="rmse", n_boot=200, seed=0)
    assert res["delta"] == 0.0
    assert res["lo"] <= 0.0 <= res["hi"]
    assert res["p_value"] == 1.0


def test_bootstrap_is_deterministic_under_seed() -> None:
    rng = np.random.default_rng(3)
    e_model = rng.normal(0.0, 2.0, size=150)
    e_expert = rng.normal(0.0, 3.0, size=150)
    a = sig.paired_bootstrap_metric_ci(e_model, e_expert, n_boot=200, seed=42)
    b = sig.paired_bootstrap_metric_ci(e_model, e_expert, n_boot=200, seed=42)
    assert a == b


def test_bootstrap_clustered_runs_and_matches_observed_delta() -> None:
    rng = np.random.default_rng(4)
    e_model = rng.normal(0.0, 2.0, size=300)
    e_expert = rng.normal(0.0, 4.0, size=300)
    groups = np.arange(300) // 3  # 100 clusters of 3 player-weeks each
    clustered = sig.paired_bootstrap_metric_ci(
        e_model, e_expert, metric="mae", groups=groups, n_boot=200, seed=1
    )
    plain = sig.paired_bootstrap_metric_ci(e_model, e_expert, metric="mae", n_boot=200, seed=1)
    # Observed point estimate is identical; only the resampling (hence CI) differs.
    assert clustered["delta"] == pytest.approx(plain["delta"])
    assert np.isfinite(clustered["lo"]) and np.isfinite(clustered["hi"])


def test_bootstrap_invalid_metric_raises() -> None:
    with pytest.raises(ValueError):
        sig.paired_bootstrap_metric_ci([1.0, 2.0], [3.0, 4.0], metric="mape")


def test_bootstrap_groups_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        sig.paired_bootstrap_metric_ci([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], groups=[0, 1], n_boot=10)
