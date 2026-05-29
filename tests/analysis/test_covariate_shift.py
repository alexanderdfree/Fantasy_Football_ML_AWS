"""Unit tests for src/analysis/covariate_shift.py (train->test shift guard).

Pure-function coverage on synthetic arrays/frames — no data splits, no nflverse loader.
``shift_report_for_position`` (which calls ``build_position_features``) is out of scope for
unit tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import covariate_shift as cs

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    assert hasattr(cs, "ood_scaler_stats")
    assert hasattr(cs, "population_stability_index")
    assert hasattr(cs, "compute_feature_shift")
    assert hasattr(cs, "gate_check")
    assert hasattr(cs, "main")
    assert cs.OOD_CLIP_K == 4.0  # mirrors FEATURE_CLIP


def test_depth_chart_sentinel_is_flagged():
    """Pins the regression trigger: train real ranks, test all -1 -> mean_z ~ -4, flagged."""
    rng = np.random.default_rng(0)
    train = rng.normal(1.25, 0.56, 4000)
    test = np.full(400, -1.0)
    ood = cs.ood_scaler_stats(train, test)
    assert ood["mean_z"] < -3.0
    assert ood["frac_beyond_k"] > 0.5

    train_df = pd.DataFrame({"depth_chart_rank": train})
    test_df = pd.DataFrame({"depth_chart_rank": test})
    recs = cs.compute_feature_shift(train_df, test_df, ["depth_chart_rank"])
    assert recs[0]["flagged"] is True


def test_imputed_depth_chart_not_flagged():
    """After impute (test == train mean) the standardized mean is ~0 -> not flagged."""
    rng = np.random.default_rng(1)
    train = rng.normal(1.25, 0.56, 4000)
    test = np.full(400, train.mean())
    ood = cs.ood_scaler_stats(train, test)
    assert abs(ood["mean_z"]) < cs.MEAN_Z_THRESHOLD


def test_constant_feature_not_flagged():
    """A feature constant in both train and test standardizes to 0 (auto-exempt)."""
    train_df = pd.DataFrame({"const": np.ones(500)})
    test_df = pd.DataFrame({"const": np.ones(50)})
    recs = cs.compute_feature_shift(train_df, test_df, ["const"])
    assert recs[0]["mean_z"] == pytest.approx(0.0)
    assert recs[0]["constant"] is True
    assert recs[0]["flagged"] is False


def test_psi_division_by_zero_guarded():
    """Disjoint train/test ranges must give a finite PSI (eps-floored), not inf/nan."""
    train = np.linspace(0, 1, 1000)
    test = np.linspace(5, 6, 200)  # entirely outside train support
    out = cs.population_stability_index(train, test)
    assert np.isfinite(out["psi"])
    assert out["psi"] > cs.PSI_WARN  # large shift => significant band
    assert out["band"] == "significant"


def test_psi_bands_none_for_same_distribution():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, 5000)
    test = rng.normal(0, 1, 2000)
    out = cs.population_stability_index(train, test)
    assert out["band"] == "none"
    assert out["psi"] < 0.1


def test_in_distribution_feature_not_flagged():
    rng = np.random.default_rng(3)
    train_df = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    test_df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    recs = cs.compute_feature_shift(train_df, test_df, ["x"])
    assert recs[0]["flagged"] is False


def test_gate_check_respects_allowlist():
    rng = np.random.default_rng(4)
    train_df = pd.DataFrame({"shifty": rng.normal(0, 1, 4000)})
    test_df = pd.DataFrame({"shifty": np.full(400, 6.0)})  # mean_z ~ +6
    # Without allowlist -> flagged; gate fails.
    recs = cs.compute_feature_shift(train_df, test_df, ["shifty"])
    assert recs[0]["flagged"] is True
    ok, offenders = cs.gate_check({"features": recs})
    assert ok is False and len(offenders) == 1
    # With allowlist -> not flagged; gate passes.
    recs_allow = cs.compute_feature_shift(train_df, test_df, ["shifty"], allowlist={"shifty"})
    assert recs_allow[0]["flagged"] is False
    ok2, _ = cs.gate_check({"features": recs_allow})
    assert ok2 is True


def test_few_test_rows_sets_low_n():
    rng = np.random.default_rng(5)
    train_df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    test_df = pd.DataFrame({"x": rng.normal(0, 1, 10)})  # < LOW_N_THRESHOLD
    recs = cs.compute_feature_shift(train_df, test_df, ["x"])
    assert recs[0]["low_n"] is True


def test_missing_feature_reported_not_raised():
    train_df = pd.DataFrame({"x": np.arange(100.0)})
    test_df = pd.DataFrame({"x": np.arange(20.0)})  # no "y"
    recs = cs.compute_feature_shift(train_df, test_df, ["x", "y"])
    by_name = {r["feature"]: r for r in recs}
    assert by_name["y"]["missing"] is True
    assert by_name["y"]["flagged"] is True


def test_compute_feature_shift_json_serializable():
    rng = np.random.default_rng(6)
    train_df = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(2, 1, 1000)})
    test_df = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(2, 1, 200)})
    recs = cs.compute_feature_shift(train_df, test_df, ["a", "b"], val_df=test_df)
    json.dumps(recs)  # must not raise (native types only)
