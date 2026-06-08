"""Unit tests for ``src.analysis.rmse_gap_decomposition``.

``src/analysis`` is excluded from the coverage denominator (codecov.yml), but the
operator-CLI convention (AGENTS.md) is an import-smoke + pure-helper test so signature
drift fails the unit shard rather than PR review. The decomposition identity
(``sum_t mean(r*c_t) == MSE``) and the calibration math (``recal_rmse``) are load-bearing
for the diagnosis, so they get real assertions here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import rmse_gap_decomposition as mod


@pytest.mark.unit
def test_import_smoke():
    """Module imports and the public-ish helpers exist (guards signature drift)."""
    for name in ("_weights", "_signal_stats", "_decompose", "_model_loss_labels", "_main"):
        assert hasattr(mod, name), f"missing {name}"


@pytest.mark.unit
def test_weights_are_ppr_scoring():
    """Per-target weights equal the PPR scoring dict (6/TD, 1/rec, 0.1/yd, -2/fum)."""
    w = mod._weights("RB")
    assert w["rushing_tds"] == 6.0
    assert w["receiving_tds"] == 6.0
    assert w["rushing_yards"] == pytest.approx(0.1)
    assert w["receiving_yards"] == pytest.approx(0.1)
    assert w["receptions"] == 1.0
    assert w["fumbles_lost"] == -2.0
    # WR has no rushing heads.
    assert set(mod._weights("WR")) == {
        "receiving_tds",
        "receiving_yards",
        "receptions",
        "fumbles_lost",
    }


@pytest.mark.unit
def test_signal_stats_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = mod._signal_stats(y, y)
    assert s["corr"] == pytest.approx(1.0)
    assert s["slope"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert s["recal_rmse"] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
def test_signal_stats_shrunk_is_recalibratable():
    """A perfectly-correlated but shrunk predictor has rmse>0 yet recal_rmse==0.

    This is the calibration-vs-information discriminator: shrinkage (high corr, wrong
    scale) is fully recoverable by a rescale, so recal_rmse collapses to 0.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = 0.5 * y  # perfectly correlated, half scale
    s = mod._signal_stats(y, pred)
    assert s["corr"] == pytest.approx(1.0)
    assert s["rmse"] > 0.1
    assert s["recal_rmse"] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
def test_signal_stats_constant_prediction_degrades_gracefully():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.full_like(y, 3.0)  # zero variance → corr undefined
    s = mod._signal_stats(y, pred)
    assert math.isnan(s["corr"])
    assert s["recal_rmse"] == pytest.approx(s["rmse"])  # no rescale can help


@pytest.mark.unit
def test_model_loss_labels_are_model_aware():
    """NN shows per-head families; LGBM/Ridge show their single objective."""
    fams = mod._loss_families("RB")
    nn = mod._model_loss_labels("RB", "nn", mod._targets("RB"), fams)
    assert nn["rushing_tds"] == "Poisson-NLL"
    assert nn["rushing_yards"] == "MSE"
    assert nn["receptions"] == "hurdle-NegBin"

    lgbm = mod._model_loss_labels("RB", "lgbm", mod._targets("RB"), fams)
    assert set(lgbm.values()) == {"L2 (regression)"}  # one objective for every head

    ridge = mod._model_loss_labels("RB", "ridge", mod._targets("RB"), fams)
    assert set(ridge.values()) == {"L2"}


@pytest.mark.unit
def test_decompose_is_exact():
    """Per-head MSE contributions sum to the total MSE and shares sum to 100%.

    Build a frame where ``pred_total`` / ``actual_fp`` are the weighted sums of the
    per-target columns (as the pipeline guarantees), so the additive identity must hold.
    """
    rng = np.random.default_rng(0)
    n = 40
    pos, model = "RB", "lgbm"
    targets = mod._targets(pos)
    weights = mod._weights(pos)

    data = {}
    for t in targets:
        data[t] = rng.uniform(0, 5, n)  # actual raw stat
        data[f"pred_{model}_{t}"] = rng.uniform(0, 5, n)  # predicted raw stat
    df = pd.DataFrame(data)
    df["actual_fp"] = sum(weights[t] * df[t] for t in targets)
    df[f"pred_{model}_total"] = sum(weights[t] * df[f"pred_{model}_{t}"] for t in targets)

    fams = mod._loss_families(pos)
    dec = mod._decompose(df, pos, model, targets, weights, fams)

    assert dec["contrib_mse"].sum() == pytest.approx(dec.attrs["mse"], rel=1e-6, abs=1e-6)
    assert dec["share_pct"].sum() == pytest.approx(100.0, abs=1e-6)
    assert set(dec["target"]) == set(targets)
