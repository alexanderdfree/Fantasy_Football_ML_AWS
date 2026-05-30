"""Smoke + unit tests for src/analysis/analysis_rb_lgbm_disagreement.py.

The script's ``main()`` runs the full RB pipeline (slow, data-dependent) and is
out of scope for unit tests. These cover the pure helpers against a tiny
synthetic frame so a future change to the metric/gap logic is caught by CI, and
assert that importing the module does not fire the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_rb_lgbm_disagreement as ad

pytestmark = pytest.mark.unit


def _synthetic() -> pd.DataFrame:
    # Two rows; LGBM is closest to actual on both, Ridge highest (the pattern
    # the script exists to characterise).
    return pd.DataFrame(
        {
            "fantasy_points": [4.5, 8.5],
            "week": [2, 5],
            "pred_ridge_total": [10.0, 20.0],
            "pred_nn_total": [8.0, 16.0],
            "pred_attn_nn_total": [6.0, 12.0],
            "pred_lgbm_total": [4.0, 8.0],
            "pred_ridge_rushing_yards": [100.0, 100.0],
            "pred_lgbm_rushing_yards": [50.0, 50.0],
            "pred_ridge_rushing_tds": [1.0, 1.0],
            "pred_lgbm_rushing_tds": [0.5, 0.5],
        }
    )


def test_module_imports_without_running_pipeline():
    """Importing must not pull the pipeline (``run`` is deferred into main())."""
    for fn in ("main", "per_model_metrics", "peer_gap", "calibration_table", "gap_decomposition"):
        assert hasattr(ad, fn)
    assert ad.LGBM == "LightGBM"
    assert ad.ACTUAL == "fantasy_points"
    assert set(ad.PEERS) == {"Ridge", "NN", "Attention NN"}
    assert ad.MODELS[ad.LGBM] == "pred_lgbm_total"
    # ``run`` is imported lazily inside main(), so it must NOT be bound at module
    # scope — importing this module therefore cannot pull the pipeline/torch.
    assert not hasattr(ad, "run")


def test_per_model_metrics_mae_and_bias():
    m = ad.per_model_metrics(_synthetic())
    # LGBM residuals: 4-4.5=-0.5, 8-8.5=-0.5 -> MAE 0.5, bias -0.5.
    assert m["LightGBM"]["mae"] == pytest.approx(0.5)
    assert m["LightGBM"]["bias"] == pytest.approx(-0.5)
    assert m["LightGBM"]["n"] == 2
    # Ridge over-predicts: 10-4.5=5.5, 20-8.5=11.5 -> MAE 8.5, bias +8.5.
    assert m["Ridge"]["mae"] == pytest.approx(8.5)
    assert m["Ridge"]["bias"] == pytest.approx(8.5)
    # LGBM is the most accurate of the four (the whole point).
    assert min(m, key=lambda k: m[k]["mae"]) == "LightGBM"


def test_per_model_metrics_handles_empty_slice():
    """A data-dependent slice (gap<=-4, actual>=20) can be empty on another
    season; metrics must come back NaN with n=0, not raise from sklearn."""
    m = ad.per_model_metrics(_synthetic().iloc[0:0])
    assert m["LightGBM"]["n"] == 0
    assert np.isnan(m["LightGBM"]["mae"])
    assert np.isnan(m["Ridge"]["bias"])


def test_peer_gap_is_peer_mean_minus_lgbm():
    gap = ad.peer_gap(_synthetic())
    # row0: mean(10,8,6)=8 - 4 = 4 ; row1: mean(20,16,12)=16 - 8 = 8.
    assert list(gap) == [4.0, 8.0]


def test_gap_decomposition_weights_each_target():
    decomp = ad.gap_decomposition(_synthetic(), scoring={"rushing_yards": 0.1, "rushing_tds": 6.0})
    # (100-50)*0.1 = 5.0 ; (1-0.5)*6 = 3.0.
    assert decomp["rushing_yards"] == pytest.approx(5.0)
    assert decomp["rushing_tds"] == pytest.approx(3.0)


def test_gap_decomposition_skips_missing_targets():
    # A scoring key with no pred_*_{target} columns is silently skipped.
    decomp = ad.gap_decomposition(_synthetic(), scoring={"receiving_tds": 6.0})
    assert decomp == {}


def test_calibration_table_shape():
    calib = ad.calibration_table(_synthetic(), bins=[0, 5, 10, np.inf])
    # actual 4.5 -> [0,5) ; actual 8.5 -> [5,10). Two non-empty bins.
    assert len(calib) == 2
    assert {"bin", "n", "avg_actual", "Ridge", "LightGBM"} <= set(calib.columns)
    assert calib["n"].tolist() == [1, 1]
