"""Coverage tests for ``src/evaluation/metrics.py``.

Exercises ``compute_metrics`` (happy path + single-sample R² guard),
``compute_positional_metrics`` (mixed positions + filtered-empty case),
and ``print_comparison_table``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    compute_positional_metrics,
    print_comparison_table,
)


@pytest.mark.unit
def test_compute_metrics_happy_path():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = compute_metrics(y_true, y_pred)
    assert math.isclose(m["mae"], 0.15, abs_tol=1e-9)
    assert m["rmse"] > 0
    # Why: predictions deviate from truth by ≤0.2 on a [1, 4] range — variance
    # of residuals is ~0.025, variance of truth is 1.25 → R² = 1 - 0.025/1.25
    # = 0.98. The 0.9 floor is a wide tolerance band that catches a sign
    # flip or normalization regression without wedging on float noise.
    assert m["r2"] > 0.9


@pytest.mark.unit
def test_compute_metrics_single_sample_returns_nan_r2():
    """When size < 2, r2_score would raise; we short-circuit to NaN."""
    m = compute_metrics(np.array([5.0]), np.array([4.5]))
    assert math.isclose(m["mae"], 0.5)
    assert math.isnan(m["r2"])


@pytest.mark.unit
def test_compute_positional_metrics_skips_empty_positions():
    """Positions with 0 samples in df get filtered out of the result."""
    df = pd.DataFrame(
        {
            "position": ["QB", "QB", "QB", "RB", "RB"],
            "pred": [20.0, 22.0, 18.0, 12.0, 15.0],
            "actual": [21.0, 19.0, 17.0, 11.0, 14.0],
        }
    )
    out = compute_positional_metrics(df, "pred", "actual")
    positions = set(out["position"].unique())
    # Only QB + RB have data; WR/TE/K/DST are missing.
    assert positions == {"QB", "RB"}
    assert "n_samples" in out.columns
    # QB has 3 rows, RB has 2.
    qb_row = out[out["position"] == "QB"].iloc[0]
    rb_row = out[out["position"] == "RB"].iloc[0]
    assert qb_row["n_samples"] == 3
    assert rb_row["n_samples"] == 2


@pytest.mark.unit
def test_compute_positional_metrics_all_six_positions():
    """With every position represented, the result has exactly 6 rows."""
    rows = []
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        for i in range(3):
            rows.append({"position": p, "pred": float(i), "actual": float(i) + 0.1})
    df = pd.DataFrame(rows)
    out = compute_positional_metrics(df, "pred", "actual")
    assert len(out) == 6


@pytest.mark.unit
def test_print_comparison_table(capsys):
    results = {
        "Ridge": {"mae": 5.5, "rmse": 7.1, "r2": 0.32},
        "NN": {"mae": 5.1, "rmse": 6.9, "r2": 0.37},
    }
    print_comparison_table(results)
    out = capsys.readouterr().out
    assert "Ridge" in out
    assert "NN" in out
    assert "MAE" in out
    assert "R2" in out
