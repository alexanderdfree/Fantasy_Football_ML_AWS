"""Unit tests for the rolling-origin driver in src/benchmarking/benchmark.py.

The per-origin training (``_score_origin``) and fold construction
(``_rolling_origin_inputs``) are monkeypatched, so these cover the driver's
iteration, mean±std aggregation, production-origin selection, and schema-safety
without training a model or touching data splits.
"""

from __future__ import annotations

import json
import statistics

import pytest

import src.benchmarking.benchmark as b

pytestmark = pytest.mark.unit


def _summary(pos, ridge_mae, nn_mae, *, with_lgbm=False):
    s = {
        "position": pos,
        "ridge_mae": ridge_mae,
        "ridge_r2": 0.40,
        "ridge_top12": 0.55,
        "nn_mae": nn_mae,
        "nn_r2": 0.42,
        "nn_top12": 0.58,
    }
    if with_lgbm:
        s.update({"lgbm_mae": nn_mae + 0.1, "lgbm_r2": 0.41, "lgbm_top12": 0.56})
    return s


def test_module_exposes_rolling_origin_api():
    for fn in (
        "run_rolling_origin",
        "_aggregate_rolling_origin",
        "_score_origin",
        "_rolling_origin_inputs",
    ):
        assert hasattr(b, fn)


def test_aggregate_rolling_origin_mean_std_math():
    per_origin = [
        (2023, _summary("RB", 4.9, 4.8)),
        (2024, _summary("RB", 4.7, 4.6)),
        (2025, _summary("RB", 4.5, 4.4)),
    ]
    ro = b._aggregate_rolling_origin(per_origin)
    assert ro["n_origins"] == 3
    assert ro["test_seasons"] == [2023, 2024, 2025]
    assert [o["test_season"] for o in ro["per_origin"]] == [2023, 2024, 2025]
    assert ro["aggregate"]["ridge"]["mae_mean"] == round(statistics.mean([4.9, 4.7, 4.5]), 4)
    assert ro["aggregate"]["ridge"]["mae_std"] == round(statistics.stdev([4.9, 4.7, 4.5]), 4)
    assert ro["aggregate"]["nn"]["mae_mean"] == round(statistics.mean([4.8, 4.6, 4.4]), 4)


def test_aggregate_excludes_model_not_in_every_origin():
    """LightGBM present in only one origin must not appear in the aggregate (avoids
    averaging over a non-uniform model set)."""
    per_origin = [
        (2024, _summary("RB", 4.7, 4.6, with_lgbm=True)),
        (2025, _summary("RB", 4.5, 4.4, with_lgbm=False)),
    ]
    ro = b._aggregate_rolling_origin(per_origin)
    assert "lgbm" not in ro["aggregate"]
    assert "ridge" in ro["aggregate"] and "nn" in ro["aggregate"]
    # per_origin still records the lgbm number where it exists.
    assert "lgbm_mae" in ro["per_origin"][0] and "lgbm_mae" not in ro["per_origin"][1]


def test_aggregate_single_origin_std_is_zero():
    ro = b._aggregate_rolling_origin([(2025, _summary("K", 3.0, 2.9))])
    assert ro["n_origins"] == 1
    assert ro["aggregate"]["ridge"]["mae_std"] == 0.0


def test_run_rolling_origin_assembles_block_and_keeps_flat_keys(monkeypatch):
    # te placeholder carries the season int so the mocked scorer returns per-origin data.
    origins = [(ts, None, None, ts) for ts in (2023, 2024, 2025)]
    canned = {
        2023: _summary("RB", 4.9, 4.8),
        2024: _summary("RB", 4.8, 4.7),
        2025: _summary("RB", 4.7, 4.6),  # production origin
    }
    monkeypatch.setattr(b, "_rolling_origin_inputs", lambda pos: (origins, None))
    monkeypatch.setattr(b, "_score_origin", lambda pos, tr, va, te, cfg: dict(canned[te]))

    s = b.run_rolling_origin("RB")
    # Flat keys equal the production origin (test==TEST_SEASONS[0]==2025), so the
    # row stays comparable to a normal single-split run in the History tab.
    assert s["ridge_mae"] == 4.7
    assert s["nn_mae"] == 4.6
    ro = s["rolling_origin"]
    assert ro["n_origins"] == 3
    assert ro["test_seasons"] == [2023, 2024, 2025]
    assert ro["aggregate"]["ridge"]["mae_mean"] == round(statistics.mean([4.9, 4.8, 4.7]), 4)
    # Whole summary (flat keys + nested rolling_origin block) is JSON-serializable.
    json.dumps(s)
