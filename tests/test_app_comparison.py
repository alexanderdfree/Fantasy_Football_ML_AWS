"""Tests for the Comparison tab: /api/comparison + the committed expert summary.

The route merges a LIVE model column (computed from ``_cache["results"]`` via the
same ``compute_metrics`` path as Model Performance, so it auto-updates on retrain)
with STATIC expert columns (NFL.com / RotoWire) read from the committed
``src/serving/comparison_experts.json``. Coverage:

  - ``_model_block_from_results`` helper: best-MAE arch selection, top-30 id
    filter, and the no-prediction / empty-slice → ``None`` paths (unit).
  - the route's merge, the model-unavailable fallback, and scoring passthrough
    (integration, via the Flask test client).
  - the committed JSON's data contract (six positions × two subsets, coverage
    holes nulled, metrics present) — guards the generator's output shape.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
_ARCHES = {"Ridge Regression", "Neural Network", "Attention NN", "LightGBM"}


# --------------------------------------------------------------------------- #
# _model_block_from_results — pure helper (no Flask boundary)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_block_picks_best_mae_arch(app_module):
    """Across the four architectures, the lowest-MAE one is reported."""
    rows = [
        {
            "player_id": "QB000",
            "position": "QB",
            "fantasy_points": 10.0,
            "ridge_pred_ppr": 10.0,  # perfect → MAE 0
            "nn_pred_ppr": 14.0,
            "attn_nn_pred_ppr": 15.0,
            "lgbm_pred_ppr": 16.0,
        }
        for _ in range(5)
    ]
    block = app_module._model_block_from_results(pd.DataFrame(rows), "ppr", "QB")
    assert block["best_arch"] == "Ridge Regression"
    assert block["mae"] == 0.0
    assert block["n"] == 5
    assert {"mae", "rmse", "r2", "n", "best_arch"} == set(block)


@pytest.mark.unit
def test_model_block_top30_filter_restricts_rows(app_module):
    rows = [
        {
            "player_id": pid,
            "position": "QB",
            "fantasy_points": 10.0,
            "ridge_pred_ppr": 11.0,
            "nn_pred_ppr": np.nan,
            "attn_nn_pred_ppr": np.nan,
            "lgbm_pred_ppr": np.nan,
        }
        for pid in ("QB000", "QB001", "QB999")
    ]
    block = app_module._model_block_from_results(
        pd.DataFrame(rows), "ppr", "QB", id_filter={"QB000", "QB001"}
    )
    assert block["n"] == 2  # QB999 excluded by the id filter
    assert block["best_arch"] == "Ridge Regression"


@pytest.mark.unit
def test_model_block_none_when_no_predictions(app_module):
    """K/DST have no ridge/lgbm in some configs; all-NaN preds → None."""
    df = pd.DataFrame(
        [
            {
                "player_id": "K000",
                "position": "K",
                "fantasy_points": 8.0,
                "ridge_pred_ppr": np.nan,
                "nn_pred_ppr": np.nan,
                "attn_nn_pred_ppr": np.nan,
                "lgbm_pred_ppr": np.nan,
            }
        ]
    )
    assert app_module._model_block_from_results(df, "ppr", "K") is None


@pytest.mark.unit
def test_model_block_none_when_position_absent(app_module):
    df = pd.DataFrame(
        [{"player_id": "QB000", "position": "QB", "fantasy_points": 10.0, "ridge_pred_ppr": 10.0}]
    )
    assert app_module._model_block_from_results(df, "ppr", "RB") is None


# --------------------------------------------------------------------------- #
# /api/comparison — route (Flask boundary)
# --------------------------------------------------------------------------- #


def _fake_experts():
    """Controlled expert payload whose top30_ids match the synthetic results
    (player_id ``{POS}000``..``{POS}003``), so the route's top-30 model slice is
    exercised deterministically without depending on the committed file."""

    def cell(mae):
        return {"mae": mae, "rmse": mae + 2.0, "r2": 0.3, "n": 100}

    def subset(mult):
        out = {}
        for p in _POSITIONS:
            out[p] = {
                "nflcom": None if p == "DST" else cell(round(5.0 * mult, 3)),
                "rotowire": None if p == "K" else cell(round(5.5 * mult, 3)),
            }
        return out

    return {
        "generated_at": "2026-05-31T00:00:00+00:00",
        "scoring": "ppr",
        "top_n": 30,
        "experts_meta": {"model": {}, "nflcom": {"note": "n"}, "rotowire": {"note": "r"}},
        "top30_ids": {p: [f"{p}000", f"{p}001"] for p in _POSITIONS},
        "subsets": {"all": subset(1.0), "top30": subset(1.1)},
    }


@pytest.mark.integration
def test_comparison_merges_live_model_with_static_experts(app_module, synthetic_cache, monkeypatch):
    monkeypatch.setattr(app_module, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    assert body["model_source"] == "live"
    assert set(body["subsets"]) == {"all", "top30"}

    qb = body["subsets"]["all"]["QB"]
    assert qb["model"] is not None
    assert qb["model"]["best_arch"] in _ARCHES
    assert {"mae", "rmse", "r2", "n"} <= set(qb["model"])
    # Static experts passed through verbatim from the (faked) committed JSON.
    assert qb["nflcom"] == {"mae": 5.0, "rmse": 7.0, "r2": 0.3, "n": 100}
    assert qb["rotowire"] == {"mae": 5.5, "rmse": 7.5, "r2": 0.3, "n": 100}

    # Coverage holes survive the merge.
    assert body["subsets"]["all"]["DST"]["nflcom"] is None
    assert body["subsets"]["all"]["K"]["rotowire"] is None

    # Top-30 model column is computed on the id-filtered slice (2 players × 7 wk).
    top_qb = body["subsets"]["top30"]["QB"]
    assert top_qb["model"] is not None
    assert top_qb["model"]["n"] == 14
    assert top_qb["nflcom"]["mae"] == 5.5  # 5.0 * 1.1

    assert body["generated_at"] and "experts_meta" in body


@pytest.mark.integration
def test_comparison_scoring_param_passthrough(app_module, synthetic_cache, monkeypatch):
    monkeypatch.setattr(app_module, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        assert c.get("/api/comparison?scoring=half_ppr").get_json()["scoring"] == "half_ppr"
        # Unknown scoring silently falls back to ppr (matches other endpoints).
        assert c.get("/api/comparison?scoring=bogus").get_json()["scoring"] == "ppr"


@pytest.mark.integration
def test_comparison_model_unavailable_when_no_results(app_module, monkeypatch):
    """No loaded models (empty cache): experts still render; model cells are None
    and ``model_source`` reports the degraded state — the tab never 500s."""
    monkeypatch.setattr(app_module, "_load_comparison_experts", _fake_experts)
    # Don't let _ensure_metrics try to load real models off disk/S3.
    monkeypatch.setattr(app_module, "_ensure_metrics", lambda: None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    assert body["model_source"] == "unavailable"
    qb = body["subsets"]["all"]["QB"]
    assert qb["model"] is None
    assert qb["nflcom"] is not None  # experts unaffected


@pytest.mark.integration
def test_comparison_500_when_expert_data_missing(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_load_comparison_experts", lambda: None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        r = c.get("/api/comparison")
    assert r.status_code == 500


# --------------------------------------------------------------------------- #
# Committed comparison_experts.json — data contract
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_committed_expert_summary_contract():
    """The generator's committed output has the shape the route + UI rely on."""
    import src.serving.app as app_mod

    with open(app_mod._COMPARISON_EXPERTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert set(data["subsets"]) == {"all", "top30"}
    for subset in ("all", "top30"):
        assert set(data["subsets"][subset]) == set(_POSITIONS)
        for pos in _POSITIONS:
            assert set(data["subsets"][subset][pos]) == {"nflcom", "rotowire"}
        # Coverage holes: NFL.com has no DST; RotoWire has no K.
        assert data["subsets"][subset]["DST"]["nflcom"] is None
        assert data["subsets"][subset]["K"]["rotowire"] is None
        # A present cell carries all three metrics + a sample count.
        qb_nfl = data["subsets"][subset]["QB"]["nflcom"]
        assert {"mae", "rmse", "r2", "n"} <= set(qb_nfl)

    assert set(data["top30_ids"]) == set(_POSITIONS)
    for pos in _POSITIONS:
        ids = data["top30_ids"][pos]
        assert 0 < len(ids) <= data["top_n"]
    assert "experts_meta" in data and "generated_at" in data
