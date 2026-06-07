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

import src.serving.comparison as comparison
import src.serving.core as core

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
    block = comparison._model_block_from_results(pd.DataFrame(rows), "ppr", "QB")
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
    block = comparison._model_block_from_results(
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
    assert comparison._model_block_from_results(df, "ppr", "K") is None


@pytest.mark.unit
def test_model_block_none_when_position_absent(app_module):
    df = pd.DataFrame(
        [{"player_id": "QB000", "position": "QB", "fantasy_points": 10.0, "ridge_pred_ppr": 10.0}]
    )
    assert comparison._model_block_from_results(df, "ppr", "RB") is None


# --------------------------------------------------------------------------- #
# _model_reliability_from_results — live model residual σ (2025 test rows)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_reliability_sigma_bias(app_module):
    """σ + bias come from the best-MAE arch's residuals (pred − actual)."""
    fp = [10.0, 12.0, 14.0, 16.0, 18.0]
    ridge = [11.0, 11.0, 14.0, 17.0, 17.0]  # resid +1,-1,0,+1,-1 → bias 0, sample σ 1
    rows = [
        {
            "player_id": f"QB00{i}",
            "position": "QB",
            "fantasy_points": fp[i],
            "ridge_pred_ppr": ridge[i],
            "nn_pred_ppr": fp[i] + 5.0,  # all worse → ridge wins on MAE
            "attn_nn_pred_ppr": fp[i] + 6.0,
            "lgbm_pred_ppr": fp[i] + 7.0,
        }
        for i in range(5)
    ]
    block = comparison._model_reliability_from_results(pd.DataFrame(rows), "ppr", "QB")
    assert block["best_arch"] == "Ridge Regression"
    assert block["n"] == 5
    assert block["bias"] == 0.0
    assert block["sigma"] == 1.0  # sample std (ddof=1) of [+1,-1,0,+1,-1]
    assert {"n", "mae", "bias", "sigma", "best_arch"} == set(block)


@pytest.mark.unit
def test_model_reliability_none_when_no_predictions(app_module):
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
    assert comparison._model_reliability_from_results(df, "ppr", "K") is None


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

    def rel_cell(sigma):
        return {
            "n": 100,
            "mae": 5.0,
            "rmse": 7.0,
            "r2": 0.3,
            "bias": 0.4,
            "sigma": sigma,
            # Frontend renders the 2025 slice; pooled σ rides along for the hover.
            "per_season": {
                "2025": {
                    "n": 40,
                    "mae": 5.2,
                    "rmse": 7.1,
                    "r2": 0.28,
                    "bias": 0.5,
                    "sigma": sigma + 0.3,
                }
            },
            "seasons": [2024, 2025],
        }

    return {
        "generated_at": "2026-05-31T00:00:00+00:00",
        "scoring": "ppr",
        "top_n": 30,
        "experts_meta": {"model": {}, "nflcom": {"note": "n"}, "rotowire": {"note": "r"}},
        "top30_ids": {p: [f"{p}000", f"{p}001"] for p in _POSITIONS},
        "subsets": {"all": subset(1.0), "top30": subset(1.1)},
        "expert_reliability": {
            "seasons": [2024, 2025],
            "scoring": "ppr",
            "residual_convention": "projection_minus_actual",
            "note": "Residual σ ... provenance ...",
            "positions": {
                p: {
                    "nflcom": None if p == "DST" else rel_cell(6.9),
                    "rotowire": None if p == "K" else rel_cell(7.4),
                }
                for p in _POSITIONS
            },
        },
    }


@pytest.mark.integration
def test_comparison_merges_live_model_with_static_experts(app_module, synthetic_cache, monkeypatch):
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
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
def test_comparison_passes_expert_reliability_through(app_module, synthetic_cache, monkeypatch):
    """The per-source residual-σ block is forwarded verbatim from the committed JSON,
    including the position-coverage gaps (NFL.com no-DST, RotoWire no-K)."""
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    rel = body["expert_reliability"]
    assert rel["seasons"] == [2024, 2025]
    assert rel["positions"]["QB"]["nflcom"]["sigma"] == 6.9
    assert rel["positions"]["QB"]["rotowire"]["sigma"] == 7.4
    assert rel["positions"]["DST"]["nflcom"] is None
    assert rel["positions"]["K"]["rotowire"] is None


@pytest.mark.integration
def test_comparison_includes_live_model_reliability(app_module, synthetic_cache, monkeypatch):
    """The model side of the reliability table is computed live per position from the
    cached test predictions (auto-updates on retrain), alongside the static experts."""
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    mr = body["model_reliability"]
    assert set(mr) == set(_POSITIONS)
    qb = mr["QB"]
    assert qb is not None
    assert {"n", "mae", "bias", "sigma", "best_arch"} == set(qb)
    assert qb["best_arch"] in _ARCHES
    assert qb["sigma"] >= 0.0


@pytest.mark.integration
def test_comparison_scoring_param_passthrough(app_module, synthetic_cache, monkeypatch):
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
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
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    # Don't let _ensure_metrics try to load real models off disk/S3.
    monkeypatch.setattr(core, "_ensure_metrics", lambda: None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    assert body["model_source"] == "unavailable"
    qb = body["subsets"]["all"]["QB"]
    assert qb["model"] is None
    assert qb["nflcom"] is not None  # experts unaffected
    # The live model reliability column is also unavailable, per position.
    assert body["model_reliability"]["QB"] is None


@pytest.mark.integration
def test_comparison_500_when_expert_data_missing(app_module, monkeypatch):
    monkeypatch.setattr(comparison, "_load_comparison_experts", lambda: None)
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

    with open(comparison._COMPARISON_EXPERTS_PATH, encoding="utf-8") as f:
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

    # Per-source residual-σ block (expert uncertainty): multi-season, expert-only,
    # same coverage holes as the head-to-head (NFL.com no DST, RotoWire no K).
    rel = data["expert_reliability"]
    assert rel["seasons"] and rel["residual_convention"] == "projection_minus_actual"
    assert set(rel["positions"]) == set(_POSITIONS)
    assert rel["positions"]["DST"]["nflcom"] is None
    assert rel["positions"]["K"]["rotowire"] is None
    qb_nfl_rel = rel["positions"]["QB"]["nflcom"]
    assert {"n", "mae", "rmse", "bias", "sigma"} <= set(qb_nfl_rel)
    # The reliability table renders the 2025 slice per source — lock that it ships.
    assert "2025" in qb_nfl_rel["per_season"]
    assert {"n", "bias", "sigma"} <= set(qb_nfl_rel["per_season"]["2025"])


# --------------------------------------------------------------------------- #
# Prediction intervals — _load_expert_intervals + the /api/comparison block +
# the committed expert_intervals.json calibration contract
# --------------------------------------------------------------------------- #

_INTERVAL_SOURCES = ("nflcom", "rotowire")


@pytest.mark.unit
def test_load_expert_intervals_missing_file_returns_none(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "_EXPERT_INTERVALS_PATH", str(tmp_path / "nope.json"))
    assert comparison._load_expert_intervals() is None


@pytest.mark.integration
def test_comparison_includes_intervals_block(app_module, synthetic_cache, monkeypatch):
    """The intervals ride along on the /api/comparison payload (one fetch)."""
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    assert "intervals" in body
    iv = body["intervals"]
    assert iv is not None and set(iv["intervals"]) == set(_INTERVAL_SOURCES)


@pytest.mark.integration
def test_comparison_intervals_optional(app_module, synthetic_cache, monkeypatch):
    """A missing intervals file degrades to intervals=None; accuracy tables unaffected."""
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    monkeypatch.setattr(comparison, "_load_expert_intervals", lambda: None)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        body = c.get("/api/comparison").get_json()

    assert body["intervals"] is None
    assert body["subsets"]["all"]["QB"]["nflcom"] is not None


@pytest.mark.unit
def test_committed_expert_intervals_contract():
    """The fitter's committed output has the shape the route + UI rely on, AND the
    nominal-80% bands empirically cover ≈80% of held-out actuals (the key check —
    pinned here so a regeneration that miscalibrates fails CI)."""

    with open(comparison._EXPERT_INTERVALS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert data["nominal_coverage"] == 0.8
    assert data["tau"] == [0.1, 0.5, 0.9]
    assert set(data["intervals"]) == set(_INTERVAL_SOURCES)
    # Coverage holes: NFL.com has no DST; RotoWire has no K.
    assert data["intervals"]["nflcom"]["DST"] is None
    assert data["intervals"]["rotowire"]["K"] is None

    for source in _INTERVAL_SOURCES:
        for pos, block in data["intervals"][source].items():
            if block is None or block.get("skipped"):
                continue
            assert set(block["params"]) == {"floor", "median", "ceiling"}
            cov = block["calibration"]["coverage"]
            assert 0.6 <= cov <= 0.95, f"{source}/{pos} coverage {cov} off-nominal"
            assert 0 < len(block["examples"]) <= 6
            for e in block["examples"]:
                assert e["floor"] <= e["median"] <= e["ceiling"]
                assert e["in_band"] == (e["floor"] <= e["actual"] <= e["ceiling"])

    # The NFL.com look-ahead leak (the hvpkod archive's pre-2024 offense 'projected'
    # files are backfilled with realized box scores) was detected + excluded; RotoWire
    # is clean. NFL.com offense therefore fits on the single genuine 2024 season.
    assert {2021, 2022, 2023} <= set(data["sources_meta"]["nflcom"]["look_ahead_seasons"])
    assert data["sources_meta"]["rotowire"]["look_ahead_seasons"] == []
    assert data["intervals"]["nflcom"]["RB"]["fit_seasons"] == [2024]
