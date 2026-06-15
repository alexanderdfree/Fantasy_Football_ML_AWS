"""Tests for the Comparison tab: /api/comparison + the committed expert summary.

The route merges LIVE per-model columns (one block per architecture — ridge / nn /
attn_nn / lgbm — computed from ``_cache["results"]`` via the same ``compute_metrics``
path as Model Performance, so they auto-update on retrain) with STATIC expert columns
(NFL.com / RotoWire) read from the committed ``src/serving/comparison_experts.json``.
Coverage:

  - ``_model_blocks_from_results`` / ``_model_reliabilities_from_results`` helpers:
    per-model metric + residual-σ dicts, top-30 id filter, and the no-prediction /
    empty-slice → all-``None`` paths (unit).
  - the route's merge, the model-unavailable fallback, and scoring passthrough
    (integration, via the Flask test client).
  - the committed JSON's data contract (six positions × three subsets, coverage
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
_MODEL_KEYS = {"ridge", "nn", "attn_nn", "lgbm"}


# --------------------------------------------------------------------------- #
# _model_blocks_from_results — pure helper (no Flask boundary)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_blocks_returns_per_model_metrics(app_module):
    """Every architecture gets its own {mae,rmse,r2,n} block, computed independently."""
    rows = [
        {
            "player_id": "QB000",
            "position": "QB",
            "fantasy_points": 10.0,
            "ridge_pred_ppr": 10.0,  # |10-10| → MAE 0
            "nn_pred_ppr": 14.0,  # |14-10| → MAE 4
            "attn_nn_pred_ppr": 15.0,  # |15-10| → MAE 5
            "lgbm_pred_ppr": 16.0,  # |16-10| → MAE 6
        }
        for _ in range(5)
    ]
    blocks = comparison._model_blocks_from_results(pd.DataFrame(rows), "ppr", "QB")
    assert set(blocks) == _MODEL_KEYS
    assert {"mae", "rmse", "r2", "n"} == set(blocks["ridge"])
    assert blocks["ridge"]["mae"] == 0.0
    assert blocks["ridge"]["n"] == 5
    assert blocks["nn"]["mae"] == 4.0
    assert blocks["attn_nn"]["mae"] == 5.0
    assert blocks["lgbm"]["mae"] == 6.0


@pytest.mark.unit
def test_model_blocks_top30_filter_restricts_rows(app_module):
    """id_filter restricts the rows; models with no prediction in the slice are None."""
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
    blocks = comparison._model_blocks_from_results(
        pd.DataFrame(rows), "ppr", "QB", id_filter={"QB000", "QB001"}
    )
    assert blocks["ridge"]["n"] == 2  # QB999 excluded by the id filter
    assert blocks["nn"] is None and blocks["lgbm"] is None and blocks["attn_nn"] is None


@pytest.mark.unit
def test_model_blocks_all_none_when_no_predictions(app_module):
    """All-NaN preds → every model None, but the dict still carries all four keys."""
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
    blocks = comparison._model_blocks_from_results(df, "ppr", "K")
    assert set(blocks) == _MODEL_KEYS
    assert all(v is None for v in blocks.values())


@pytest.mark.unit
def test_model_blocks_all_none_when_position_absent(app_module):
    df = pd.DataFrame(
        [{"player_id": "QB000", "position": "QB", "fantasy_points": 10.0, "ridge_pred_ppr": 10.0}]
    )
    blocks = comparison._model_blocks_from_results(df, "ppr", "RB")
    assert set(blocks) == _MODEL_KEYS
    assert all(v is None for v in blocks.values())


# --------------------------------------------------------------------------- #
# _model_reliabilities_from_results — live per-model residual σ (2025 test rows)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_reliabilities_sigma_bias_per_model(app_module):
    """σ + bias come from each model's residuals (pred − actual), independently."""
    fp = [10.0, 12.0, 14.0, 16.0, 18.0]
    ridge = [11.0, 11.0, 14.0, 17.0, 17.0]  # resid +1,-1,0,+1,-1 → bias 0, sample σ 1
    rows = [
        {
            "player_id": f"QB00{i}",
            "position": "QB",
            "fantasy_points": fp[i],
            "ridge_pred_ppr": ridge[i],
            "nn_pred_ppr": fp[i] + 2.0,  # constant +2 resid → bias +2, σ 0
            "attn_nn_pred_ppr": np.nan,
            "lgbm_pred_ppr": np.nan,
        }
        for i in range(5)
    ]
    rel = comparison._model_reliabilities_from_results(pd.DataFrame(rows), "ppr", "QB")
    assert set(rel) == _MODEL_KEYS
    assert {"n", "mae", "bias", "sigma"} == set(rel["ridge"])
    assert rel["ridge"]["n"] == 5
    assert rel["ridge"]["bias"] == 0.0
    assert rel["ridge"]["sigma"] == 1.0  # sample std (ddof=1) of [+1,-1,0,+1,-1]
    assert rel["nn"]["bias"] == 2.0
    assert rel["nn"]["sigma"] == 0.0
    # Models with no predictions for the slice are None.
    assert rel["attn_nn"] is None and rel["lgbm"] is None


@pytest.mark.unit
def test_model_reliabilities_all_none_when_no_predictions(app_module):
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
    rel = comparison._model_reliabilities_from_results(df, "ppr", "K")
    assert set(rel) == _MODEL_KEYS
    assert all(v is None for v in rel.values())


# --------------------------------------------------------------------------- #
# /api/comparison — route (Flask boundary)
# --------------------------------------------------------------------------- #


def _fake_experts():
    """Controlled expert payload whose top12_ids / top30_ids match the synthetic
    results (player_id ``{POS}000``..``{POS}003``), so the route's ranked model
    slices are exercised deterministically without depending on the committed file."""

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
        "top12_n": 12,
        "experts_meta": {"model": {}, "nflcom": {"note": "n"}, "rotowire": {"note": "r"}},
        "top12_ids": {p: [f"{p}000", f"{p}001"] for p in _POSITIONS},
        "top30_ids": {p: [f"{p}000", f"{p}001"] for p in _POSITIONS},
        "subsets": {"all": subset(1.0), "top12": subset(1.05), "top30": subset(1.1)},
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
    assert set(body["subsets"]) == {"all", "top12", "top30"}

    qb = body["subsets"]["all"]["QB"]
    # Each architecture is its own block now (no single "Our Model" / best_arch).
    assert "model" not in qb
    assert set(qb) >= _MODEL_KEYS
    for key in _MODEL_KEYS:  # QB has all four models in the synthetic cache
        assert qb[key] is not None, key
        assert {"mae", "rmse", "r2", "n"} <= set(qb[key])
    # Static experts passed through verbatim from the (faked) committed JSON.
    assert qb["nflcom"] == {"mae": 5.0, "rmse": 7.0, "r2": 0.3, "n": 100}
    assert qb["rotowire"] == {"mae": 5.5, "rmse": 7.5, "r2": 0.3, "n": 100}

    # attn_nn / lgbm aren't in the synthetic K/DST rows → those cells null out.
    assert body["subsets"]["all"]["K"]["attn_nn"] is None
    assert body["subsets"]["all"]["K"]["lgbm"] is None
    assert body["subsets"]["all"]["K"]["ridge"] is not None

    # Coverage holes survive the merge.
    assert body["subsets"]["all"]["DST"]["nflcom"] is None
    assert body["subsets"]["all"]["K"]["rotowire"] is None

    # Top-30 model columns are computed on the id-filtered slice (2 players × 7 wk).
    top_qb = body["subsets"]["top30"]["QB"]
    assert top_qb["ridge"]["n"] == 14
    assert top_qb["nflcom"]["mae"] == 5.5  # 5.0 * 1.1

    # Top-12 uses the same id filter here, so the model slice matches; experts carry
    # the top12 subset's own (faked) numbers (5.0 * 1.05).
    top12_qb = body["subsets"]["top12"]["QB"]
    assert top12_qb["ridge"]["n"] == 14
    assert top12_qb["nflcom"]["mae"] == 5.25  # 5.0 * 1.05

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
    qb = mr["QB"]  # per-model dict now, not a single best-arch block
    assert qb is not None
    assert set(qb) >= _MODEL_KEYS
    for key in _MODEL_KEYS:  # QB has all four models in the synthetic cache
        assert qb[key] is not None, key
        assert {"n", "mae", "bias", "sigma"} == set(qb[key])
        assert qb[key]["sigma"] >= 0.0
    # K lacks attn_nn / lgbm predictions in the synthetic cache → null per model.
    assert mr["K"]["attn_nn"] is None and mr["K"]["lgbm"] is None
    assert mr["K"]["ridge"] is not None


@pytest.mark.integration
def test_comparison_scoring_pinned_to_ppr(app_module, synthetic_cache, monkeypatch):
    # The committed expert columns are baked PPR-only, so /api/comparison ignores
    # any ?scoring= and always scores the model block at — and echoes — ppr
    # (audit #653). The frontend only ever requests it with no scoring param.
    monkeypatch.setattr(comparison, "_load_comparison_experts", _fake_experts)
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        assert c.get("/api/comparison").get_json()["scoring"] == "ppr"
        # A non-ppr request arg is ignored — the response stays ppr so the model
        # block and the PPR-only experts compare apples-to-apples.
        assert c.get("/api/comparison?scoring=half_ppr").get_json()["scoring"] == "ppr"
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
    # No live models → the per-model columns are absent; experts still render.
    assert all(qb.get(key) is None for key in _MODEL_KEYS)
    assert qb["nflcom"] is not None  # experts unaffected
    # The live model reliability columns are also unavailable, per position.
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

    assert set(data["subsets"]) == {"all", "top12", "top30"}
    for subset in ("all", "top12", "top30"):
        assert set(data["subsets"][subset]) == set(_POSITIONS)
        for pos in _POSITIONS:
            assert set(data["subsets"][subset][pos]) == {"nflcom", "rotowire"}
        # Coverage holes: NFL.com has no DST; RotoWire has no K.
        assert data["subsets"][subset]["DST"]["nflcom"] is None
        assert data["subsets"][subset]["K"]["rotowire"] is None
        # A present cell carries all three metrics + a sample count.
        qb_nfl = data["subsets"][subset]["QB"]["nflcom"]
        assert {"mae", "rmse", "r2", "n"} <= set(qb_nfl)

    # Each ranked tier emits a per-position id set, capped by its own cutoff.
    for ids_key, cap in (("top30_ids", data["top_n"]), ("top12_ids", data["top12_n"])):
        assert set(data[ids_key]) == set(_POSITIONS)
        for pos in _POSITIONS:
            ids = data[ids_key][pos]
            assert 0 < len(ids) <= cap
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
