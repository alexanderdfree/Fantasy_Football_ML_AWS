"""Tests for the per-stat breakdown drill-down feature.

Covers the ``/api/predictions/breakdown`` JSON contract (QB full models; K/DST
with lgbm unavailable but attn_nn present), the error paths (404 / 400), graceful
degradation on a stale snapshot, and the per-target columns surviving the parquet
persist -> hydrate round-trip. The ``_apply_position_models`` write path is
asserted in ``test_app_apply_position_models.py`` where the loader-stub fixtures
live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
def test_breakdown_qb_contract(client_with_data):
    """QB row: every model present, components match POSITION_INFO order."""
    import src.serving.app as app

    resp = client_with_data.get("/api/predictions/breakdown?player_id=QB000&week=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["position"] == "QB"
    assert body["unavailable"] is False
    assert body["unavailable_models"] == []  # QB has all four models

    keys = [c["key"] for c in body["components"]]
    expected = [t["key"] for t in app.POSITION_INFO["QB"]["targets"]]
    assert keys == expected  # order preserved

    # Each component carries label, unit, actual, and all four model values.
    first = body["components"][0]
    assert first["label"]
    assert "unit" in first
    for prefix in ("actual", "ridge", "nn", "attn_nn", "lgbm"):
        assert prefix in first
        assert first[prefix] is not None


@pytest.mark.integration
@pytest.mark.parametrize("pos", ["K", "DST"])
def test_breakdown_lgbm_unavailable_for_k_dst(client_with_data, pos):
    """K/DST have no LightGBM → lgbm in unavailable_models; attn_nn stays real."""
    resp = client_with_data.get(f"/api/predictions/breakdown?player_id={pos}000&week=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["position"] == pos
    assert "lgbm" in body["unavailable_models"]
    assert "attn_nn" not in body["unavailable_models"]
    # Every lgbm cell is null; attn_nn cells are populated.
    for c in body["components"]:
        assert c["lgbm"] is None
        assert c["attn_nn"] is not None


@pytest.mark.integration
def test_breakdown_dst_has_all_ten_targets(client_with_data):
    """DST exposes its full 10-stat set incl. points_allowed / yards_allowed."""
    resp = client_with_data.get("/api/predictions/breakdown?player_id=DST000&week=1")
    body = resp.get_json()
    keys = {c["key"] for c in body["components"]}
    assert "points_allowed" in keys
    assert "yards_allowed" in keys
    assert len(body["components"]) == 10


@pytest.mark.integration
def test_breakdown_404_unknown_player(client_with_data):
    resp = client_with_data.get("/api/predictions/breakdown?player_id=NOPE&week=1")
    assert resp.status_code == 404


@pytest.mark.integration
def test_breakdown_400_bad_week(client_with_data):
    resp = client_with_data.get("/api/predictions/breakdown?player_id=QB000&week=notaweek")
    assert resp.status_code == 400


@pytest.mark.integration
def test_breakdown_400_missing_player_id(client_with_data):
    resp = client_with_data.get("/api/predictions/breakdown?week=1")
    assert resp.status_code == 400


@pytest.mark.integration
def test_breakdown_does_not_collide_with_player_route(client_with_data):
    """The static /breakdown segment must win over /api/predictions/<player_id>.

    A regression here would route ``breakdown`` into ``api_player`` (treating
    "breakdown" as a player_id → 404) instead of the breakdown handler.
    """
    resp = client_with_data.get("/api/predictions/breakdown?player_id=QB000&week=1")
    assert resp.status_code == 200
    assert "components" in resp.get_json()


@pytest.mark.integration
def test_breakdown_degrades_on_stale_snapshot(client_with_data):
    """When the per-target columns are absent (old cache), report degraded."""
    import src.serving.app as app

    results = app._cache["results"]
    drop = [c for c in results.columns if c.startswith(("pred_", "actual_"))]
    app._cache["results"] = results.drop(columns=drop)

    resp = client_with_data.get("/api/predictions/breakdown?player_id=QB000&week=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["unavailable"] is True
    assert body["components"] == []


@pytest.mark.integration
def test_per_target_columns_survive_parquet_roundtrip(tmp_path):
    """The breakdown columns persist through to_parquet -> read_parquet, so the
    feature works on the disk-hydrate boot path (not just fresh inference)."""
    from tests.conftest import _synthetic_results

    results = _synthetic_results()
    assert any(c.startswith("pred_ridge_") for c in results.columns)
    path = tmp_path / "predictions.parquet"
    results.to_parquet(path, index=True)
    back = pd.read_parquet(path)

    target_cols = [c for c in results.columns if c.startswith(("pred_", "actual_"))]
    assert target_cols  # sanity: there are some
    for c in target_cols:
        assert c in back.columns
    # A QB row's passing_yards actual is preserved (not silently dropped/NaN'd).
    qb = back[back["position"] == "QB"].iloc[0]
    assert not np.isnan(qb["actual_passing_yards"])
