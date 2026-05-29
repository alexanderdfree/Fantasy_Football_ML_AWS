"""Unit tests for the ``/warm`` cache-warm endpoint (D14 backend pre-warm).

``/warm`` runs the same ``_ensure_metrics()`` the first
``/api/predictions?position=ALL`` call would, so a CI / operator probe pays the
30-60s cold compute (or a disk/S3 hydrate) instead of a real visitor — and the
recompute re-uploads the S3 prediction cache so later containers hydrate. See
``src/serving/app.py::warm`` and ``docs/ARCHITECTURE.md`` D14.

Both collaborators are monkeypatched (``_ensure_metrics`` and
``_compute_models_fingerprint``) so these stay fast unit tests with no
dependency on a real ``src/{pos}/outputs/models/`` tree. The ``app_module``
fixture (tests/conftest.py) supplies a clean ``_cache`` plus a
``tmp_path``-redirected ``_PREDICTIONS_CACHE_DIR`` per test.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _install_fake_ensure(app_mod, monkeypatch, calls):
    """Replace ``_ensure_metrics`` with a stub that records each call and seeds
    the minimal ``_cache`` state ``/warm`` reads back (mirrors the post-success
    invariant: ``results`` + ``positions_loaded`` + ``metrics_by_format`` set)."""

    def _fake():
        calls.append(1)
        app_mod._cache.setdefault("metrics_by_format", {"ppr": {}})
        app_mod._cache.setdefault("positions_loaded", {"QB", "RB", "WR", "TE", "K", "DST"})
        app_mod._cache.setdefault("results", pd.DataFrame({"position": ["QB", "RB"]}))

    monkeypatch.setattr(app_mod, "_ensure_metrics", _fake)


def test_warm_returns_status_fingerprint_and_rows(app_module, monkeypatch):
    calls = []
    _install_fake_ensure(app_module, monkeypatch, calls)
    monkeypatch.setattr(
        app_module, "_compute_models_fingerprint", lambda: ("abc123def4567890ff", [])
    )
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        resp = client.get("/warm")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["fingerprint"] == "abc123def456"  # first 12 hex chars only
    assert body["rows"] == 2
    assert set(body["positions_loaded"]) == {"QB", "RB", "WR", "TE", "K", "DST"}
    assert body["degraded_positions"] == []
    assert isinstance(body["elapsed_s"], (int, float))
    assert calls == [1]  # _ensure_metrics invoked exactly once per probe


def test_warm_surfaces_degraded_positions(app_module, monkeypatch):
    calls = []
    _install_fake_ensure(app_module, monkeypatch, calls)
    monkeypatch.setattr(app_module, "_compute_models_fingerprint", lambda: ("d" * 64, []))
    # A per-model load error must surface so CI logs (and the frontend banner)
    # can see a degraded warm rather than a falsely-clean one.
    app_module._cache["position_load_errors"] = {"DST_ridge": "missing artifact"}
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        body = client.get("/warm").get_json()

    assert body["degraded_positions"] == ["DST"]


def test_warm_idempotent_returns_stable_fingerprint(app_module, monkeypatch):
    calls = []
    _install_fake_ensure(app_module, monkeypatch, calls)
    monkeypatch.setattr(app_module, "_compute_models_fingerprint", lambda: ("f" * 64, []))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        first = client.get("/warm").get_json()
        second = client.get("/warm").get_json()

    assert first["fingerprint"] == second["fingerprint"]
    assert len(calls) == 2  # the route delegates to _ensure_metrics on every probe


def test_warm_propagates_all_positions_failed(app_module, monkeypatch):
    """When every position fails, ``_ensure_metrics`` raises; ``/warm`` must
    surface it (like ``/api/predictions``), not swallow it into a false 200."""

    def _boom():
        raise RuntimeError("All positions failed to load")

    monkeypatch.setattr(app_module, "_ensure_metrics", _boom)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        with pytest.raises(RuntimeError, match="All positions failed"):
            client.get("/warm")
