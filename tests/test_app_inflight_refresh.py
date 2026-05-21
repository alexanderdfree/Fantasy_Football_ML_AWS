"""Tests for the in-flight refresh branch in app._ensure_position_loaded.

When src.shared.model_sync's refresh poller swaps in a new model and touches
the sentinel, _ensure_position_loaded must detect the advance, drop cached
state for that position, and re-run _apply_position_models. When the sentinel
hasn't advanced (or is absent — dev/CI), behavior must match the pre-refresh
fast path.

These tests stub _apply_position_models and refresh_sentinel_mtime so we can
exercise the cache-invalidation logic without real model artifacts.
"""

from __future__ import annotations

import pytest

from src.serving import app


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Each test gets a fresh _cache and a no-op _ensure_base_data."""
    monkeypatch.setattr(app, "_cache", {})
    monkeypatch.setattr(app, "_ensure_base_data", lambda: None)
    # Pre-populate the minimum cache shape _ensure_position_loaded expects.
    app._cache["splits"] = {pos: (None, None, None) for pos in ("QB", "RB", "WR", "TE", "K", "DST")}
    app._cache["results"] = None  # _apply is stubbed so the value is irrelevant.
    app._cache["positions_loaded"] = set()
    yield


def _stub_apply(monkeypatch):
    """Replace _apply_position_models with a counter keyed by position."""
    counts: dict[str, int] = {}

    def fake_apply(train, val, test, pos, results):
        counts[pos] = counts.get(pos, 0) + 1

    monkeypatch.setattr(app, "_apply_position_models", fake_apply)
    return counts


def _stub_sentinel(monkeypatch, mtimes: dict[str, float]):
    """Replace refresh_sentinel_mtime with a lookup against ``mtimes``.
    Mutating ``mtimes`` between calls simulates the poller advancing the
    sentinel for one position."""
    monkeypatch.setattr(app, "refresh_sentinel_mtime", lambda pos: mtimes.get(pos, 0.0))


@pytest.mark.unit
def test_fast_path_skip_after_first_load(monkeypatch):
    """Once a position is loaded, repeated calls return without re-applying."""
    counts = _stub_apply(monkeypatch)
    _stub_sentinel(monkeypatch, {})

    for _ in range(5):
        app._ensure_position_loaded("QB")

    assert counts == {"QB": 1}


@pytest.mark.unit
def test_no_sentinel_keeps_pre_refresh_behavior(monkeypatch):
    """When the sentinel doesn't exist (refresh_sentinel_mtime returns 0.0
    for everything — dev / CI / brand-new container before any refresh), the
    function behaves exactly like the pre-refresh code: load once, then fast
    path forever."""
    counts = _stub_apply(monkeypatch)
    _stub_sentinel(monkeypatch, {})  # all sentinels return 0.0

    for _ in range(3):
        app._ensure_position_loaded("RB")
        app._ensure_position_loaded("WR")

    assert counts == {"RB": 1, "WR": 1}


@pytest.mark.unit
def test_sentinel_advance_triggers_reload(monkeypatch):
    """After the refresh poller touches the sentinel for a position, the next
    _ensure_position_loaded call must re-run _apply_position_models."""
    counts = _stub_apply(monkeypatch)
    mtimes = {"WR": 100.0}
    _stub_sentinel(monkeypatch, mtimes)

    app._ensure_position_loaded("WR")
    assert counts == {"WR": 1}

    # No advance → fast path, no re-load.
    app._ensure_position_loaded("WR")
    assert counts == {"WR": 1}

    # Simulate the poller having swapped in a new model.
    mtimes["WR"] = 200.0
    app._ensure_position_loaded("WR")
    assert counts == {"WR": 2}

    # Subsequent calls go back to the fast path against the new stored mtime.
    app._ensure_position_loaded("WR")
    assert counts == {"WR": 2}


@pytest.mark.unit
def test_sentinel_advance_only_reloads_changed_position(monkeypatch):
    """If only WR's sentinel advances, QB / RB / etc. must NOT re-load."""
    counts = _stub_apply(monkeypatch)
    mtimes = {"QB": 50.0, "WR": 50.0, "RB": 50.0}
    _stub_sentinel(monkeypatch, mtimes)

    app._ensure_position_loaded("QB")
    app._ensure_position_loaded("WR")
    app._ensure_position_loaded("RB")
    assert counts == {"QB": 1, "WR": 1, "RB": 1}

    mtimes["WR"] = 999.0
    app._ensure_position_loaded("QB")
    app._ensure_position_loaded("WR")
    app._ensure_position_loaded("RB")

    assert counts == {"QB": 1, "WR": 2, "RB": 1}


@pytest.mark.unit
def test_refresh_invalidates_metrics_by_format(monkeypatch):
    """metrics_by_format aggregates across positions, so any per-position
    refresh must invalidate it. _ensure_metrics will recompute on the next
    call."""
    _stub_apply(monkeypatch)
    mtimes = {"TE": 100.0}
    _stub_sentinel(monkeypatch, mtimes)

    app._ensure_position_loaded("TE")
    app._cache["metrics_by_format"] = {"ppr": "stale-cached-value"}

    mtimes["TE"] = 200.0
    app._ensure_position_loaded("TE")

    assert "metrics_by_format" not in app._cache


@pytest.mark.unit
def test_refresh_clears_failed_state_and_retries(monkeypatch):
    """A position that previously hard-failed should retry after the sentinel
    advances — the new model on disk might fix whatever broke the prior load.
    Without this, an in-flight refresh can't recover a degraded position."""
    calls: list[str] = []

    def fake_apply(train, val, test, pos, results):
        calls.append(pos)
        if len(calls) == 1:
            raise RuntimeError("first load fails")
        # Subsequent loads succeed.

    monkeypatch.setattr(app, "_apply_position_models", fake_apply)
    mtimes = {"DST": 100.0}
    _stub_sentinel(monkeypatch, mtimes)

    app._ensure_position_loaded("DST")
    assert "DST" in app._cache["positions_failed"]
    assert app._cache.get("position_load_errors", {}).get("DST") is not None

    # Without sentinel advance, the failed state persists.
    app._ensure_position_loaded("DST")
    assert calls == ["DST"]

    # Poller swaps in a new (hopefully-fixed) model.
    mtimes["DST"] = 200.0
    app._ensure_position_loaded("DST")

    assert "DST" not in app._cache["positions_failed"]
    assert "DST" in app._cache["positions_loaded"]
    assert calls == ["DST", "DST"]


@pytest.mark.unit
def test_first_load_with_existing_sentinel_does_not_log_refresh(monkeypatch, capsys):
    """If a sentinel happens to exist on the very first request (rare —
    ephemeral ECS filesystems don't carry it across tasks, but possible on
    re-runs), the first load should NOT emit the 'in-flight refresh detected'
    log line, because no in-memory state existed to invalidate."""
    _stub_apply(monkeypatch)
    _stub_sentinel(monkeypatch, {"K": 500.0})

    app._ensure_position_loaded("K")
    out = capsys.readouterr().out

    assert "in-flight refresh detected" not in out
    assert "Applying K-specific model" in out
