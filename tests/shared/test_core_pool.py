"""Tests for src.shared.core_pool — the work-conserving CPU core pool.

Three layers: the pure ``CoreAllocator`` grant/fairness policy (no I/O), the
no-op-without-env contract (so non-orchestrated callers stay byte-identical), and a real
``AF_UNIX`` coordinator round-trip including crash-reclaim and blocking-until-release.
All in-process and fast — no subprocess, no training.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest

from src.shared.core_pool import (
    ENV_ADDR,
    ENV_POS,
    CoreAllocator,
    lease_cores,
    start_coordinator,
)

pytestmark = pytest.mark.unit

_BIG = 1 << 30  # "want as many as fair allows"


# --------------------------------------------------------------- allocator policy


def test_fair_cap_formula():
    assert CoreAllocator.fair_cap(16, 6) == 3  # ceil(16/6)
    assert CoreAllocator.fair_cap(16, 3) == 6
    assert CoreAllocator.fair_cap(16, 2) == 8
    assert CoreAllocator.fair_cap(16, 1) == 16
    assert CoreAllocator.fair_cap(16, 0) == 16  # guarded: max(1, active)
    assert CoreAllocator.fair_cap(16, 16) == 1
    assert CoreAllocator.fair_cap(16, 99) == 1  # never below 1


def test_grant_respects_fair_cap():
    a = CoreAllocator(list(range(16)))
    a.active_count = 6
    g = a.grant(key=1, want=_BIG, min_cores=1)
    assert len(g) == 3  # ceil(16/6)
    assert set(g).isdisjoint(a.free)  # granted cores left the free set
    assert len(a.free) == 13


def test_grant_gives_all_cores_to_sole_active_position():
    a = CoreAllocator(list(range(16)))
    a.active_count = 1
    g = a.grant(key=1, want=_BIG, min_cores=1)
    assert len(g) == 16  # work-conserving: the last position alive uses everything
    assert a.free == set()


def test_grant_shrinks_to_what_is_free():
    a = CoreAllocator(list(range(16)))
    a.active_count = 1  # cap would allow 16
    a.free = {0, 1}  # but only 2 are free
    assert sorted(a.grant(key=1, want=_BIG, min_cores=1)) == [0, 1]


def test_grant_blocks_below_min_cores():
    a = CoreAllocator(list(range(16)))
    a.active_count = 1
    a.free = {0}
    assert a.grant(key=1, want=_BIG, min_cores=3) is None  # 1 free < 3 requested floor


def test_grant_min_cores_floor_beats_cap():
    a = CoreAllocator(list(range(16)))
    a.active_count = 6  # cap 3
    g = a.grant(key=1, want=_BIG, min_cores=5)
    assert len(g) == 5  # an explicit floor overrides the fair-share cap


def test_release_restores_free_and_is_idempotent():
    a = CoreAllocator(list(range(16)))
    a.active_count = 4
    g = a.grant(key=7, want=_BIG, min_cores=1)
    assert g and 7 in a.held
    assert sorted(a.release(7)) == sorted(g)
    assert a.free == set(range(16))
    assert a.release(7) == []  # second release is a no-op


def test_cap_rises_as_active_count_drops():
    """The headline behaviour: a finished position lets survivors' next lease widen."""
    a = CoreAllocator(list(range(16)))
    a.active_count = 6
    g1 = a.grant(key=1, want=_BIG, min_cores=1)
    assert len(g1) == 3  # ceil(16/6) while 6 contend
    a.active_count = 2  # four positions finished -> orchestrator lowered the count
    g2 = a.grant(key=2, want=_BIG, min_cores=1)
    assert len(g2) == 8  # ceil(16/2), bounded by the 13 still free
    assert set(g1).isdisjoint(g2)  # concurrent grants never overlap


# --------------------------------------------------------------- no-op contract


def test_lease_cores_is_a_noop_without_env(monkeypatch):
    monkeypatch.delenv(ENV_ADDR, raising=False)
    # The unset path must never open a socket nor touch affinity.
    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: pytest.fail("no-op lease must not open a socket")
    )
    aff_calls: list = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda *a: aff_calls.append(a), raising=False)
    with lease_cores("ridge_cv") as n:
        assert n == -1  # joblib "use all cores" status quo
    with lease_cores("lgbm", default=None) as n:
        assert n is None  # LightGBM falls back to reading LGBM_N_JOBS
    assert aff_calls == []  # affinity untouched on the no-op path


# --------------------------------------------------------- coordinator round-trip


def _acquire_raw(addr: str, want: int, min_cores: int, timeout: float = 5.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(addr)
    s.settimeout(timeout)
    s.sendall((json.dumps({"op": "acquire", "want": want, "min_cores": min_cores}) + "\n").encode())
    reply = s.recv(4096)
    return reply, s  # caller owns closing s (closing == release)


def test_coordinator_grant_release_roundtrip(tmp_path, monkeypatch):
    addr, set_active, stop = start_coordinator([0, 1, 2, 3], str(tmp_path))
    try:
        monkeypatch.setenv(ENV_ADDR, addr)
        monkeypatch.setenv(ENV_POS, "QB")
        set_active(2)  # cap = ceil(4/2) = 2
        with lease_cores("ridge_cv") as n:
            assert 1 <= n <= 2
        # the lease released on exit, so with one position left a lease takes all four
        set_active(1)
        with lease_cores("lgbm", default=None) as n:
            assert n == 4
    finally:
        stop()


def test_coordinator_reclaims_cores_on_abrupt_disconnect(tmp_path):
    """A worker that dies mid-lease (socket closes without releasing) must not leak cores."""
    addr, set_active, stop = start_coordinator([0, 1], str(tmp_path))
    try:
        set_active(1)
        reply, s = _acquire_raw(addr, want=2, min_cores=1)
        assert b'"cores"' in reply
        s.close()  # abrupt close == death; the handler must reclaim on EOF
        # A fresh client requesting both cores blocks server-side until reclaim, then grants.
        reply2, s2 = _acquire_raw(addr, want=2, min_cores=2, timeout=5.0)
        s2.close()
        assert b'"cores"' in reply2
    finally:
        stop()


def test_coordinator_blocks_until_release(tmp_path):
    addr, set_active, stop = start_coordinator([0], str(tmp_path))  # a single core
    try:
        set_active(1)
        held_reply, holder = _acquire_raw(addr, want=1, min_cores=1)
        assert b'"cores"' in held_reply  # holder owns the only core

        proceeded = threading.Event()

        def _waiter():
            _, w = _acquire_raw(addr, want=1, min_cores=1, timeout=5.0)
            proceeded.set()
            w.close()

        t = threading.Thread(target=_waiter, daemon=True)
        t.start()
        assert not proceeded.wait(0.5)  # blocked while the holder keeps the core
        holder.sendall(b'{"op": "release"}\n')
        holder.close()
        assert proceeded.wait(5.0)  # released -> the waiter is granted the core
    finally:
        stop()
