"""Work-conserving CPU core pool for the parallel local trainer.

``src/benchmarking/parallel_train.py`` fans the six positions out as one process each
on a many-core CUDA box. The hazard is CPU oversubscription during each position's
*CPU-heavy stages* — the Ridge/ElasticNet alpha-CV (``joblib.Parallel(prefer="threads")``)
and LightGBM (``LGBM_N_JOBS`` threads). The original orchestrator handed each worker a
static slice of physical cores at launch, baking ``LGBM_N_JOBS`` / ``LOKY_MAX_CPU_COUNT``
into env that is **immutable post-spawn**. So when a fast position finished and freed its
cores, the survivors' affinity could be widened but their thread *counts* could not — the
freed cores went unused (verified: ``joblib.cpu_count()`` honours ``LOKY_MAX_CPU_COUNT``,
so the frozen cap throttled both joblib and LightGBM). Re-pinning was cosmetic.

This module replaces the static slices with a **runtime token pool**. A coordinator thread
inside the orchestrator owns the box's physical cores. Each worker, on entering a CPU stage,
``lease_cores(...)`` acquires up to its fair share, pins affinity to the granted core ids,
sets that stage's thread count to the grant, runs the stage, and releases on exit. The
fair-share cap is ``ceil(total / active_positions)`` — so as positions *finish*, the
orchestrator lowers the active count and survivors' *next* acquisitions widen (3 cores each
when 6 run, 8 when 2 remain, all 16 for the last). Cores follow the work, and the thread
count — not just affinity — moves with them. The cap is keyed on the orchestrator's count of
running positions (not the instantaneous set of CPU contenders) so a position that reaches
its CPU stage first, while its peers are still loading data, cannot grab every core.

The transport is an ``AF_UNIX`` stream socket. The held tokens are tied to the *connection*,
not a named primitive, so a worker that dies mid-stage (LightGBM/OpenMP has segfaulted on
this codebase before) closes its socket → the handler hits EOF → its cores are reclaimed and
waiters woken. A leaked semaphore or lockfile could not offer that.

NO-OP CONTRACT (mandatory): if ``FF_CORE_POOL_ADDR`` is unset, ``lease_cores`` is a complete
no-op — no socket, no affinity change — that yields the caller's status-quo ``n_jobs``. So
``python -m src.{pos}.run_pipeline``, AWS Batch, and pytest are byte-identical to today; only
a run launched by the parallel orchestrator (which sets the env) ever touches the pool. The
change is numerically inert: ``n_jobs`` controls thread count only, never the math.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import socketserver
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

# Env handles set by the orchestrator on each worker (see parallel_train._launch).
ENV_ADDR = "FF_CORE_POOL_ADDR"  # AF_UNIX socket path; unset => lease_cores is a no-op
ENV_POS = "FF_CORE_POOL_POS"  # this worker's position label (diagnostics only)

_RECV_CHUNK = 4096


# --------------------------------------------------------------------------- allocator


class CoreAllocator:
    """Pure physical-core token allocator (no I/O). Caller holds the external lock.

    Owns the free-core set and a per-holder ledger. The fairness denominator is
    ``active_count`` — the number of running positions, set by the orchestrator — so the
    cap reserves a share for positions that are running but have not yet reached their CPU
    stage (preventing an early arriver from grabbing every core). As positions finish, the
    orchestrator lowers ``active_count`` and the cap rises for the survivors.
    """

    def __init__(self, all_cores: list[int]):
        self.all: list[int] = list(all_cores)
        self.total: int = len(self.all)
        self.free: set[int] = set(all_cores)
        self.held: dict[int, list[int]] = {}
        # Conservative default until the orchestrator sets the real count; the orchestrator
        # sets it before any worker can acquire, so this only guards a pre-launch query.
        self.active_count: int = max(1, self.total)

    @staticmethod
    def fair_cap(total: int, active_count: int) -> int:
        """Per-position core cap: ``ceil(total / active)``, at least 1."""
        return max(1, math.ceil(total / max(1, active_count)))

    def grant(self, key: int, want: int, min_cores: int) -> list[int] | None:
        """Grant ``min(want, fair_cap, free)`` cores, or ``None`` if it must block.

        Blocks (returns ``None``) only when fewer than ``min_cores`` are free; otherwise
        always grants ≥ ``min_cores`` so a head-of-line waiter makes progress.
        """
        free_n = len(self.free)
        if free_n < min_cores:
            return None
        cap = self.fair_cap(self.total, self.active_count)
        target = min(want, cap, free_n)
        if target < min_cores:  # honour the floor past the cap (free_n >= min_cores here)
            target = min_cores
        granted = sorted(self.free)[:target]
        self.free.difference_update(granted)
        self.held[key] = granted
        return list(granted)

    def release(self, key: int) -> list[int]:
        """Return ``key``'s held cores to the free set (idempotent)."""
        ids = self.held.pop(key, [])
        self.free.update(ids)
        return ids


# ----------------------------------------------------------------------- coordinator


class _Coordinator:
    """Shared state behind the socket: the allocator, a lock, and a FIFO waiter queue."""

    def __init__(self, all_cores: list[int]):
        self.alloc = CoreAllocator(all_cores)
        self.cond = threading.Condition()
        self.waiters: deque[int] = deque()  # FIFO of blocked acquirer keys (fairness)

    def set_active_count(self, n: int) -> None:
        """Orchestrator hook: update the fairness denominator and wake waiters."""
        with self.cond:
            self.alloc.active_count = max(1, int(n))
            self.cond.notify_all()  # caps loosened (or tightened) — re-evaluate waiters


class _Handler(socketserver.StreamRequestHandler):
    """One thread per connected worker. The connection lifetime == the lease lifetime."""

    def handle(self) -> None:
        coord: _Coordinator = self.server.coord  # type: ignore[attr-defined]
        key = id(self)  # unique among live handlers
        try:
            for raw in self.rfile:  # iterate lines until EOF (release or worker death)
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                op = msg.get("op")
                if op == "acquire":
                    granted = self._acquire(
                        coord,
                        key,
                        int(msg.get("want", 1 << 30)),
                        max(1, int(msg.get("min_cores", 1))),
                    )
                    self.wfile.write((json.dumps({"cores": granted}) + "\n").encode())
                    self.wfile.flush()
                elif op == "release":
                    with coord.cond:
                        coord.alloc.release(key)
                        coord.cond.notify_all()
        except (OSError, ConnectionError):
            pass
        finally:
            # Authoritative reclaim: a clean release already emptied the ledger; a crash or
            # abrupt close did not, so return whatever this connection still holds.
            with coord.cond:
                if key in coord.waiters:
                    coord.waiters.remove(key)
                coord.alloc.release(key)
                coord.cond.notify_all()

    def _acquire(self, coord: _Coordinator, key: int, want: int, min_cores: int) -> list[int]:
        """Block (FIFO, head-of-line) until a grant of ≥ ``min_cores`` is available."""
        with coord.cond:
            coord.waiters.append(key)
            try:
                while True:
                    if coord.waiters[0] == key:
                        granted = coord.alloc.grant(key, want, min_cores)
                        if granted is not None:
                            coord.waiters.popleft()
                            coord.cond.notify_all()  # next head may now be grantable too
                            return granted
                    coord.cond.wait()
            except BaseException:
                # Don't leave a dead/cancelled acquirer at the head of the queue.
                if key in coord.waiters:
                    coord.waiters.remove(key)
                    coord.cond.notify_all()
                raise


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def start_coordinator(all_cores: list[int], socket_dir: str):
    """Bind an ``AF_UNIX`` socket and serve the pool on a daemon thread.

    Returns ``(addr, set_active_count, stop)``: the socket path to hand workers via
    ``FF_CORE_POOL_ADDR``, a callable the orchestrator calls with its running-position count,
    and a shutdown callable (idempotent) that stops the thread and unlinks the socket.
    """
    addr = os.path.join(socket_dir, "core_pool.sock")
    if len(addr.encode()) >= 108:  # AF_UNIX sun_path limit
        raise ValueError(f"AF_UNIX socket path too long ({len(addr.encode())} >= 108): {addr}")
    with contextlib.suppress(FileNotFoundError):
        os.unlink(addr)
    server = _ThreadingUnixServer(addr, _Handler)
    server.coord = _Coordinator(all_cores)  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, name="core-pool", daemon=True)
    thread.start()

    def stop() -> None:
        with contextlib.suppress(Exception):
            server.shutdown()
        with contextlib.suppress(Exception):
            server.server_close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(addr)

    return addr, server.coord.set_active_count, stop  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------- client


def _send(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode())


def _recv(sock: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(_RECV_CHUNK)
        if not chunk:
            raise ConnectionError("core-pool coordinator closed the connection")
        buf += chunk
    return json.loads(buf)


@contextmanager
def lease_cores(
    stage: str, want: int | None = None, min_cores: int = 1, default: int | None = -1
) -> Iterator[int | None]:
    """Lease CPU cores for one CPU-heavy stage; yield the granted count (use as ``n_jobs``).

    While the context is open, this process's affinity is pinned to the granted core ids; on
    exit (including exceptions) the cores are released. When the pool is active the yielded
    value is the granted core count (a positive int). When it is **not** active
    (``FF_CORE_POOL_ADDR`` unset, or the coordinator is unreachable), this is a no-op that
    yields ``default`` — the caller's status-quo (``-1`` for joblib "use all", ``None`` for
    "let LightGBM read ``LGBM_N_JOBS``") — so non-orchestrated callers are byte-identical.

    ``want=None`` requests the full fair share; pass an int to self-limit. The pool is strictly
    an optimisation: any socket error degrades to ``default`` rather than failing the run.
    """
    addr = os.environ.get(ENV_ADDR)
    if not addr:
        yield default
        return

    want_wire = (1 << 30) if want is None else int(want)
    sock = None
    granted: list[int] = []
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(addr)
        _send(
            sock,
            {
                "op": "acquire",
                "stage": stage,
                "want": want_wire,
                "min_cores": int(min_cores),
                "pos": os.environ.get(ENV_POS, "?"),
            },
        )
        granted = _recv(sock).get("cores") or []
    except OSError:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        yield default  # pool unreachable — degrade to status quo
        return

    if not granted:  # coordinator only blocks until >=1 free, so this is defensive
        with contextlib.suppress(OSError):
            sock.close()
        yield default
        return

    prev_aff: set[int] | None = None
    with contextlib.suppress(AttributeError, OSError):
        prev_aff = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(granted))
    try:
        yield len(granted)
    finally:
        if prev_aff is not None:
            with contextlib.suppress(AttributeError, OSError):
                os.sched_setaffinity(0, prev_aff)
        with contextlib.suppress(OSError):
            _send(sock, {"op": "release"})
        with contextlib.suppress(OSError):
            sock.close()
