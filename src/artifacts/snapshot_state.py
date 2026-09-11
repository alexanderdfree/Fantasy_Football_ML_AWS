"""HTTP-independent build state and complete request snapshot generations."""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import uuid4

_scoped_state = ContextVar("serving_state", default=None)
_request_snapshot = ContextVar("serving_request_snapshot", default=None)


# Optional HTTP context adapters are installed only by src.serving.state.
def _context_state():
    return None


def _has_request_context():
    return False


def _context_inference():
    return None


def allows_runtime_inference():
    scoped = _scoped_state.get()
    if scoped is not None:
        return scoped.allow_runtime_inference
    configured = _context_inference()
    return DEFAULT_STATE.allow_runtime_inference if configured is None else configured


@dataclass(frozen=True)
class ServingSnapshot:
    generation: str
    cache: MappingProxyType


class SnapshotRepository:
    """Publish one detached generation; readers keep a stable reference."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = None

    def current(self) -> ServingSnapshot | None:
        with self._lock:
            return self._snapshot

    def discard(self, generation=None):
        with self._lock:
            if self._snapshot is not None and (
                generation is None or self._snapshot.cache.get("snapshot_generation") == generation
            ):
                self._snapshot = None

    def publish(self, cache: dict) -> ServingSnapshot:
        # Raw training splits and kick source frames are builder inputs, not
        # request data. Copy the completed projections/metrics together once.
        excluded = {"splits", "k_kicks_df"}
        values = copy.deepcopy(
            {k: v for k, v in cache.items() if k not in excluded and not k.startswith("wiki:")}
        )
        if "results" not in values or "metrics" not in values:
            raise ValueError("Serving snapshot requires completed results and metrics")
        generation = values.get("snapshot_generation") or uuid4().hex
        snapshot = ServingSnapshot(generation, MappingProxyType(values))
        with self._lock:
            self._snapshot = snapshot
        return snapshot


@dataclass
class ServingState:
    cache: dict = field(default_factory=dict)
    cache_lock: object = field(default_factory=threading.RLock)
    results_write_lock: object = field(default_factory=threading.Lock)
    wiki_cache_lock: object = field(default_factory=threading.Lock)
    wiki_cache: dict = field(default_factory=dict)
    snapshots: SnapshotRepository = field(default_factory=SnapshotRepository)
    publish_remote: bool = True
    allow_runtime_inference: bool = True

    def publish(self):
        with self.cache_lock:
            snapshot = self.snapshots.publish(self.cache)
        if _has_request_context():
            _request_snapshot.set(snapshot)
        return snapshot


DEFAULT_STATE = ServingState()


def current_state() -> ServingState:
    scoped = _scoped_state.get()
    if scoped is not None:
        return scoped
    return _context_state() or DEFAULT_STATE


def current_snapshot() -> ServingSnapshot | None:
    return _request_snapshot.get()


@contextmanager
def use_state(state: ServingState):
    token = _scoped_state.set(state)
    snapshot_token = _request_snapshot.set(None)
    try:
        yield state
    finally:
        _request_snapshot.reset(snapshot_token)
        _scoped_state.reset(token)


def __getattr__(name):
    attributes = {
        "_cache": "cache",
        "_cache_lock": "cache_lock",
        "_results_write_lock": "results_write_lock",
        "_wiki_cache_lock": "wiki_cache_lock",
        "_wiki_cache": "wiki_cache",
    }
    if name not in attributes:
        raise AttributeError(name)
    if name == "_cache" and current_snapshot() is not None:
        return current_snapshot().cache
    return getattr(current_state(), attributes[name])
