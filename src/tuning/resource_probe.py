"""Lightweight peak-compute probe for tune / stacked-ensemble Batch runs.

Answers "what did this run actually use?" (peak RSS, container cgroup peak,
CPU-seconds → effective cores, GPU memory) with stdlib + torch only — no
psutil in the training image. A daemon thread samples the cgroup memory
counter (the number the OOM killer acts on, covering child processes too);
everything else is collected at ``stop()``.

Usage::

    probe = ResourceProbe().start()
    ...work...
    report["resources"] = probe.stop()

The probe is fail-open by design: any sampling problem degrades to ``None``
fields rather than breaking the run it measures.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

_CGROUP_V2 = "/sys/fs/cgroup/memory.current"
_CGROUP_V1 = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_GIB = 1024**3


def _read_cgroup_bytes() -> int | None:
    for path in (_CGROUP_V2, _CGROUP_V1):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _maxrss_gb(who: int) -> float:
    peak = resource.getrusage(who).ru_maxrss
    # ru_maxrss is bytes on macOS, kibibytes on Linux.
    return peak / (_GIB if sys.platform == "darwin" else 1024**2)


class ResourceProbe:
    """``start()`` → work → ``stop() -> dict`` of peaks for the probed window."""

    def __init__(self, interval_sec: float = 0.5):
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cgroup_peak: int | None = None
        self._t0_wall = 0.0
        self._t0_cpu = (0.0, 0.0)

    def start(self) -> ResourceProbe:
        self._t0_wall = time.monotonic()
        t = os.times()
        self._t0_cpu = (t.user + t.system, t.children_user + t.children_system)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 — the probe must never break the run
            pass
        if _read_cgroup_bytes() is not None:
            self._thread = threading.Thread(target=self._sample, name="resource-probe", daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop_event.wait(self._interval):
            cur = _read_cgroup_bytes()
            if cur is not None and (self._cgroup_peak is None or cur > self._cgroup_peak):
                self._cgroup_peak = cur

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        wall = time.monotonic() - self._t0_wall
        t = os.times()
        cpu_self = t.user + t.system - self._t0_cpu[0]
        # Children CPU accumulates at wait(); call stop() after joining workers.
        cpu_children = t.children_user + t.children_system - self._t0_cpu[1]
        out = {
            "wall_sec": round(wall, 1),
            "cpu_sec_self": round(cpu_self, 1),
            "cpu_sec_children": round(cpu_children, 1),
            "cpu_util_cores": round((cpu_self + cpu_children) / max(wall, 1e-9), 2),
            "peak_rss_self_gb": round(_maxrss_gb(resource.RUSAGE_SELF), 3),
            "peak_rss_children_gb": round(_maxrss_gb(resource.RUSAGE_CHILDREN), 3),
            "cgroup_peak_gb": (
                round(self._cgroup_peak / _GIB, 3) if self._cgroup_peak is not None else None
            ),
        }
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / _GIB, 3)
                out["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / _GIB, 3)
        except Exception:  # noqa: BLE001
            pass
        return out
