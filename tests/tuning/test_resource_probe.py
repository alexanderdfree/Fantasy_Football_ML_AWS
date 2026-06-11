"""Smoke tests for src/tuning/resource_probe.py (stdlib peak-compute probe)."""

from __future__ import annotations

import json
import time

import pytest

from src.tuning.resource_probe import ResourceProbe

pytestmark = pytest.mark.unit


def test_probe_reports_usage_fields_and_is_serializable():
    probe = ResourceProbe(interval_sec=0.05).start()
    t0 = time.monotonic()
    x = 0
    while time.monotonic() - t0 < 0.15:  # burn a little CPU inside the window
        x += 1
    out = probe.stop()
    assert out["wall_sec"] >= 0.1
    assert out["cpu_sec_self"] >= 0.0
    assert out["peak_rss_self_gb"] > 0.0
    for field in ("cpu_sec_children", "cpu_util_cores", "peak_rss_children_gb", "cgroup_peak_gb"):
        assert field in out
    json.dumps(out)  # the report embeds into results.json — must serialize
    assert x > 0


def test_probe_is_fail_open_off_cgroup():
    # On macOS/CI there is no cgroup file: the sampler thread never starts and
    # cgroup_peak_gb degrades to None instead of raising.
    out = ResourceProbe(interval_sec=0.05).start().stop()
    assert "cgroup_peak_gb" in out
