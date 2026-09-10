"""Smoke tests for src/tuning/resource_probe.py (stdlib peak-compute probe)."""

from __future__ import annotations

import builtins
import importlib.util
import json
import time
from types import SimpleNamespace

import pytest

from src.tuning import resource_probe
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
    if resource_probe.resource is None:
        assert out["peak_rss_self_gb"] is None
    else:
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


def test_probe_imports_and_runs_without_unix_resource(monkeypatch):
    original_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ModuleNotFoundError("No module named 'resource'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    spec = importlib.util.spec_from_file_location("probe_without_resource", resource_probe.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.ResourceProbe().start().stop()
    assert out["peak_rss_self_gb"] is None
    assert out["peak_rss_children_gb"] is None
    assert out["cpu_sec_self"] >= 0.0
    json.dumps(out)


def test_unavailable_rss_does_not_abort_probe(monkeypatch):
    def unavailable(who):
        raise OSError("RSS unavailable")

    monkeypatch.setattr(
        resource_probe,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, RUSAGE_CHILDREN=-1, getrusage=unavailable),
    )
    out = ResourceProbe().start().stop()
    assert out["peak_rss_self_gb"] is None
    assert out["peak_rss_children_gb"] is None
