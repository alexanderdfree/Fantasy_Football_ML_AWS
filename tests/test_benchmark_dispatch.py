"""Unit tests for ``benchmark.main``'s concurrency dispatch.

``benchmark.py`` is the autodetect front door: on a many-core CUDA box it delegates to
``parallel_train.orchestrate`` (parallel fan-out), elsewhere it runs the in-process
sequential loop. These tests pin which path each flag/host combination takes without
spawning a subprocess or training a model — ``orchestrate`` is replaced by a recorder and
``detect_platform`` is faked, and every call passes ``--dry-run`` so a mis-routed
sequential fallback returns at the dry-run guard instead of training.
"""

from __future__ import annotations

import pytest

from src.benchmarking import benchmark

pytestmark = pytest.mark.unit

ALL_SIX = ["QB", "RB", "WR", "TE", "K", "DST"]


@pytest.fixture
def delegate(monkeypatch):
    """Replace the real orchestrator with a recorder; returns the list of captured calls."""
    calls: list[dict] = []

    def _fake(positions, jobs, passthrough, note, no_sync, dry_run, rolling_origin=False):
        calls.append(
            {
                "positions": positions,
                "jobs": jobs,
                "passthrough": passthrough,
                "note": note,
                "no_sync": no_sync,
                "dry_run": dry_run,
                "rolling_origin": rolling_origin,
            }
        )
        return 0

    monkeypatch.setattr("src.benchmarking.parallel_train.orchestrate", _fake)
    return calls


def _patch_platform(monkeypatch, backend: str, cpu_count: int) -> None:
    """Fake ``detect_platform`` (imported at call time inside ``_default_jobs``)."""

    class _Plat:
        pass

    plat = _Plat()
    plat.backend = backend
    plat.cpu_count = cpu_count
    monkeypatch.setattr("src.shared.platform_detect.detect_platform", lambda: plat)


def test_capable_host_autodetects_parallel_all_positions(monkeypatch, delegate):
    _patch_platform(monkeypatch, "cuda", 16)
    rc = benchmark.main([*ALL_SIX, "--dry-run"])
    assert rc == 0
    assert len(delegate) == 1
    assert delegate[0]["jobs"] == 6  # _default_jobs returns n_positions on a capable box
    assert delegate[0]["positions"] == ALL_SIX
    assert delegate[0]["dry_run"] is True


def test_non_cuda_host_runs_sequential(monkeypatch, delegate):
    _patch_platform(monkeypatch, "cpu", 16)
    rc = benchmark.main(["QB", "RB", "--dry-run"])
    assert rc == 0
    assert delegate == []  # sequential path; orchestrator never invoked


def test_low_core_cuda_host_runs_sequential(monkeypatch, delegate):
    # The autodetect gate is cuda AND >=12 cores; an 8-core CUDA box stays sequential.
    _patch_platform(monkeypatch, "cuda", 8)
    rc = benchmark.main([*ALL_SIX, "--dry-run"])
    assert rc == 0
    assert delegate == []


def test_sequential_flag_overrides_autodetect(monkeypatch, delegate):
    _patch_platform(monkeypatch, "cuda", 16)  # capable, but --sequential wins
    rc = benchmark.main([*ALL_SIX, "--sequential", "--dry-run"])
    assert rc == 0
    assert delegate == []


def test_jobs_one_forces_sequential(monkeypatch, delegate):
    _patch_platform(monkeypatch, "cuda", 16)
    rc = benchmark.main(["QB", "RB", "-j", "1", "--dry-run"])
    assert rc == 0
    assert delegate == []


def test_explicit_jobs_delegates_with_that_count(delegate):
    # Explicit -j skips autodetect entirely (no platform patch needed).
    rc = benchmark.main(["QB", "RB", "WR", "-j", "3", "--dry-run"])
    assert rc == 0
    assert len(delegate) == 1
    assert delegate[0]["jobs"] == 3


def test_single_position_never_spawns_even_with_explicit_jobs(delegate):
    # One position can't benefit from a subprocess; the len>1 guard keeps it in-process.
    rc = benchmark.main(["QB", "-j", "4", "--dry-run"])
    assert rc == 0
    assert delegate == []


def test_passthrough_flags_forwarded(monkeypatch, delegate):
    _patch_platform(monkeypatch, "cuda", 16)
    rc = benchmark.main(
        [*ALL_SIX, "--rolling-origin", "--significance", "--no-sync", "--note", "x", "--dry-run"]
    )
    assert rc == 0
    assert len(delegate) == 1
    call = delegate[0]
    assert call["passthrough"] == ["--rolling-origin", "--significance"]
    assert call["rolling_origin"] is True
    assert call["no_sync"] is True
    assert call["note"] == "x"
