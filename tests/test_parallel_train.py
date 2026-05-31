"""Unit tests for src/benchmarking/parallel_train.py.

Covers the pure scheduling logic (physical-core split, heaviest-first ordering), the
single-consolidated-record merge, and the dry-run plan path — all without launching a
subprocess or training a model. The orchestration loop's process management is left to
the integration smoke (a real ``--dry-run`` / small run) documented in SETUP.md.
"""

from __future__ import annotations

import json

import pytest

import src.benchmarking.parallel_train as pt

pytestmark = pytest.mark.unit


def test_module_exposes_api():
    for fn in (
        "orchestrate",
        "physical_cores",
        "_split_cores",
        "_sort_by_cost",
        "_run_worker",
        "_record_and_sync",
        "main",
    ):
        assert hasattr(pt, fn)


# --------------------------------------------------------------------- core split


@pytest.mark.parametrize(
    "n,expected_sizes",
    [
        (6, [3, 3, 3, 3, 2, 2]),  # 16 physical cores, 6 positions
        (4, [4, 4, 4, 4]),
        (1, [16]),
        (16, [1] * 16),
    ],
)
def test_split_cores_sizes_and_coverage(n, expected_sizes):
    phys = list(range(16))
    chunks = pt._split_cores(phys, n)
    assert [len(c) for c in chunks] == expected_sizes
    # disjoint, contiguous, and covering every physical core exactly once
    flat = [c for chunk in chunks for c in chunk]
    assert flat == phys
    for chunk in chunks:
        assert chunk == list(range(chunk[0], chunk[0] + len(chunk)))


def test_split_cores_clamps_when_more_positions_than_cores():
    chunks = pt._split_cores([0, 1, 2], 5)
    assert chunks == [[0], [1], [2]]  # clamped to one core each, no empties


def test_sort_by_cost_heaviest_first():
    # arbitrary input order -> WR/RB/QB ahead of the lighter TE/DST/K
    assert pt._sort_by_cost(["K", "QB", "DST", "WR", "TE", "RB"]) == [
        "WR",
        "RB",
        "QB",
        "TE",
        "DST",
        "K",
    ]


# ------------------------------------------------------------------------- merge


def _fake_summary(pos, ridge_mae):
    return {
        "position": pos,
        "ridge_mae": ridge_mae,
        "nn_mae": ridge_mae - 0.1,
        "elapsed_sec": 12.3,
    }


def test_record_and_sync_produces_one_consolidated_entry(tmp_path, monkeypatch):
    """Two per-position summaries -> a single history entry listing both positions."""
    monkeypatch.chdir(tmp_path)
    captured = {}

    monkeypatch.setattr(pt, "print_comparison_table", lambda *a, **k: None)
    monkeypatch.setattr(pt, "print_history_comparison", lambda *a, **k: None)
    monkeypatch.setattr(pt, "collect_global_config", lambda: {"train_seasons": [2012]})
    monkeypatch.setattr(pt, "collect_pos_config", lambda p: {"name": p})
    monkeypatch.setattr(pt, "get_git_hash", lambda: "abc1234")
    monkeypatch.setattr(pt, "utc_now_iso", lambda: "2026-01-01T00:00:00")

    def _fake_append(history_dir, entry):
        captured["entry"] = entry
        return f"{history_dir}/{entry['run_id']}.json"

    monkeypatch.setattr(pt, "append_to_history", _fake_append)

    sync_calls = []
    monkeypatch.setattr(pt, "_maybe_upload_to_s3", lambda p: sync_calls.append(p))

    summaries = [_fake_summary("QB", 6.3), _fake_summary("K", 1.1)]
    pt._record_and_sync(summaries, ["QB", "K"], note="parallel run", no_sync=False)

    entry = captured["entry"]
    assert entry["positions"] == ["QB", "K"]
    assert entry["results"] == summaries
    assert entry["run_id"] == "2026-01-01T00:00:00_abc1234"
    assert entry["note"] == "parallel run"
    # per-position config folded in under lowercase keys
    assert entry["config"]["qb"] == {"name": "QB"} and entry["config"]["k"] == {"name": "K"}
    # default (no --no-sync) mirrors to S3 exactly once
    assert sync_calls == ["benchmark_history/2026-01-01T00:00:00_abc1234.json"]
    # the backwards-compat latest-results file holds both positions
    assert json.loads((tmp_path / "benchmark_results.json").read_text()) == summaries


def test_record_and_sync_no_sync_skips_s3(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pt, "print_comparison_table", lambda *a, **k: None)
    monkeypatch.setattr(pt, "print_history_comparison", lambda *a, **k: None)
    monkeypatch.setattr(pt, "collect_global_config", lambda: {})
    monkeypatch.setattr(pt, "collect_pos_config", lambda p: {})
    monkeypatch.setattr(pt, "get_git_hash", lambda: "deadbee")
    monkeypatch.setattr(pt, "utc_now_iso", lambda: "2026-01-01T00:00:00")
    monkeypatch.setattr(pt, "append_to_history", lambda d, e: f"{d}/x.json")

    called = []
    monkeypatch.setattr(pt, "_maybe_upload_to_s3", lambda p: called.append(p))
    pt._record_and_sync([_fake_summary("QB", 6.3)], ["QB"], note="", no_sync=True)
    assert called == []


# ----------------------------------------------------------------------- dry run


def test_dry_run_launches_nothing(monkeypatch, capsys):
    monkeypatch.setattr(pt, "physical_cores", lambda: list(range(8)))

    def _boom(*a, **k):
        raise AssertionError("dry-run must not spawn a process")

    monkeypatch.setattr(pt.subprocess, "Popen", _boom)

    rc = pt.main(["QB", "K", "-j", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "QB" in out and "K" in out
