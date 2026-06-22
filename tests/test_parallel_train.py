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
        "_history_cost_order",
        "_run_worker",
        "_record_and_sync",
        "_build_cells",
        "_merge_cell_results",
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


def test_history_cost_order_slowest_first(tmp_path):
    hist = tmp_path / "h"
    hist.mkdir()
    # an entry with no per-position timings is skipped...
    (hist / "2026-01-01T00-00-00_aaa.json").write_text(json.dumps({"results": []}))
    # ...in favour of the newest entry that carries elapsed_sec, ordered slowest-first
    (hist / "2026-01-02T00-00-00_bbb.json").write_text(
        json.dumps(
            {
                "results": [
                    {"position": "QB", "elapsed_sec": 90},
                    {"position": "DST", "elapsed_sec": 200},
                    {"position": "K", "elapsed_sec": 50},
                ]
            }
        )
    )
    assert pt._history_cost_order(str(hist)) == ["DST", "QB", "K"]


def test_history_cost_order_none_when_no_usable_entry(tmp_path):
    assert pt._history_cost_order(str(tmp_path)) is None


def test_sort_by_cost_history_then_fallback(monkeypatch):
    # measured history wins when present
    monkeypatch.setattr(pt, "_history_cost_order", lambda *a, **k: ["DST", "K", "QB"])
    assert pt._sort_by_cost(["QB", "DST", "K"]) == ["DST", "K", "QB"]
    # no usable history -> static _COST_ORDER fallback (WR/RB/QB ahead of TE/DST/K)
    monkeypatch.setattr(pt, "_history_cost_order", lambda *a, **k: None)
    assert pt._sort_by_cost(["K", "QB", "DST", "WR", "TE", "RB"]) == [
        "WR",
        "RB",
        "QB",
        "TE",
        "DST",
        "K",
    ]


def test_launch_wires_pool_env_and_drops_frozen_thread_caps(tmp_path, monkeypatch):
    # The pool supplies thread counts at runtime, so _launch must NOT freeze
    # LGBM_N_JOBS / LOKY_MAX_CPU_COUNT per slice (the bug the pool fixes).
    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LGBM_N_JOBS",
        "LOKY_MAX_CPU_COUNT",
    ):
        monkeypatch.delenv(v, raising=False)
    captured = {}

    class _FakeProc:
        pid = 4321

    captured["argv"] = None

    def _fake_popen(argv, env=None, **kwargs):
        captured["env"] = env
        captured["argv"] = argv
        return _FakeProc()

    monkeypatch.setattr(pt.subprocess, "Popen", _fake_popen)
    # Single-split cell: cell_key == pos, origin None.
    info = pt._launch(
        "QB", "QB", None, [0, 2, 4], str(tmp_path), str(tmp_path), [], pool_addr="/tmp/pool.sock"
    )
    info["logf"].close()
    env = captured["env"]
    assert env[pt.ENV_ADDR] == "/tmp/pool.sock"  # worker wired to the pool
    assert env[pt.ENV_POS] == "QB"
    assert env["OMP_NUM_THREADS"] == "1"  # BLAS still capped to 1
    assert "LGBM_N_JOBS" not in env  # no longer frozen per slice
    assert "LOKY_MAX_CPU_COUNT" not in env
    assert "--origin" not in captured["argv"]  # no origin flag for a single-split cell


def test_launch_rolling_origin_cell_isolates_files_and_passes_origin(tmp_path, monkeypatch):
    # A (position × origin) cell keys its summary/log on cell_key (e.g. "RB:2025" ->
    # "RB-2025") so concurrent origins of one position never collide, and forwards
    # --origin <ts> so the worker scores just that origin.
    captured = {}

    class _FakeProc:
        pid = 99

    def _fake_popen(argv, env=None, **kwargs):
        captured["argv"] = argv
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(pt.subprocess, "Popen", _fake_popen)
    info = pt._launch(
        "RB:2025",
        "RB",
        2025,
        [0, 1],
        str(tmp_path),
        str(tmp_path),
        ["--rolling-origin"],
        pool_addr="/tmp/p.sock",
    )
    info["logf"].close()
    argv = captured["argv"]
    assert "--worker" in argv and argv[argv.index("--worker") + 1] == "RB"
    assert "--origin" in argv and argv[argv.index("--origin") + 1] == "2025"
    assert info["summary_path"].endswith("RB-2025.json")
    assert info["log_path"].endswith("local-train-RB-2025.log")
    assert info["cell_key"] == "RB:2025" and info["origin"] == 2025
    assert captured["env"][pt.ENV_POS] == "RB:2025"


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
    pt._record_and_sync(
        summaries, ["QB", "K"], note="parallel run", no_sync=False, total_wall_sec=137.0
    )

    entry = captured["entry"]
    assert entry["positions"] == ["QB", "K"]
    assert entry["results"] == summaries
    assert entry["run_id"] == "2026-01-01T00:00:00_abc1234"
    assert entry["note"] == "parallel run"
    assert entry["total_wall_sec"] == 137.0  # recorded automatically on every run
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


def test_run_worker_rolling_origin_no_origin_loops_all(tmp_path, monkeypatch):
    """The in-process fallback (no --origin): the worker scores every origin via
    ``run_rolling_origin`` and writes the production-origin summary + block."""
    calls = []

    def _fake_rolling_origin(pos):
        calls.append(pos)
        return {
            "position": pos,
            "ridge_mae": 4.5,
            "rolling_origin": {"test_seasons": [2025], "n_origins": 1},
        }

    monkeypatch.setattr(pt, "run_rolling_origin", _fake_rolling_origin)
    monkeypatch.setattr(pt, "run_one", lambda *a, **k: pytest.fail("run_one must not handle --cv"))
    monkeypatch.setattr(
        pt, "score_one_origin", lambda *a, **k: pytest.fail("no --origin -> must loop in-process")
    )

    out_path = tmp_path / "RB.json"
    rc = pt._run_worker("RB", str(out_path), rolling_origin=True, significance=True)

    assert rc == 0
    assert calls == ["RB"]
    assert json.loads(out_path.read_text()) == {
        "position": "RB",
        "ridge_mae": 4.5,
        "rolling_origin": {"test_seasons": [2025], "n_origins": 1},
    }


def test_run_worker_rolling_origin_with_origin_scores_one_cell(tmp_path, monkeypatch):
    """With --origin set, the worker scores ONLY that origin via ``score_one_origin``
    (never loops every origin) and writes that single-origin summary."""
    seen = {}

    def _fake_score_one_origin(pos, test_season):
        seen["args"] = (pos, test_season)
        return test_season, {"position": pos, "ridge_mae": 4.4, "test_season": test_season}

    monkeypatch.setattr(pt, "score_one_origin", _fake_score_one_origin)
    monkeypatch.setattr(
        pt,
        "run_rolling_origin",
        lambda pos: pytest.fail("with --origin the worker must score one origin only"),
    )
    monkeypatch.setattr(pt, "run_one", lambda *a, **k: pytest.fail("rolling-origin path"))

    out_path = tmp_path / "RB-2024.json"
    rc = pt._run_worker("RB", str(out_path), rolling_origin=True, significance=False, origin="2024")

    assert rc == 0
    assert seen["args"] == ("RB", 2024)  # --origin parsed to int
    assert json.loads(out_path.read_text()) == {
        "position": "RB",
        "ridge_mae": 4.4,
        "test_season": 2024,
    }


def test_run_worker_single_split_calls_run_one_with_real_signature(tmp_path, monkeypatch):
    """Regression for the parallel-trainer ``run_one(cv=...)`` drift: PR #719
    removed ``run_one``'s ``cv`` param but this caller still passed ``cv=False``
    (CI never exercises the parallel trainer, so it broke silently). The stub
    mirrors ``benchmark.run_one``'s REAL signature ``run_one(position)`` so any
    stray kwarg raises TypeError — the rolling-origin test above uses
    ``lambda *a, **k`` and cannot catch a signature drift."""
    calls = []

    def _fake_run_one(position):  # must mirror benchmark.run_one's real signature
        calls.append(position)
        return {"position": position, "ridge_mae": 4.2}

    monkeypatch.setattr(pt, "run_one", _fake_run_one)
    monkeypatch.setattr(
        pt, "summarize_pipeline_result", lambda pos, result: {"position": pos, "summarized": True}
    )
    monkeypatch.setattr(
        pt,
        "run_rolling_origin",
        lambda pos: pytest.fail("single-split path must not call run_rolling_origin"),
    )

    out_path = tmp_path / "RB.json"
    rc = pt._run_worker("RB", str(out_path), rolling_origin=False, significance=False)

    assert rc == 0
    assert calls == ["RB"]
    assert json.loads(out_path.read_text()) == {"position": "RB", "summarized": True}


def test_record_and_sync_rolling_origin_marks_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}
    printed_rolling = []

    monkeypatch.setattr(pt, "_print_rolling_origin_table", lambda s: printed_rolling.append(s))
    monkeypatch.setattr(pt, "print_comparison_table", lambda *a, **k: None)
    monkeypatch.setattr(pt, "print_history_comparison", lambda *a, **k: None)
    monkeypatch.setattr(pt, "collect_global_config", lambda: {})
    monkeypatch.setattr(pt, "collect_pos_config", lambda p: {})
    monkeypatch.setattr(pt, "get_git_hash", lambda: "abc1234")
    monkeypatch.setattr(pt, "utc_now_iso", lambda: "2026-06-01T00:00:00")
    monkeypatch.setattr(pt, "_maybe_upload_to_s3", lambda p: pytest.fail("unexpected S3 sync"))

    def _fake_append(history_dir, entry):
        captured["entry"] = entry
        return f"{history_dir}/{entry['run_id']}.json"

    monkeypatch.setattr(pt, "append_to_history", _fake_append)

    summary = _fake_summary("RB", 4.5)
    summary["rolling_origin"] = {"test_seasons": [2025], "n_origins": 1}
    pt._record_and_sync(
        [summary],
        ["RB"],
        note="rolling run",
        no_sync=True,
        total_wall_sec=42.0,
        rolling_origin=True,
    )

    assert printed_rolling == [[summary]]
    assert captured["entry"]["mode"] == "rolling_origin"
    assert captured["entry"]["results"] == [summary]


# ------------------------------------------------------------ cell flattening / merge


def test_build_cells_single_split_one_cell_per_position(monkeypatch):
    monkeypatch.setattr(pt, "_sort_by_cost", lambda ps: ["WR", "QB"])
    cells = pt._build_cells(["QB", "WR"], rolling_origin=False)
    assert cells == [("WR", "WR", None), ("QB", "QB", None)]


def test_build_cells_rolling_origin_position_times_origin_latest_first(monkeypatch):
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    monkeypatch.setattr(pt, "_sort_by_cost", lambda ps: ["RB", "QB"])
    cells = pt._build_cells(["QB", "RB"], rolling_origin=True)
    # heaviest position first (RB), and within each position the latest origin first
    # (reversed(ROLLING_ORIGIN_TEST_SEASONS)) since the latest year trains on the most data.
    latest_first = list(reversed(ROLLING_ORIGIN_TEST_SEASONS))
    expected = [(f"{p}:{ts}", p, ts) for p in ("RB", "QB") for ts in latest_first]
    assert cells == expected
    assert len(cells) == 2 * len(ROLLING_ORIGIN_TEST_SEASONS)


def test_merge_cell_results_single_split_passthrough():
    cells = [("QB", "QB", None), ("K", "K", None)]
    results = {"QB": {"position": "QB"}, "K": {"position": "K"}}
    summaries, ordered, failed = pt._merge_cell_results(
        ["QB", "K"], cells, results, rolling_origin=False
    )
    assert ordered == ["QB", "K"]
    assert failed == []
    assert summaries == [{"position": "QB"}, {"position": "K"}]


def test_merge_cell_results_rolling_origin_finalizes_per_position(monkeypatch):
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    seasons = list(ROLLING_ORIGIN_TEST_SEASONS)
    cells = [(f"RB:{ts}", "RB", ts) for ts in reversed(seasons)]
    results = {f"RB:{ts}": {"position": "RB", "test_season": ts} for ts in seasons}

    captured = {}

    def _fake_finalize(pos, per_origin):
        captured["pos"] = pos
        captured["per_origin"] = per_origin
        return {"position": pos, "finalized": True}

    monkeypatch.setattr(pt, "finalize_rolling_origin", _fake_finalize)
    summaries, ordered, failed = pt._merge_cell_results(["RB"], cells, results, rolling_origin=True)
    assert ordered == ["RB"]
    assert failed == []
    assert summaries == [{"position": "RB", "finalized": True}]
    # finalize is fed the origins in ROLLING_ORIGIN_TEST_SEASONS (chronological) order.
    assert [ts for ts, _ in captured["per_origin"]] == seasons
    assert captured["pos"] == "RB"


def test_merge_cell_results_rolling_origin_partial_position_is_failed(monkeypatch):
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    seasons = list(ROLLING_ORIGIN_TEST_SEASONS)
    cells = [(f"RB:{ts}", "RB", ts) for ts in reversed(seasons)]
    # one origin failed (None) -> the whole position is FAILED, not a partial mean.
    results = {f"RB:{ts}": {"position": "RB"} for ts in seasons}
    results[f"RB:{seasons[0]}"] = None

    monkeypatch.setattr(
        pt,
        "finalize_rolling_origin",
        lambda *a, **k: pytest.fail("a position missing an origin must not be finalized"),
    )
    summaries, ordered, failed = pt._merge_cell_results(["RB"], cells, results, rolling_origin=True)
    assert ordered == []
    assert failed == ["RB"]
    assert summaries == []


class _DoneProc:
    """A fake Popen that is immediately finished with returncode 0."""

    def __init__(self, *a, **k):
        self.pid = 1
        self.returncode = 0

    def poll(self):
        return 0


def test_orchestrate_rolling_origin_dispatches_cells_and_merges(tmp_path, monkeypatch):
    """End-to-end (process-faked) orchestrate for --rolling-origin: it launches one
    worker per (position × origin) cell, then merges each position's origins via
    finalize_rolling_origin into ONE per-position summary."""
    from src.config import ROLLING_ORIGIN_TEST_SEASONS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pt, "physical_cores", lambda: list(range(4)))
    monkeypatch.setattr(pt, "_sort_by_cost", lambda ps: ["RB", "QB"])
    # Stub the core-pool coordinator so no socket is opened.
    monkeypatch.setattr(
        pt, "start_coordinator", lambda phys, tmp: ("/tmp/p.sock", lambda n: None, lambda: None)
    )

    launched = []

    def _fake_launch(cell_key, pos, origin, cores, tmpdir, logdir, passthrough, pool_addr):
        launched.append((cell_key, pos, origin))
        # Write the per-cell summary the worker would have produced.
        summary_path = tmp_path / f"{pt._cell_slug(cell_key)}.json"
        summary_path.write_text(json.dumps({"position": pos, "test_season": origin}))
        log_path = tmp_path / f"{pt._cell_slug(cell_key)}.log"
        logf = open(log_path, "w")  # noqa: SIM115
        return {
            "cell_key": cell_key,
            "pos": pos,
            "origin": origin,
            "proc": _DoneProc(),
            "cores": cores,
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "logf": logf,
            "t0": 0.0,
        }

    monkeypatch.setattr(pt, "_launch", _fake_launch)

    finalized = []

    def _fake_finalize(pos, per_origin):
        finalized.append((pos, [ts for ts, _ in per_origin]))
        return {"position": pos, "finalized": True}

    monkeypatch.setattr(pt, "finalize_rolling_origin", _fake_finalize)

    recorded = {}

    def _fake_record(summaries, positions, note, no_sync, total_wall_sec, rolling_origin=False):
        recorded["summaries"] = summaries
        recorded["positions"] = positions
        recorded["rolling_origin"] = rolling_origin

    monkeypatch.setattr(pt, "_record_and_sync", _fake_record)

    rc = pt.orchestrate(
        ["QB", "RB"],
        jobs=4,
        passthrough=["--rolling-origin"],
        note="",
        no_sync=True,
        dry_run=False,
        rolling_origin=True,
    )

    assert rc == 0
    n = len(ROLLING_ORIGIN_TEST_SEASONS)
    # 2 positions × n origins cells launched.
    assert len(launched) == 2 * n
    assert {pos for _, pos, _ in launched} == {"QB", "RB"}
    # one finalized summary per position, each fed all origins (chronological).
    assert sorted(p for p, _ in finalized) == ["QB", "RB"]
    for _, season_order in finalized:
        assert season_order == list(ROLLING_ORIGIN_TEST_SEASONS)
    # records in the ORIGINAL positions order, one summary per position.
    assert recorded["positions"] == ["QB", "RB"]
    assert recorded["summaries"] == [
        {"position": "QB", "finalized": True},
        {"position": "RB", "finalized": True},
    ]
    assert recorded["rolling_origin"] is True


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


def test_main_cv_alias_routes_parallel_runner_to_rolling_origin(monkeypatch, capsys):
    captured = {}

    def _fake_orchestrate(positions, jobs, passthrough, note, no_sync, dry_run, rolling_origin):
        captured.update(
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

    monkeypatch.setattr(pt, "_default_jobs", lambda n: 1)
    monkeypatch.setattr(pt, "orchestrate", _fake_orchestrate)

    rc = pt.main(["QB", "--cv", "--dry-run", "--note", "check", "--no-sync"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "DEPRECATED: parallel_train --cv now aliases --rolling-origin" in out
    assert captured == {
        "positions": ["QB"],
        "jobs": 1,
        "passthrough": ["--rolling-origin"],
        "note": "check",
        "no_sync": True,
        "dry_run": True,
        "rolling_origin": True,
    }


def test_main_rolling_origin_with_cv_runs_once_without_deprecation(monkeypatch, capsys):
    captured = {}

    def _fake_orchestrate(positions, jobs, passthrough, note, no_sync, dry_run, rolling_origin):
        captured.update(
            {
                "positions": positions,
                "passthrough": passthrough,
                "rolling_origin": rolling_origin,
            }
        )
        return 0

    monkeypatch.setattr(pt, "_default_jobs", lambda n: 1)
    monkeypatch.setattr(pt, "orchestrate", _fake_orchestrate)

    rc = pt.main(["RB", "--rolling-origin", "--cv"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "DEPRECATED" not in out
    assert captured == {
        "positions": ["RB"],
        "passthrough": ["--rolling-origin"],
        "rolling_origin": True,
    }
