"""Real process and output-isolation checks for the shared ablation executor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tuning._execution import isolated_outputs, run_tasks

pytestmark = pytest.mark.unit
_SEEN = 0


def _ridge_task(seed):
    from sklearn.linear_model import Ridge

    global _SEEN
    _SEEN += 1
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(40, 5))
    y = x @ np.arange(5.0) + rng.normal(size=40)
    prediction = Ridge(alpha=3.0).fit(x[:30], y[:30]).predict(x[30:])
    return {"seed": seed, "predictions": prediction, "pid": os.getpid(), "seen": _SEEN}


def _failure(task, exc):
    return {"task": task, "error": f"{type(exc).__name__}: {exc}"}


def _sometimes_fails(value):
    if value == 2:
        raise ValueError("bad cell")
    return {"task": value, "ok": True}


def _native_crash(value):
    if value == 0:
        os._exit(17)
    return {"task": value, "ok": True}


def test_native_crash_is_limited_to_its_cell_and_queued_work_continues():
    rows = run_tasks([0, 1, 2, 3], _native_crash, max_workers=2, on_error=_failure)
    assert "code 17" in rows[0]["error"]
    assert rows[1:] == [{"task": value, "ok": True} for value in (1, 2, 3)]


class _ContextError(Exception):
    def __init__(self, message, *, context):
        super().__init__(message)
        self.context = context


def _bad_exception(value):
    if value == 0:
        # Exception pickling succeeds but reconstruction lacks the keyword arg.
        raise _ContextError("bad variant", context="required")
    return {"task": value, "ok": True}


def test_exception_decode_failure_does_not_stop_other_cells():
    rows = run_tasks([0, 1, 2], _bad_exception, max_workers=2, on_error=_failure)
    assert "context" in rows[0]["error"]
    assert rows[1:] == [{"task": value, "ok": True} for value in (1, 2)]


def test_parallel_ridge_matches_serial_and_each_cell_has_fresh_state():
    # More tasks than workers verifies worker replacement, not just initial spawn.
    seeds = [42, 7, 123, 9]
    serial = run_tasks(seeds, _ridge_task, on_error=_failure)
    parallel = run_tasks(seeds, _ridge_task, max_workers=2, on_error=_failure)
    assert [r["seed"] for r in parallel] == seeds
    assert len({r["pid"] for r in parallel}) == len(seeds)
    assert all(r["seen"] == 1 for r in parallel)
    for left, right in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(left["predictions"], right["predictions"])


def test_worker_environment_precedes_entry_module_import_and_restores_parent(tmp_path):
    # A real CLI is needed: spawn re-imports __main__ before its initializer.
    # Pytest's entry module does not reproduce a numeric CLI's import-time reads.
    script = tmp_path / "worker_environment.py"
    script.write_text(
        "import os\n"
        "BOOT_VALUE = os.environ.get('FF_TEST_WORKER_BOOT')\n"
        "from src.tuning._execution import run_tasks\n"
        "def report(_): return BOOT_VALUE\n"
        "def failed(_, exc): raise exc\n"
        "if __name__ == '__main__':\n"
        "    assert BOOT_VALUE is None\n"
        "    assert run_tasks([1, 2, 3], report, max_workers=2, on_error=failed,\n"
        "        environment={'FF_TEST_WORKER_BOOT': 'ready'}) == ['ready'] * 3\n"
        "    assert 'FF_TEST_WORKER_BOOT' not in os.environ\n"
    )
    environment = dict(os.environ)
    environment.pop("FF_TEST_WORKER_BOOT", None)
    root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [root, environment.get("PYTHONPATH")]))
    subprocess.run([sys.executable, str(script)], env=environment, check=True, timeout=30)


@pytest.mark.parametrize(
    "relative",
    [
        "src/tuning/ab_harness.py",
        "src/tuning/ablate_batch.py",
        "src/analysis/analysis_feature_audit.py",
        "src/analysis/analysis_rb_feature_audit.py",
        "src/analysis/analysis_k_feature_audit.py",
    ],
)
def test_operator_files_bootstrap_without_pythonpath(relative, tmp_path):
    path = Path(__file__).resolve().parents[2] / relative
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='probe')",
            str(path),
        ],
        cwd=tmp_path,
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_failed_cell_does_not_discard_other_results(workers):
    reported = []
    results = run_tasks(
        [1, 2, 3],
        _sometimes_fails,
        max_workers=workers,
        on_error=_failure,
        on_result=lambda index, result: reported.append((index, result)),
    )
    assert results == [
        {"task": 1, "ok": True},
        {"task": 2, "error": "ValueError: bad cell"},
        {"task": 3, "ok": True},
    ]
    assert sorted(index for index, _ in reported) == [0, 1, 2]


def test_isolation_restores_cwd_and_preserves_cache_policy_on_failure(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "ready").write_text("primed")
    monkeypatch.chdir(tmp_path)
    with (
        pytest.raises(RuntimeError, match="training failed"),
        isolated_outputs(
            str(data),
            share_cache=True,
            legacy_cwd=True,
        ),
    ):
        isolated = Path.cwd()
        assert Path("data").resolve() == data
        assert Path(".cache/ready").read_text() == "primed"
        Path("model.pt").write_text("test output")
        raise RuntimeError("training failed")
    assert Path.cwd() == tmp_path
    assert not isolated.exists()
    assert not (tmp_path / "model.pt").exists()
    with isolated_outputs(str(data), legacy_cwd=True):
        assert not Path(".cache").exists()


def test_context_isolation_keeps_cwd_and_separates_output_roots(tmp_path, monkeypatch):
    from src.training.context import current_context

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    with isolated_outputs(str(data)) as first:
        assert Path.cwd() == tmp_path
        path = first.output_dir("RB") / "model"
        path.parent.mkdir(parents=True)
        path.write_text("first")
        with isolated_outputs(str(data)) as second:
            assert Path.cwd() == tmp_path
            assert second.output_root != first.output_root
            assert not (second.output_dir("RB") / "model").exists()
            assert current_context() is second
        assert current_context() is first
        assert path.read_text() == "first"
    assert current_context() is None
    assert not first.output_root.exists()


@pytest.mark.parametrize("kind", ["cell", "group"])
def test_spec_worker_preserves_mode_metrics_and_failure_labels(kind, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.shared.core_pool import ENV_POS
    from src.tuning import ab_harness as harness

    variant = harness.Variant("alternate", label="Detailed report label")
    monkeypatch.setattr(
        harness,
        "resolve_spec",
        lambda *args, **kwargs: SimpleNamespace(variants={"alternate": variant}, metric_fn=None),
    )
    work = (
        harness.Cell("K", "alternate", 7)
        if kind == "cell"
        else harness.Group("RB", "alternate", (7, 9))
    )
    log = tmp_path / "cell.log"
    task = (kind, work, "example.spec", str(tmp_path), 13, str(log))
    called = []

    def execute(work, variant, metric_fn, **kwargs):
        called.append(kwargs)
        if kind == "cell":
            return {"ok": True, "metrics": {"yards_mae": 2.0}, "label": variant.label}
        return [{"ok": True, "seed": seed, "metrics": {"yards_mae": 2.0}} for seed in work.seeds]

    method = "run_cell" if kind == "cell" else "run_group_stacked"
    monkeypatch.setattr(harness, method, execute)
    monkeypatch.setenv("FF_FEATURE_CACHE_DISABLE", "0")
    monkeypatch.setenv(ENV_POS, "test-parent")
    rows = harness._execute_spec_task(task)
    assert all(row["metrics"]["yards_mae"] == 2.0 for row in rows)
    assert called[0]["data_dir"] == str(tmp_path)
    assert ("stacked_epochs" in called[0]) == (kind == "group")
    if kind == "group":
        assert called[0]["stacked_epochs"] == 13
    assert os.environ["FF_FEATURE_CACHE_DISABLE"] == "0"

    def fail(*args, **kwargs):
        raise RuntimeError("failed model")

    monkeypatch.setattr(harness, method, fail)
    failed = harness._execute_spec_task(task)
    assert len(failed) == (1 if kind == "cell" else 2)
    assert all(not row["ok"] and row["label"] == variant.label for row in failed)
    assert "failed model" in log.read_text()
