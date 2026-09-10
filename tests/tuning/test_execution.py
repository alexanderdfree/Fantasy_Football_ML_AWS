"""Real process and output-isolation checks for the shared ablation executor."""

from __future__ import annotations

import os
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
    with isolated_outputs(str(data)):
        assert not Path(".cache").exists()


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
