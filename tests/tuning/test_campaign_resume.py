"""Resume must retain completed work, enforce identity, and repair lost outputs."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import optuna
import pytest

from src.tuning import campaign_worker as worker
from src.tuning.campaign_io import CellCheckpoint, Journal, atomic_json
from src.tuning.study_checkpoint import (
    CampaignBudget,
    recover_trials,
    remaining_attempts,
    sqlite_backup,
)

pytestmark = pytest.mark.unit


def test_sqlite_backup_includes_committed_wal_and_excludes_uncommitted_rows(tmp_path):
    path = tmp_path / "study.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE evidence (trial INTEGER)")
        connection.execute("INSERT INTO evidence VALUES (1)")
        connection.commit()
        connection.execute("INSERT INTO evidence VALUES (2)")
        snapshot = Path(sqlite_backup(path))
        try:
            with sqlite3.connect(snapshot) as read:
                assert read.execute("SELECT trial FROM evidence").fetchall() == [(1,)]
        finally:
            snapshot.unlink()


def test_trial_budget_survives_restart_and_counts_interrupted_attempts():
    study = optuna.create_study()
    complete = study.ask()
    study.tell(complete, 1.0)
    abandoned = study.ask()
    recover_trials(study)
    assert study.trials[abandoned.number].state == optuna.trial.TrialState.FAIL
    assert remaining_attempts(study, 3) == 1
    recover_trials(study)
    assert len(study.trials) == 2


def test_active_timeout_resumes_without_charging_queue_time(monkeypatch):
    import src.tuning.study_checkpoint as checkpoint

    clock = [100.0]
    monkeypatch.setattr(checkpoint.time, "monotonic", lambda: clock[0])
    study = optuna.create_study()
    budget = CampaignBudget(study, 60)
    with budget:
        clock[0] += 20
        budget.save()
    clock[0] += 10000
    resumed = CampaignBudget(study, 60)
    assert resumed.remaining == 40
    with resumed:
        clock[0] += 10
    assert study.user_attrs["campaign_active_seconds"] == 30


def test_checkpoint_detects_tampering_and_other_manifest(tmp_path):
    journal = Journal(tmp_path)
    checkpoint = CellCheckpoint(journal, "cells", "identity-one")
    checkpoint.save("RB/baseline", {"metric": 1.2})
    assert checkpoint.load("RB/baseline") == {"metric": 1.2}
    with pytest.raises(ValueError, match="different campaign"):
        CellCheckpoint(journal, "cells", "identity-two").load("RB/baseline")
    name = checkpoint._name("RB/baseline")
    value, token = journal.read(name)
    value["result"]["metric"] = 4.5
    journal.write(name, value, token)
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.load("RB/baseline")


def test_ab_resume_keeps_successful_cells_when_another_cell_fails(tmp_path, monkeypatch):
    from src.shared import core_pool
    from src.tuning import ab_harness

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        core_pool, "start_coordinator", lambda *a: ("unused", lambda n: None, lambda: None)
    )
    spec = ab_harness.resolve_spec("src.tuning.ab_example", positions=["RB"], seeds=[42])
    units = [("cell", cell) for cell in ab_harness.build_cells(spec)]
    calls = []
    fail = [True]

    def execute(task):
        key = task[1].key
        calls.append(key)
        ok = not (key == units[1][1].key and fail[0])
        if not ok:
            fail[0] = False
        return [{"ok": ok, "metric": 1.0}]

    monkeypatch.setattr(ab_harness, "_execute_spec_task", execute)
    checkpoint = CellCheckpoint(Journal(tmp_path / "journal"), "cells", "frozen")
    first = ab_harness._run_parallel_units(spec, units, 1, str(tmp_path), 30, checkpoint)
    assert sum(row["ok"] for row in first) == len(units) - 1
    second = ab_harness._run_parallel_units(spec, units, 1, str(tmp_path), 30, checkpoint)
    assert all(row["ok"] for row in second)
    assert len(calls) == len(units) + 1


def test_benchmark_resume_runs_only_missing_or_corrupt_folds(tmp_path, monkeypatch):
    from src import config
    from src.benchmarking import benchmark
    from src.shared import core_pool

    monkeypatch.setattr(config, "ROLLING_ORIGIN_TEST_SEASONS", [2024, 2025])
    monkeypatch.setattr(
        core_pool, "start_coordinator", lambda *a: ("unused", lambda n: None, lambda: None)
    )
    monkeypatch.setattr(
        benchmark, "finalize_rolling_origin", lambda pos, rows: {"position": pos, "rows": rows}
    )
    calls = []
    fail_once = [True]

    def train(task):
        calls.append(task["key"])
        if task["key"] == "RB-2025" and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("interrupted fold")
        row = {"position": task["position"], "origin": task["origin"], "summary": {"metric": 1}}
        atomic_json(Path(task["output"]) / "cell_result.json", row)
        return row

    monkeypatch.setattr(worker, "_benchmark_cell", train)
    checkpoint = CellCheckpoint(Journal(tmp_path / "journal"), "cells", "frozen")
    step = {"positions": ["RB", "WR"], "options": {"rolling_origin": True, "jobs": 1}}
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="1 benchmark cells failed"):
        worker.benchmark_step(step, tmp_path, output, checkpoint)
    assert len(calls) == 4
    result = worker.benchmark_step(step, tmp_path, output, checkpoint)
    assert calls[-1] == "RB-2025" and len(calls) == 5
    assert len(result["results"]) == 2
    worker.benchmark_step(step, tmp_path, output, checkpoint)
    assert len(calls) == 5
    (output / "WR-2024/cell_result.json").write_text("corrupt")
    worker.benchmark_step(step, tmp_path, output, checkpoint)
    assert calls[-1] == "WR-2024" and len(calls) == 6


def minimal_manifest():
    return {
        "id": "test",
        "manifest_id": "identity",
        "backend": "local",
        "fresh": False,
        "code_sha": "a" * 40,
        "dataset_id": "b" * 64,
        "bucket": None,
        "execution_environment": {"FF_DEVICE": "cpu", "FF_NN_FIXED_EPOCHS": "2"},
        "spec": {
            "steps": [
                {"id": name, "kind": "benchmark", "positions": ["RB"], "options": {}, "env": {}}
                for name in ["first", "second"]
            ]
        },
    }


def test_resume_skips_verified_step_and_retries_failed_or_corrupt_output(tmp_path, monkeypatch):
    manifest = minimal_manifest()
    unit = {"id": "local", "position": None, "resource": "local", "steps": ["first", "second"]}
    journal = Journal(tmp_path / "journal")
    calls = []

    def launch(argv, **kwargs):
        name = argv[argv.index("--step") + 1]
        calls.append(name)
        code = 1 if calls == ["first", "second"] else 0
        if not code:
            atomic_json(Path(argv[argv.index("--output") + 1]) / "result.json", {"metric": 1})
        return Mock(wait=Mock(return_value=code), poll=Mock(return_value=code))

    monkeypatch.setattr(worker, "verify_runtime", lambda *a: None)
    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    directory = tmp_path / "unit"
    args = dict(data_dir=tmp_path, directory=directory)
    assert worker.run_unit(manifest, unit, journal, **args) == 1
    assert worker.run_unit(manifest, unit, journal, **args) == 0
    assert calls == ["first", "second", "second"]
    (directory / "steps/first/result.json").write_text("tampered")
    assert worker.run_unit(manifest, unit, journal, **args) == 0
    assert calls == ["first", "second", "second", "first"]
    progress = json.loads((journal.root / "units/local/progress.json").read_text())
    assert all(s["state"] == "SUCCEEDED" for s in progress["steps"].values())


def test_child_cannot_inherit_unfrozen_model_or_dispatch_overrides(tmp_path, monkeypatch):
    manifest = minimal_manifest()
    monkeypatch.setenv("FF_NN_FIXED_EPOCHS", "900")
    monkeypatch.setenv("FF_TUNE_ABLATE_MOD", "src.tuning.unrelated")
    monkeypatch.setenv("FF_NEW_NUMERICAL_FLAG", "1")
    env = worker.child_environment(
        manifest,
        manifest["spec"]["steps"][0],
        {"id": "local", "resource": "local"},
        tmp_path,
        tmp_path,
    )
    assert env["FF_NN_FIXED_EPOCHS"] == "2"
    assert "FF_TUNE_ABLATE_MOD" not in env
    assert "FF_NEW_NUMERICAL_FLAG" not in env
    assert env["FF_FRESH"] == "1"
