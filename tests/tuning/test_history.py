"""Unit tests for src/tuning/history.py.

``append_tuning_run`` wraps the shared ``append_to_history`` helper to record
one git-tracked JSON per ``tune_nn`` / ``tune_lgbm`` run under
``benchmark_history/tuning/``. These tests pin the entry schema and the
filename / run_id convention so a refactor can't silently change what gets
committed, and guard that the default target stays a tracked path.
"""

from __future__ import annotations

import json
import os

import pytest

from src.tuning import history

pytestmark = pytest.mark.unit


@pytest.fixture
def _frozen(monkeypatch):
    """Freeze timestamp + git hash so run_id / filename are deterministic."""
    monkeypatch.setattr(history, "utc_now_iso", lambda: "2026-05-29T12:00:00")
    monkeypatch.setattr(history, "get_git_hash", lambda: "abc1234")


def test_writes_json_with_full_schema(tmp_path, _frozen):
    results = {
        "QB": {
            "best_trial": 2,
            "best_val_loss": 1.5,
            "best_params": {"attn_d_model": 32},
            "n_trials": 15,
        },
        "RB": {"best_trial": 7, "best_val_loss": 0.9, "best_params": {}, "n_trials": 15},
    }

    path = history.append_tuning_run(
        "tune_nn", results, n_trials=15, positions=["QB", "RB"], history_dir=str(tmp_path)
    )

    assert os.path.exists(path)
    # Colons in the ISO timestamp are sanitized to hyphens by append_to_history.
    assert os.path.basename(path) == "2026-05-29T12-00-00_abc1234_tune_nn.json"

    with open(path) as f:
        entry = json.load(f)
    assert entry["run_id"] == "2026-05-29T12:00:00_abc1234_tune_nn"
    assert entry["run_id"].endswith("_tune_nn")
    assert entry["kind"] == "tune_nn"
    assert entry["git_hash"] == "abc1234"
    assert entry["n_trials"] == 15
    assert entry["positions"] == ["QB", "RB"]
    assert entry["results"] == results  # stored verbatim
    assert "note" not in entry


def test_positions_default_to_sorted_results(tmp_path, _frozen):
    results = {"WR": {}, "DST": {}, "K": {}}

    path = history.append_tuning_run("tune_lgbm", results, history_dir=str(tmp_path))

    with open(path) as f:
        entry = json.load(f)
    assert entry["kind"] == "tune_lgbm"
    assert entry["run_id"].endswith("_tune_lgbm")
    assert entry["positions"] == ["DST", "K", "WR"]  # sorted(results)
    assert entry["n_trials"] is None


def test_note_recorded_when_provided(tmp_path, _frozen):
    path = history.append_tuning_run(
        "tune_nn", {"QB": {}}, note="backfilled", history_dir=str(tmp_path)
    )

    with open(path) as f:
        entry = json.load(f)
    assert entry["note"] == "backfilled"


def test_default_history_dir_is_tracked_subdir():
    # Guard the committed location so it can't drift to a gitignored path.
    expected = os.path.join("benchmark_history", "tuning")
    assert expected == history.HISTORY_DIR
