"""Unit tests for src/tuning/ab_batch.py (Batch container entry for the A/B
harness).

No AWS, no pipeline runs — boto3 and run_cell are mocked; we assert the
checkpoint/resume contract (which cells run, what lands at which S3 key) and
the tune_nn env-flag dispatch route.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from src.tuning import ab_batch

pytestmark = pytest.mark.unit

SPEC = "src.tuning.ab_example"


def _fake_s3(existing_keys=()):
    """MagicMock S3 client whose list paginator reports ``existing_keys``."""
    s3 = MagicMock()
    page = {"Contents": [{"Key": k} for k in existing_keys]} if existing_keys else {}
    s3.get_paginator.return_value.paginate.return_value = [page]
    return s3


def _ok_result(cell, variant, metric_fn, *, data_dir):
    return {
        "position": cell.position,
        "variant": cell.variant,
        "seed": cell.seed,
        "label": cell.variant,
        "ok": True,
        "metrics": {"Ridge": {"mae": 1.0}},
        "ridge_mae": 1.0,
        "error": None,
    }


@pytest.fixture
def batch_env(monkeypatch):
    monkeypatch.setenv("FF_TUNE_AB_SPEC", SPEC)
    monkeypatch.setenv("FF_AB_RUN_ID", "test-run")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_AB_SEEDS", "42,123")
    monkeypatch.setenv("FF_AB_ONLY", "nn_dropout=0")
    monkeypatch.delenv("FF_AB_S3_PREFIX", raising=False)
    # Keep the entry hermetic: no S3 data pull, no real pipeline, fixed
    # provenance (the real one imports torch via cuda_graph_enabled).
    monkeypatch.setattr("src.tuning.tune_nn._ensure_data_from_s3", lambda: None)
    monkeypatch.setattr(ab_batch, "_provenance", lambda: {"git_sha": "abc"})


def test_cell_result_key_shape():
    assert (
        ab_batch.cell_result_key("ab_runs", "r1", "TE-baseline-42")
        == "ab_runs/r1/cells/TE-baseline-42.json"
    )
    # Stray slashes on the prefix don't double up in the key.
    assert ab_batch.cell_result_key("/ab_runs/", "r1", "k") == "ab_runs/r1/cells/k.json"


def test_run_batch_entry_runs_grid_and_uploads(batch_env, monkeypatch):
    """--only keeps the baseline, seeds come from FF_AB_SEEDS, and every cell's
    JSON (with elapsed + provenance stamped) lands at cell_result_key."""
    ran = []

    def _run(cell, variant, metric_fn, *, data_dir):
        ran.append(cell.key)
        return _ok_result(cell, variant, metric_fn, data_dir=data_dir)

    monkeypatch.setattr(ab_batch, "run_cell", _run)
    s3 = _fake_s3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    ab_batch.run_batch_entry("TE")

    # 2 variants (baseline + the --only one) x 2 seeds.
    assert ran == [
        "TE-baseline-42",
        "TE-baseline-123",
        "TE-nn_dropout=0-42",
        "TE-nn_dropout=0-123",
    ]
    assert s3.put_object.call_count == 4
    first = s3.put_object.call_args_list[0].kwargs
    assert first["Bucket"] == "test-bucket"
    assert first["Key"] == "ab_runs/test-run/cells/TE-baseline-42.json"
    body = json.loads(first["Body"])
    assert body["ok"] is True
    assert body["provenance"] == {"git_sha": "abc"}
    assert "elapsed_sec" in body


def test_run_batch_entry_resume_skips_completed_cells(batch_env, monkeypatch):
    """A Spot-retry attempt must not re-run cells whose JSON already landed."""
    ran = []

    def _run(cell, variant, metric_fn, *, data_dir):
        ran.append(cell.key)
        return _ok_result(cell, variant, metric_fn, data_dir=data_dir)

    monkeypatch.setattr(ab_batch, "run_cell", _run)
    s3 = _fake_s3(
        existing_keys=[
            "ab_runs/test-run/cells/TE-baseline-42.json",
            "ab_runs/test-run/cells/TE-nn_dropout=0-42.json",
        ]
    )
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    ab_batch.run_batch_entry("TE")

    assert ran == ["TE-baseline-123", "TE-nn_dropout=0-123"]
    assert s3.put_object.call_count == 2


def test_run_batch_entry_uploads_failure_and_exits_nonzero(batch_env, monkeypatch):
    """A failing cell is recorded to S3 like a success (so aggregation can
    surface it) and the job still exits non-zero at the end."""

    def _run(cell, variant, metric_fn, *, data_dir):
        if cell.seed == 123:
            raise RuntimeError("boom")
        return _ok_result(cell, variant, metric_fn, data_dir=data_dir)

    monkeypatch.setattr(ab_batch, "run_cell", _run)
    s3 = _fake_s3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    with pytest.raises(SystemExit, match="2 cell"):
        ab_batch.run_batch_entry("TE")

    # All four cells uploaded regardless — failures carry the error message.
    assert s3.put_object.call_count == 4
    bodies = [json.loads(c.kwargs["Body"]) for c in s3.put_object.call_args_list]
    failed = [b for b in bodies if not b["ok"]]
    assert len(failed) == 2
    assert all("boom" in b["error"] for b in failed)


def test_run_batch_entry_requires_env(monkeypatch):
    monkeypatch.delenv("FF_TUNE_AB_SPEC", raising=False)
    with pytest.raises(SystemExit, match="FF_TUNE_AB_SPEC"):
        ab_batch.run_batch_entry("TE")


def test_tune_nn_main_dispatches_on_ab_spec_env(monkeypatch):
    """The Batch route: FF_TUNE_AB_SPEC in the job env makes tune_nn.main
    divert into run_batch_entry before any Optuna work (the #1142 env-flag
    pattern; the container command is train.py's fixed --mode=tune argv)."""
    from src.tuning import tune_nn

    monkeypatch.setenv("FF_TUNE_AB_SPEC", SPEC)
    monkeypatch.delenv("FF_TUNE_ENSEMBLE_AB", raising=False)
    called = []
    monkeypatch.setattr("src.tuning.ab_batch.run_batch_entry", lambda pos: called.append(pos))
    monkeypatch.setattr(sys, "argv", ["tune_nn", "qb"])

    tune_nn.main()

    assert called == ["QB"]
