"""Unit tests for src/tuning/ablate_batch.py (Batch container entry for the
eager ablation_runner family).

No AWS, no pipeline runs — boto3 and ablation_runner._run_job are mocked; we
assert the checkpoint/resume contract (which cells run, what lands at which S3
key), the module-contract validation, and the tune_nn env-flag dispatch route.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from src.tuning import ablate_batch
from src.tuning.ablation_runner import AblationJob, AblationResult

pytestmark = pytest.mark.unit

MOD = "src.tuning.ablate_fake"


def _noop_run_fn(job):  # never called (we mock _run_job), but AblationJob needs one
    return {}


def _fake_module():
    """A minimal Batch-runnable ablation module satisfying the contract."""

    def _build_jobs(*, position, seeds, variants):
        return [
            AblationJob(
                position=position,
                seed=seed,
                variant=v,
                label=v,
                run_fn=_noop_run_fn,
                base_cfg={},
                metadata={"run_kind": "experiment"},
            )
            for v in variants
            for seed in seeds
        ]

    return types.SimpleNamespace(
        ABLATION_NAME="fake",
        BASELINE="baseline",
        VARIANTS={"baseline": ("base", {}), "selfattn": ("self-attn", {"attn_self_layers": 1})},
        _build_jobs=_build_jobs,
        print_summary=lambda *a, **k: True,
    )


def _fake_s3(existing_keys=()):
    s3 = MagicMock()
    page = {"Contents": [{"Key": k} for k in existing_keys]} if existing_keys else {}
    s3.get_paginator.return_value.paginate.return_value = [page]
    return s3


def _ok_result(job):
    return AblationResult(
        position=job.position,
        seed=job.seed,
        variant=job.variant,
        metrics={"attn_fp_mae": 4.0, "ridge_fp_mae": 5.0},
        timings={},
        metadata={},
        error=None,
    )


@pytest.fixture
def batch_env(monkeypatch):
    monkeypatch.setenv("FF_TUNE_ABLATE_MOD", MOD)
    monkeypatch.setenv("FF_ABLATE_RUN_ID", "test-run")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_ABLATE_SEEDS", "42,123")
    monkeypatch.setenv("FF_ABLATE_VARIANTS", "selfattn")
    monkeypatch.delenv("FF_ABLATE_S3_PREFIX", raising=False)
    # Hermetic: no S3 data pull, no real pipeline, fixed provenance, fake module.
    monkeypatch.setattr("src.tuning.tune_nn._ensure_data_from_s3", lambda: None)
    monkeypatch.setattr(ablate_batch, "_provenance", lambda: {"git_sha": "abc"})
    monkeypatch.setattr(ablate_batch, "load_ablation_module", lambda dotted: _fake_module())


def test_cell_key_and_result_key_shape():
    assert ablate_batch.cell_key("RB", "selfattn", 42) == "RB-selfattn-42"
    assert (
        ablate_batch.cell_result_key("ablation_runs", "r1", "RB-selfattn-42")
        == "ablation_runs/r1/cells/RB-selfattn-42.json"
    )
    # Stray slashes on the prefix don't double up.
    assert (
        ablate_batch.cell_result_key("/ablation_runs/", "r1", "k")
        == "ablation_runs/r1/cells/k.json"
    )


def test_resolve_grid_always_includes_baseline():
    module = _fake_module()
    jobs = ablate_batch.resolve_grid(module, position="RB", seeds=[42], variants=["selfattn"])
    assert {j.variant for j in jobs} == {"baseline", "selfattn"}


def test_load_ablation_module_rejects_incomplete_contract(monkeypatch):
    bad = types.SimpleNamespace(VARIANTS={}, BASELINE="baseline")  # no _build_jobs
    monkeypatch.setattr("importlib.import_module", lambda d: bad)
    with pytest.raises(SystemExit, match="_build_jobs"):
        ablate_batch.load_ablation_module("src.tuning.whatever")


def test_run_batch_entry_runs_grid_and_uploads(batch_env, monkeypatch):
    """--only keeps the baseline, seeds come from FF_ABLATE_SEEDS, and every
    cell's JSON (with elapsed + provenance stamped) lands at cell_result_key."""
    ran = []

    def _run(job, log_path, data_dir):
        ran.append(ablate_batch.cell_key(job.position, job.variant, job.seed))
        return _ok_result(job)

    monkeypatch.setattr(ablate_batch, "_run_job", _run)
    s3 = _fake_s3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    ablate_batch.run_batch_entry("RB")

    # 2 variants (baseline + selfattn) x 2 seeds.
    assert set(ran) == {"RB-baseline-42", "RB-baseline-123", "RB-selfattn-42", "RB-selfattn-123"}
    assert s3.put_object.call_count == 4
    keys = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
    assert "ablation_runs/test-run/cells/RB-baseline-42.json" in keys
    body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
    assert body["provenance"] == {"git_sha": "abc"}
    assert "elapsed_sec" in body and body["error"] is None


def test_run_batch_entry_resume_skips_completed_cells(batch_env, monkeypatch):
    """A Spot-retry attempt must not re-run cells whose JSON already landed."""
    ran = []

    def _run(job, log_path, data_dir):
        ran.append(ablate_batch.cell_key(job.position, job.variant, job.seed))
        return _ok_result(job)

    monkeypatch.setattr(ablate_batch, "_run_job", _run)
    s3 = _fake_s3(
        existing_keys=[
            "ablation_runs/test-run/cells/RB-baseline-42.json",
            "ablation_runs/test-run/cells/RB-selfattn-42.json",
        ]
    )
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    ablate_batch.run_batch_entry("RB")

    assert set(ran) == {"RB-baseline-123", "RB-selfattn-123"}
    assert s3.put_object.call_count == 2


def test_run_batch_entry_uploads_failure_and_exits_nonzero(batch_env, monkeypatch):
    """A failing cell (_run_job captures it into AblationResult.error) is
    uploaded like a success, and the job still exits non-zero at the end."""

    def _run(job, log_path, data_dir):
        if job.seed == 123:
            return AblationResult(
                position=job.position,
                seed=job.seed,
                variant=job.variant,
                metrics={},
                timings={},
                metadata={},
                error="RuntimeError: boom",
            )
        return _ok_result(job)

    monkeypatch.setattr(ablate_batch, "_run_job", _run)
    s3 = _fake_s3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    with pytest.raises(SystemExit, match="2 cell"):
        ablate_batch.run_batch_entry("RB")

    assert s3.put_object.call_count == 4
    bodies = [json.loads(c.kwargs["Body"]) for c in s3.put_object.call_args_list]
    failed = [b for b in bodies if b["error"]]
    assert len(failed) == 2 and all("boom" in b["error"] for b in failed)


def test_run_batch_entry_requires_env(monkeypatch):
    monkeypatch.delenv("FF_TUNE_ABLATE_MOD", raising=False)
    with pytest.raises(SystemExit, match="FF_TUNE_ABLATE_MOD"):
        ablate_batch.run_batch_entry("RB")


def test_tune_nn_main_dispatches_on_ablate_mod_env(monkeypatch):
    """FF_TUNE_ABLATE_MOD in the job env makes tune_nn.main divert into
    run_batch_entry before any Optuna work (the env-flag pattern)."""
    from src.tuning import tune_nn

    monkeypatch.delenv("FF_TUNE_ENSEMBLE_AB", raising=False)
    monkeypatch.delenv("FF_TUNE_ENSEMBLE_COMPARE", raising=False)
    monkeypatch.delenv("FF_TUNE_AB_SPEC", raising=False)
    monkeypatch.setenv("FF_TUNE_ABLATE_MOD", MOD)
    called = []
    monkeypatch.setattr("src.tuning.ablate_batch.run_batch_entry", lambda pos: called.append(pos))
    monkeypatch.setattr(sys, "argv", ["tune_nn", "rb"])

    tune_nn.main()

    assert called == ["RB"]
