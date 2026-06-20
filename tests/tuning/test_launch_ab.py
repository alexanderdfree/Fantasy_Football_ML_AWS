"""Unit tests for src/tuning/launch_ab.py.

Mirrors test_launch_tune.py: no AWS — boto3 clients are mocked and we assert
the *shape* of the submission (command, env, job definition cloning) plus the
collect/aggregate contract over per-cell S3 JSONs.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from src.tuning import launch_ab
from src.tuning.ab_harness import resolve_spec

pytestmark = pytest.mark.unit

SPEC = "src.tuning.ab_example"
TEMPLATE_IMAGE = "123.dkr.ecr.us-east-1.amazonaws.com/ff-training:oldsha"


def _template_def(image=TEMPLATE_IMAGE):
    return {
        "jobDefinitionName": "ff-training-job",
        "revision": 41,
        "type": "container",
        "containerProperties": {
            "image": image,
            "vcpus": 4,
            "memory": 15000,
            "resourceRequirements": [{"type": "GPU", "value": "1"}],
            "environment": [{"name": "OMP_NUM_THREADS", "value": "1"}],
        },
        "platformCapabilities": ["EC2"],
    }


def test_swap_image_tag():
    assert (
        launch_ab._swap_image_tag(TEMPLATE_IMAGE, "newsha")
        == "123.dkr.ecr.us-east-1.amazonaws.com/ff-training:newsha"
    )
    with pytest.raises(ValueError):
        launch_ab._swap_image_tag("no-tag-image", "x")


def test_resolve_job_definition_registers_clone():
    """No matching ff-ab-job revision -> clone the production GPU definition
    with only the image swapped; the baked training timeout is NOT carried
    (A/B jobs set theirs at submit time)."""
    batch = MagicMock()
    batch.describe_job_definitions.side_effect = [
        {"jobDefinitions": [_template_def()]},  # template lookup
        {"jobDefinitions": []},  # no existing ff-ab-job
    ]
    batch.register_job_definition.return_value = {"revision": 7}

    resolved = launch_ab.resolve_job_definition("newsha", batch)

    assert resolved == f"{launch_ab.AB_JOB_DEFINITION}:7"
    kwargs = batch.register_job_definition.call_args.kwargs
    assert kwargs["jobDefinitionName"] == launch_ab.AB_JOB_DEFINITION
    assert kwargs["type"] == "container"
    container = kwargs["containerProperties"]
    assert container["image"].endswith("ff-training:newsha")
    # GPU requirement + env caps carried over from the production template.
    assert container["resourceRequirements"] == [{"type": "GPU", "value": "1"}]
    assert {"name": "OMP_NUM_THREADS", "value": "1"} in container["environment"]
    assert kwargs["retryStrategy"] == launch_ab.RETRY_STRATEGY
    assert kwargs["platformCapabilities"] == ["EC2"]
    assert "timeout" not in kwargs


def test_resolve_job_definition_reuses_matching_revision():
    batch = MagicMock()
    ab_def = _template_def(image="123.dkr.ecr.us-east-1.amazonaws.com/ff-training:newsha")
    ab_def["jobDefinitionName"] = launch_ab.AB_JOB_DEFINITION
    ab_def["revision"] = 3
    batch.describe_job_definitions.side_effect = [
        {"jobDefinitions": [_template_def()]},
        {"jobDefinitions": [ab_def]},
    ]

    resolved = launch_ab.resolve_job_definition("newsha", batch)

    assert resolved == f"{launch_ab.AB_JOB_DEFINITION}:3"
    batch.register_job_definition.assert_not_called()


def test_submit_ab_job_shape():
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-1"}

    pos, job_id = launch_ab.submit_ab_job(
        "RB",
        spec_dotted=SPEC,
        run_id="run-1",
        s3_prefix="ab_runs",
        job_definition="ff-ab-job:7",
        image_sha="abc1234",
        seeds=[42, 123],
        only=["nn_dropout=0"],
        cuda_graph="false",
        attempt_timeout=3600,
        batch_client=batch,
    )

    assert (pos, job_id) == ("RB", "job-1")
    kwargs = batch.submit_job.call_args.kwargs
    assert kwargs["jobName"].startswith("ff-ab-rb-")
    assert kwargs["jobDefinition"] == "ff-ab-job:7"
    assert kwargs["retryStrategy"] == launch_ab.RETRY_STRATEGY
    assert kwargs["timeout"] == {"attemptDurationSeconds": 3600}
    overrides = kwargs["containerOverrides"]
    # train.py's --mode=tune path; the env flag does the actual routing.
    assert overrides["command"] == ["--position", "RB", "--mode", "tune"]
    env = {e["name"]: e["value"] for e in overrides["environment"]}
    assert env["FF_TUNE_AB_SPEC"] == SPEC
    assert env["FF_AB_RUN_ID"] == "run-1"
    assert env["FF_AB_S3_PREFIX"] == "ab_runs"
    assert env["FF_AB_SEEDS"] == "42,123"
    assert env["FF_AB_ONLY"] == "nn_dropout=0"
    assert env["FF_DEVICE"] == "cuda"
    assert env["FF_TRAIN_GIT_SHA"] == "abc1234"
    assert env["FF_CUDA_GRAPH"] == "0"
    # S3 data bootstrap for _ensure_data_from_s3 inside the container.
    assert env["S3_BUCKET"] == launch_ab.S3_BUCKET
    assert env["S3_DATA_PREFIX"] == "data"


def test_submit_ab_job_auto_graph_forwards_nothing():
    """cuda_graph=auto must NOT set FF_CUDA_GRAPH — the container's sm_80+
    autodetect (the production graphed metric path) stays in charge."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-1"}
    launch_ab.submit_ab_job(
        "QB",
        spec_dotted=SPEC,
        run_id="r",
        s3_prefix="ab_runs",
        job_definition="ff-ab-job:7",
        image_sha="abc",
        seeds=None,
        only=None,
        batch_client=batch,
    )
    env_names = {
        e["name"] for e in batch.submit_job.call_args.kwargs["containerOverrides"]["environment"]
    }
    assert "FF_CUDA_GRAPH" not in env_names
    assert "FF_AB_SEEDS" not in env_names
    assert "FF_AB_ONLY" not in env_names
    assert "FF_FEATURE_CACHE_DISABLE" not in env_names


def test_collect_results_synthesizes_missing_cells():
    """A cell whose JSON never landed becomes a not-ok row (so aggregate
    surfaces the gap) instead of silently shrinking the grid."""
    spec = resolve_spec(SPEC, positions=["RB"], seeds=[42], only=["nn_dropout=0"])
    ok_row = {
        "position": "RB",
        "variant": "baseline",
        "seed": 42,
        "label": "baseline",
        "ok": True,
        "metrics": {"Ridge": {"mae": 1.0}},
        "ridge_mae": 1.0,
        "error": None,
    }

    s3 = MagicMock()

    def _get(Bucket, Key):
        if "baseline" in Key:
            body = MagicMock()
            body.read.return_value = json.dumps(ok_row).encode()
            return {"Body": body}
        raise RuntimeError("NoSuchKey")

    s3.get_object.side_effect = _get

    results = launch_ab.collect_results(
        spec, bucket="b", s3_prefix="ab_runs", run_id="r", s3_client=s3
    )

    assert len(results) == 2  # baseline + nn_dropout=0
    by_variant = {r["variant"]: r for r in results}
    assert by_variant["baseline"]["ok"] is True
    missing = by_variant["nn_dropout=0"]
    assert missing["ok"] is False
    assert "ab_runs/r/cells/RB-nn_dropout=0-42.json" in missing["error"]


def test_collect_results_parallel_preserves_order():
    """The parallel collector returns rows in build_cells order and is identical to
    the serial (max_workers=1) path — aggregate keys by cell fields, but order-stable
    output keeps the contract simple."""
    from src.tuning.ab_harness import build_cells

    spec = resolve_spec(SPEC, positions=["RB", "WR"], seeds=[42, 7])
    cells = build_cells(spec)
    assert len(cells) > 4  # a real grid, so the pool actually parallelizes

    def _get(Bucket, Key):
        for c in cells:
            if Key.endswith(f"/{c.key}.json"):
                body = MagicMock()
                body.read.return_value = json.dumps(
                    {
                        "position": c.position,
                        "variant": c.variant,
                        "seed": c.seed,
                        "ok": True,
                        "metrics": {},
                        "ridge_mae": None,
                    }
                ).encode()
                return {"Body": body}
        raise RuntimeError("NoSuchKey")

    s3 = MagicMock()
    s3.get_object.side_effect = _get

    parallel = launch_ab.collect_results(
        spec, bucket="b", s3_prefix="ab_runs", run_id="r", s3_client=s3, max_workers=8
    )
    serial = launch_ab.collect_results(
        spec, bucket="b", s3_prefix="ab_runs", run_id="r", s3_client=s3, max_workers=1
    )
    expected = [(c.position, c.variant, c.seed) for c in cells]
    assert [(r["position"], r["variant"], r["seed"]) for r in parallel] == expected
    assert parallel == serial


def test_main_max_cells_guard(monkeypatch):
    """The cost guard refuses an oversized grid before any AWS call."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_ab", "--spec", SPEC, "--image-sha", "abc", "--max-cells", "2", "--dry-run"],
    )
    with pytest.raises(SystemExit, match="max-cells"):
        launch_ab.main()


def test_main_dry_run_prints_plan(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_ab",
            "--spec",
            SPEC,
            "--image-sha",
            "abc1234",
            "--positions",
            "RB",
            "--seeds",
            "42",
            "--dry-run",
        ],
    )
    launch_ab.main()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert SPEC in out
    assert "ff-training:abc1234" in out
    # 3 ab_example variants x 1 seed x 1 position.
    assert "cells:         3" in out


def test_default_run_id_shape():
    rid = launch_ab._default_run_id("src.tuning.ab_example", "abcdef0123456789")
    assert rid.startswith("ab_example-")
    assert rid.endswith("-abcdef0")


def test_submit_ab_job_stacked_env():
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-2"}
    launch_ab.submit_ab_job(
        "RB",
        spec_dotted=SPEC,
        run_id="run-2",
        s3_prefix="ab_runs",
        job_definition="ff-ab-job:7",
        image_sha="abc1234",
        seeds=[42, 123],
        only=None,
        stacked=True,
        stacked_epochs=12,
        batch_client=batch,
    )
    env = {
        e["name"]: e["value"]
        for e in batch.submit_job.call_args.kwargs["containerOverrides"]["environment"]
    }
    assert env["FF_AB_STACKED"] == "1"
    assert env["FF_AB_STACKED_EPOCHS"] == "12"
