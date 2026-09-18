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
# The resolver / ECR preflight accept only a full 40-char lowercase SHA.
NEW_SHA = "deadbeef" * 5


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
    assert launch_ab._swap_image_tag(
        TEMPLATE_IMAGE + "@sha256:" + "a" * 64, "newsha"
    ) == launch_ab._swap_image_tag(TEMPLATE_IMAGE, "newsha")


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

    resolved = launch_ab.resolve_job_definition(NEW_SHA, batch)

    assert resolved == f"{launch_ab.AB_JOB_DEFINITION}:7"
    kwargs = batch.register_job_definition.call_args.kwargs
    assert kwargs["jobDefinitionName"] == launch_ab.AB_JOB_DEFINITION
    assert kwargs["type"] == "container"
    container = kwargs["containerProperties"]
    assert container["image"].endswith(f"ff-training:{NEW_SHA}")
    # GPU requirement + env caps carried over from the production template.
    assert container["resourceRequirements"] == [{"type": "GPU", "value": "1"}]
    assert {"name": "OMP_NUM_THREADS", "value": "1"} in container["environment"]
    assert kwargs["retryStrategy"] == launch_ab.RETRY_STRATEGY
    assert kwargs["platformCapabilities"] == ["EC2"]
    assert "timeout" not in kwargs


def test_resolve_job_definition_reuses_matching_revision():
    batch = MagicMock()
    ab_def = _template_def(image=f"123.dkr.ecr.us-east-1.amazonaws.com/ff-training:{NEW_SHA}")
    ab_def["jobDefinitionName"] = launch_ab.AB_JOB_DEFINITION
    ab_def["revision"] = 3
    batch.describe_job_definitions.side_effect = [
        {"jobDefinitions": [_template_def()]},
        {"jobDefinitions": [ab_def]},
    ]

    resolved = launch_ab.resolve_job_definition(NEW_SHA, batch)

    assert resolved == f"{launch_ab.AB_JOB_DEFINITION}:3"
    batch.register_job_definition.assert_not_called()


@pytest.mark.parametrize("reuse", [False, True])
def test_resolve_job_definition_pins_exact_bytes_in_isolated_revision(reuse):
    digest = "sha256:" + "b" * 64
    image = launch_ab._swap_image_tag(TEMPLATE_IMAGE, "a" * 40) + "@" + digest
    template = _template_def()
    pinned = {**_template_def(image=image), "revision": 7}
    batch = MagicMock()
    batch.describe_job_definitions.side_effect = [
        {"jobDefinitions": [template]},
        {"jobDefinitions": [pinned] if reuse else []},
    ]
    batch.register_job_definition.return_value = {"revision": 7}
    assert (
        launch_ab.resolve_job_definition("a" * 40, batch, image_digest=digest)
        == f"{launch_ab.AB_JOB_DEFINITION}:7"
    )
    assert template == _template_def()
    if reuse:
        batch.register_job_definition.assert_not_called()
    else:
        registered = batch.register_job_definition.call_args.kwargs
        assert registered["jobDefinitionName"] == launch_ab.AB_JOB_DEFINITION
        assert registered["containerProperties"] == {
            **template["containerProperties"],
            "image": image,
        }


def test_resolve_job_definition_rejects_malformed_digest_before_registration():
    batch = MagicMock()
    batch.describe_job_definitions.return_value = {"jobDefinitions": [_template_def()]}
    with pytest.raises(ValueError, match="complete sha256"):
        launch_ab.resolve_job_definition("a" * 40, batch, image_digest="sha256:short")
    batch.register_job_definition.assert_not_called()


@pytest.mark.parametrize("bad_sha", ["abc1234", "A" * 40, "a" * 39, "a" * 41, ""])
def test_resolve_job_definition_rejects_non_full_lowercase_sha_before_any_batch_call(bad_sha):
    """A short / uppercase / empty tag would otherwise mint a real ff-ab-job
    revision pointing at a non-existent ff-training:<tag>; refuse before the
    template lookup, let alone registration."""
    batch = MagicMock()
    with pytest.raises(ValueError, match="full 40-char lowercase git SHA"):
        launch_ab.resolve_job_definition(bad_sha, batch)
    batch.describe_job_definitions.assert_not_called()
    batch.register_job_definition.assert_not_called()


@pytest.mark.parametrize("bad_sha", ["abc1234", "A" * 40])
def test_check_image_exists_rejects_non_full_lowercase_sha_before_any_ecr_call(bad_sha):
    ecr = MagicMock()
    with pytest.raises(ValueError, match="full 40-char lowercase git SHA"):
        launch_ab.check_image_exists(bad_sha, ecr)
    ecr.describe_images.assert_not_called()


@pytest.mark.parametrize("matches", [False, True])
def test_main_verifies_digest_before_mutations_and_records_pin(monkeypatch, matches):
    sha, digest = "a" * 40, "sha256:" + "b" * 64
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_ab",
            "--spec",
            SPEC,
            "--positions",
            "WR",
            "--seeds",
            "42",
            "--image-sha",
            sha,
            "--image-digest",
            digest,
            "--skip-image-check",
            "--wait",
            "false",
        ],
    )
    clients = {name: MagicMock() for name in ("ecr", "batch", "s3")}
    ecr = clients["ecr"]
    ecr.describe_images.return_value = {
        "imageDetails": [{"imageDigest": digest if matches else "sha256:" + "c" * 64}]
    }
    ecr.describe_repositories.return_value = {"repositories": [{"repositoryUri": "registry/repo"}]}
    monkeypatch.setattr("boto3.client", lambda name, **kw: clients[name])
    register = MagicMock(return_value="ff-ab-job:7")
    pin = MagicMock()
    submit = MagicMock(return_value=("WR", "job-1"))
    manifest = MagicMock()
    monkeypatch.setattr(launch_ab, "resolve_job_definition", register)
    monkeypatch.setattr(
        launch_ab,
        "resolve_definition",
        lambda *a: {"image_sha": sha, "job_definition": "ff-ab-job:7"},
    )
    monkeypatch.setattr(launch_ab, "pin_data_release", pin)
    monkeypatch.setattr(launch_ab, "submit_ab_job", submit)
    monkeypatch.setattr(launch_ab, "write_run_manifest", manifest)
    if not matches:
        with pytest.raises(RuntimeError, match="no longer matches"):
            launch_ab.main()
        for mutation in (register, pin, submit, manifest):
            mutation.assert_not_called()
        return
    launch_ab.main()
    register.assert_called_once_with(sha, clients["batch"], image_digest=digest)
    assert submit.call_args.kwargs["job_definition"] == "ff-ab-job:7"
    evidence = manifest.call_args.kwargs["manifest"]
    assert (evidence["image_sha"], evidence["image_digest"]) == (sha, digest)
    ecr.describe_images.assert_called_once_with(
        repositoryName="ff-training", imageIds=[{"imageTag": sha}]
    )


@pytest.mark.parametrize("data_prefix", [None, "data/validation/live-data-fix"])
def test_submit_ab_job_shape(data_prefix):
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
        **({"data_prefix": data_prefix} if data_prefix else {}),
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
    assert env["S3_DATA_PREFIX"] == (data_prefix or "data")


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
