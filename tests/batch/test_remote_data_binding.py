"""Operator CLIs pair data with frozen remote image revisions before publishing."""

import hashlib
import io
import json
from unittest import mock

import pytest
from botocore.exceptions import ClientError

from src.batch import launch
from src.data import release
from src.scripts import wait_data_release
from src.scripts.resolve_training_image import resolve_batch, resolve_definition

pytestmark = pytest.mark.unit
SHA_A, SHA_B = "a" * 40, "b" * 40
RECIPE_A, RECIPE_B = {"src/config.py": "a" * 64}, {"src/config.py": "b" * 64}


def definition(revision, sha):
    return {
        "jobDefinitionName": "gpu",
        "revision": revision,
        "status": "ACTIVE",
        "containerProperties": {"image": f"registry/training:{sha}"},
    }


class Batch:
    def __init__(self):
        self.values = [definition(7, SHA_A)]
        self.submitted = []
        self.registered_sources = []

    def get_paginator(self, _):
        return self

    def paginate(self, **kwargs):
        return [{"jobDefinitions": list(self.values)}]

    def describe_job_definitions(self, *, jobDefinitions):
        selected = jobDefinitions[0]
        return {"jobDefinitions": [d for d in self.values if selected == f"gpu:{d['revision']}"]}

    def submit_job(self, **kwargs):
        self.submitted.append(kwargs)
        return {"jobId": "verified-job"}


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if Key in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        return {}

    def published(self, recipe):
        info = {"sha256": "0" * 64, "bytes": 1}
        manifest = {
            "schema_version": 1,
            "producer": recipe,
            "files": {
                name: info
                for name in (
                    "raw/weekly.parquet",
                    "splits/train.parquet",
                    "splits/val.parquet",
                    "splits/test.parquet",
                )
            },
        }
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        selected = hashlib.sha256(body).hexdigest()
        self.objects[f"data/releases/{selected}/manifest.json"] = body
        pointer = json.dumps({"schema_version": 1, "release_id": selected}).encode()
        self.objects[f"data/by-producer/{release.producer_fingerprint(recipe)}/manifest.json"] = (
            pointer
        )
        self.objects["data/manifest.json"] = pointer
        return selected


@pytest.fixture
def remote(monkeypatch):
    batch, s3 = Batch(), S3()
    monkeypatch.setenv("FF_DATA_RELEASE", "")
    monkeypatch.setattr(launch, "JOB_DEFINITION", "gpu")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", None)
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", None)
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", None)
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", "")
    monkeypatch.setattr(
        launch.boto3, "client", lambda service, **kwargs: batch if service == "batch" else s3
    )
    monkeypatch.setattr(release, "data_producer_hashes", lambda root: RECIPE_B)
    monkeypatch.setattr(
        wait_data_release,
        "producer_hashes_at_revision",
        lambda sha: RECIPE_A if sha == SHA_A else RECIPE_B,
    )

    def register_source(_s3, _bucket, _prefix, source_sha):
        assert batch.submitted == [], "source registration must precede every submission"
        batch.registered_sources.append(source_sha)

    monkeypatch.setattr("src.shared.artifact_publication.register_source", register_source)
    return batch, s3


@pytest.mark.parametrize("entrypoint", ["launch", "benchmark"])
def test_local_b_remote_a_cannot_publish_or_submit(remote, monkeypatch, entrypoint, tmp_path):
    from src.batch import benchmark

    batch, _ = remote
    module = launch if entrypoint == "launch" else benchmark
    publish = mock.Mock(side_effect=AssertionError("incorrect publication"))
    monkeypatch.setattr(module, "upload_data", publish)
    monkeypatch.chdir(tmp_path)
    argv = [entrypoint, "--positions", "WR"]
    if entrypoint == "launch":
        argv.extend(["--wait", "false"])
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(
        release.DataReleaseError, match="Local data producer differs from selected image"
    ):
        module.main()
    publish.assert_not_called()
    assert batch.submitted == []
    assert batch.registered_sources == []


def test_skip_upload_selects_remote_a_data_and_freezes_revision_when_latest_moves(
    remote, monkeypatch
):
    batch, s3 = remote
    data_a = s3.published(RECIPE_A)
    s3.published(RECIPE_B)  # global current belongs to another source checkout

    def source(sha):
        assert sha == SHA_A
        batch.values.append(definition(8, SHA_B))
        return RECIPE_A

    monkeypatch.setattr(wait_data_release, "producer_hashes_at_revision", source)
    monkeypatch.setattr(
        "sys.argv", ["launch", "--positions", "WR", "--skip-upload", "--wait", "false"]
    )
    publish = mock.Mock(side_effect=AssertionError("skip-upload published"))
    monkeypatch.setattr(launch, "upload_data", publish)
    launch.main()
    publish.assert_not_called()
    assert len(batch.submitted) == 1
    call = batch.submitted[0]
    assert call["jobDefinition"] == "gpu:7"
    environment = {
        item["name"]: item["value"] for item in call["containerOverrides"]["environment"]
    }
    assert environment["FF_TRAIN_GIT_SHA"] == SHA_A
    assert environment["FF_DATA_RELEASE"] == data_a
    assert batch.registered_sources == [SHA_A]
    run_id = environment["FF_BENCHMARK_RUN_ID"]
    descriptor = json.loads(s3.objects[f"models/training_runs/{run_id}/run.json"])
    assert descriptor["git_sha"] == SHA_A
    assert descriptor["data_release"] == data_a


def test_explicit_incompatible_data_pin_is_rejected_before_submit(remote, monkeypatch):
    batch, s3 = remote
    data_b = s3.published(RECIPE_B)
    monkeypatch.setenv("FF_DATA_RELEASE", data_b)
    monkeypatch.setattr(
        "sys.argv", ["launch", "--positions", "WR", "--skip-upload", "--wait", "false"]
    )
    with pytest.raises(release.DataReleaseError, match="incompatible with selected image"):
        launch.main()
    assert batch.submitted == []


@pytest.mark.parametrize("stale_global", ["", SHA_B])
def test_bound_source_drives_preflight_registration_submission_and_history(
    remote, monkeypatch, stale_global
):
    from src.batch import benchmark

    batch, s3 = remote
    data_a = s3.published(RECIPE_A)
    binding = launch.resolve_launch_binding(batch, s3, ["WR"])
    # The validated snapshot stays authoritative if module globals are absent
    # or change after resolution. Do not rewrite process-wide source selectors.
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", stale_global)
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", "99")
    monkeypatch.setattr(launch, "resolve_launch_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(launch, "wait_for_jobs", lambda *a, **kw: {"WR": ("SUCCEEDED", 0)})
    monkeypatch.setattr(launch, "download_artifacts", lambda *a, **kw: None)
    history = mock.Mock()
    monkeypatch.setattr(benchmark, "record_benchmark_run", history)
    monkeypatch.setattr("sys.argv", ["launch", "--positions", "WR", "--skip-upload"])
    launch.main()

    assert batch.registered_sources == [SHA_A]
    assert len(batch.submitted) == 1
    call = batch.submitted[0]
    assert call["jobDefinition"] == "gpu:7"
    environment = {e["name"]: e["value"] for e in call["containerOverrides"]["environment"]}
    assert environment["FF_TRAIN_GIT_SHA"] == SHA_A
    assert environment["FF_DATA_RELEASE"] == data_a
    history.assert_called_once_with(
        ["WR"],
        backend="batch",
        note="Standalone Batch run",
        git_hash=SHA_A,
        run_id=environment["FF_BENCHMARK_RUN_ID"],
        data_release=data_a,
    )
    assert stale_global == launch.TRAIN_GIT_SHA
    assert launch.JOB_DEFINITION_REVISION == "99"


def test_bound_revision_preflight_remains_mandatory(remote, monkeypatch):
    batch, _s3 = remote
    binding = {"image_sha": SHA_A, "gpu_definition": "gpu", "cpu_definition": ""}
    monkeypatch.setattr(launch, "resolve_launch_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(
        "sys.argv", ["launch", "--positions", "WR", "--skip-upload", "--wait", "false"]
    )
    with pytest.raises(SystemExit) as result:
        launch.main()
    assert result.value.code == 2
    assert batch.registered_sources == []
    assert batch.submitted == []


def test_source_ancestry_failure_stops_before_submission(remote, monkeypatch):
    batch, _s3 = remote

    def reject(*args, **kwargs):
        raise RuntimeError("selected source is not on origin/main")

    monkeypatch.setattr("src.shared.artifact_publication.register_source", reject)
    monkeypatch.setattr(
        "sys.argv", ["launch", "--positions", "WR", "--skip-upload", "--wait", "false"]
    )
    with pytest.raises(RuntimeError, match="not on origin/main"):
        launch.main()
    assert batch.submitted == []


def test_benchmark_registers_and_records_the_resolved_image(remote, monkeypatch, tmp_path):
    from src.batch import benchmark

    batch, s3 = remote
    s3.published(RECIPE_A)
    monkeypatch.setattr(release, "data_producer_hashes", lambda root: RECIPE_A)
    monkeypatch.chdir(tmp_path)

    def publish(_bucket):
        assert batch.registered_sources == [SHA_A]
        assert batch.submitted == []

    monkeypatch.setattr(benchmark, "upload_data", publish)
    monkeypatch.setattr(benchmark, "wait_for_jobs", lambda *a, **kw: {"WR": ("SUCCEEDED", 0)})
    history = mock.Mock()
    monkeypatch.setattr(benchmark, "record_benchmark_run", history)
    monkeypatch.setattr("sys.argv", ["benchmark", "--positions", "WR"])
    benchmark.main()
    assert batch.registered_sources == [SHA_A]
    assert batch.submitted[0]["jobDefinition"] == "gpu:7"
    assert history.call_args.kwargs["git_hash"] == SHA_A


def test_active_benchmark_rejects_a_conflicting_history_sha(remote, monkeypatch, tmp_path):
    from src.batch import benchmark

    batch, _s3 = remote
    publish = mock.Mock()
    monkeypatch.setattr(benchmark, "upload_data", publish)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["benchmark", "--positions", "WR", "--git-hash", SHA_B])
    with pytest.raises(SystemExit) as result:
        benchmark.main()
    assert result.value.code == 2
    publish.assert_not_called()
    assert batch.registered_sources == []
    assert batch.submitted == []


def test_download_only_does_not_register_or_resolve_a_training_source(
    remote, monkeypatch, tmp_path
):
    from src.batch import benchmark

    batch, _s3 = remote
    resolver = mock.Mock(side_effect=AssertionError("download-only resolved training source"))
    monkeypatch.setattr(benchmark, "resolve_launch_binding", resolver)
    history = mock.Mock()
    monkeypatch.setattr(benchmark, "record_benchmark_run", history)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["benchmark", "--download-only", "--git-hash", SHA_B])
    benchmark.main()
    resolver.assert_not_called()
    assert history.call_args.kwargs["git_hash"] == SHA_B
    assert batch.registered_sources == []
    assert batch.submitted == []


def test_explicit_workflow_revision_does_not_resolve_newer_latest():
    batch = Batch()
    batch.values.append(definition(8, SHA_B))
    s3 = mock.Mock()
    s3.get_object.side_effect = AssertionError("already pinned revision was re-resolved")
    selected = resolve_batch(batch, s3, "bucket", name="gpu", sha=SHA_A, revision="7")
    assert selected["revision"] == "7" and selected["image_sha"] == SHA_A
    assert resolve_definition(batch, "gpu:7") == {"image_sha": SHA_A, "job_definition": "gpu:7"}


def test_cpu_only_launch_uses_cpu_revision_map_for_explicit_source(monkeypatch):
    cpu = {**definition(4, SHA_A), "jobDefinitionName": "cpu"}
    batch = mock.Mock()
    batch.describe_job_definitions.return_value = {"jobDefinitions": [cpu]}
    s3 = mock.Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"4")}
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu")
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", "cpu-queue")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU_REVISION", None)
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", SHA_A)
    result = launch.resolve_launch_binding(batch, s3, ["K", "DST"])
    assert result["cpu_definition"] == "cpu:4" and result["gpu_definition"] == ""
    assert s3.get_object.call_args.kwargs["Key"] == f"job-def-revisions/cpu/{SHA_A}.txt"
