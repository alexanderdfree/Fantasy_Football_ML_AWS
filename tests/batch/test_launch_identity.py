"""Publishing operators bind submissions and collection to the same request."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from src.artifacts import intent, model_sync, receipts
from src.batch import benchmark, launch
from tests.orchestration.test_datasets import MemoryS3
from tests.shared._helpers import make_tarball

pytestmark = pytest.mark.unit
SOURCE = "b" * 40
RUN_ID = "operator:request-17"


@pytest.fixture(autouse=True)
def no_inherited_identity(monkeypatch):
    for name in (
        "FF_BUILD_PLAN_ID",
        "FF_DATASET_ID",
        "FF_DATA_RELEASE",
        "FF_DATA_FORMAT",
        "FF_REQUIRE_BUILD_PLAN",
        "FF_LEGACY_RUN_ID",
        "FF_TRAIN_GIT_SHA",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("entrypoint", [launch, benchmark])
@pytest.mark.parametrize("source,run_id", [("", ""), (SOURCE, ""), ("abc1234", RUN_ID)])
def test_unbound_training_refuses_before_effects(monkeypatch, entrypoint, source, run_id):
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", source)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", run_id)
    monkeypatch.setattr("sys.argv", ["operator", "--positions", "QB"])
    effects = Mock(side_effect=AssertionError("Unbound request produced an effect"))
    for name in ("upload_data", "submit_job", "wait_for_jobs"):
        monkeypatch.setattr(launch if name == "upload_data" else entrypoint, name, effects)
    monkeypatch.setattr(entrypoint.boto3, "client", effects)
    if entrypoint is benchmark:
        monkeypatch.setattr(entrypoint, "record_benchmark_run", effects)
    with pytest.raises(SystemExit) as error:
        entrypoint.main()
    assert error.value.code == 2
    effects.assert_not_called()


def test_submit_forwards_literal_request_and_source(monkeypatch):
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", SOURCE)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", RUN_ID)
    image = "registry/train@sha256:" + "c" * 64
    monkeypatch.setenv("FF_TRAIN_IMAGE_ID", image)
    batch = Mock()
    batch.submit_job.return_value = {"jobId": "job-17"}
    monkeypatch.setenv("FF_DATA_RELEASE", "legacy")
    launch.submit_job(
        "QB",
        batch_client=batch,
        binding={
            "image_sha": SOURCE,
            "gpu_definition": "ff-training-job:1",
            "cpu_definition": "",
            "gpu_image": image,
        },
    )
    env = {
        item["name"]: item["value"]
        for item in batch.submit_job.call_args.kwargs["containerOverrides"]["environment"]
    }
    assert env["FF_TRAIN_GIT_SHA"] == SOURCE
    assert env["FF_LEGACY_RUN_ID"] == RUN_ID
    assert env["FF_TRAIN_IMAGE_ID"] == image


@pytest.mark.parametrize("entrypoint", [launch, benchmark])
def test_plan_training_requires_source_before_effects(monkeypatch, entrypoint):
    monkeypatch.setenv("FF_BUILD_PLAN_ID", "a" * 64)
    monkeypatch.setattr("sys.argv", ["operator", "--positions", "QB"])
    effects = Mock(side_effect=AssertionError("Unidentified source produced an effect"))
    monkeypatch.setattr(launch, "upload_data", effects)
    monkeypatch.setattr(entrypoint.boto3, "client", effects)
    if entrypoint is benchmark:
        monkeypatch.setattr(entrypoint, "record_benchmark_run", effects)
    with pytest.raises(SystemExit) as error:
        entrypoint.main()
    assert error.value.code == 2
    effects.assert_not_called()


def _own_output():
    s3 = MemoryS3()
    reserved = intent.reserve_intent(s3, "bucket", "models", "QB", SOURCE, None, RUN_ID)
    metrics = {
        "position": "QB",
        "git_sha": SOURCE,
        "publication_intent": reserved,
        "publication_revision": None,
        "test_marker": "own-request",
    }
    data = make_tarball(
        {"benchmark_metrics.json": json.dumps(metrics).encode(), "own-model": b"own"}
    )
    entry = {
        **metrics,
        "key": model_sync.history_prefix("models", "QB") + "own/model.tar.gz",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "smoke_passed": True,
    }
    s3.objects[entry["key"]] = data
    receipts.claim_successful_output(s3, "bucket", "models", "QB", entry)
    s3.objects[model_sync.manifest_key("models", "QB")] = json.dumps(
        {"stable": {"key": "newer-unrelated-run"}}
    ).encode()
    return s3, metrics


def test_newer_global_head_cannot_replace_own_download_or_benchmark(monkeypatch, tmp_path):
    s3, metrics = _own_output()
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", SOURCE)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", RUN_ID)
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(launch, "S3_BUCKET", "bucket")
    monkeypatch.setattr(benchmark, "S3_BUCKET", "bucket")
    monkeypatch.setattr(benchmark.boto3, "client", lambda *_, **__: s3)
    monkeypatch.setattr(benchmark, "load_manifest", lambda *_: pytest.fail("Mutable head read"))
    monkeypatch.chdir(tmp_path)
    launch.download_artifacts(["QB"], s3_client=s3)
    assert (tmp_path / "qb/outputs/models/own-model").read_bytes() == b"own"
    collected = []

    def record(positions, **kwargs):
        collected.append((benchmark.download_metrics(positions), kwargs))

    monkeypatch.setattr(benchmark, "record_benchmark_run", record)
    launch._append_benchmark_history(["QB"], note="request test")
    assert collected == [
        (
            {"QB": metrics},
            {
                "backend": "batch",
                "note": "request test",
                "git_hash": SOURCE,
                "run_id": None,
                "data_release": None,
            },
        )
    ]
    del s3.objects[receipts.run_receipt_key("models", SOURCE, "QB", RUN_ID)]
    with pytest.raises(Exception, match="NoSuchKey"):
        launch.download_artifacts(["QB"], s3_client=s3)


def test_explicit_download_only_retains_latest_retrieval(monkeypatch):
    recorded = Mock()
    monkeypatch.setattr(benchmark, "record_benchmark_run", recorded)
    monkeypatch.setattr(benchmark, "submit_job", lambda *_: pytest.fail("Unexpected submission"))
    monkeypatch.setattr("sys.argv", ["benchmark", "--positions", "QB", "--download-only"])
    benchmark.main()
    assert recorded.call_args.args == (["QB"],)
    assert recorded.call_args.kwargs["git_hash"] is None


def test_training_benchmark_records_requested_source_not_operator_checkout(monkeypatch):
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", SOURCE)
    monkeypatch.setenv("FF_LEGACY_RUN_ID", RUN_ID)
    from src.batch import run_history

    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", "1")
    monkeypatch.setattr(launch, "register_submission_source", lambda *_: None)
    monkeypatch.setattr(run_history, "create_run", lambda *_, **__: "test-history")
    monkeypatch.setattr(benchmark.boto3, "client", lambda *_, **__: Mock())
    from tests.batch._submission_fixture import mock_submission_boundaries

    mock_submission_boundaries(monkeypatch, launch)
    monkeypatch.setattr(launch, "upload_data", lambda *_, **__: None)
    monkeypatch.setattr(benchmark, "submit_job", lambda pos, _, **kw: (pos, "job"))
    monkeypatch.setattr(benchmark, "wait_for_jobs", lambda _: {"QB": ("SUCCEEDED", 0)})
    recorded = Mock()
    monkeypatch.setattr(benchmark, "record_benchmark_run", recorded)
    monkeypatch.setattr("sys.argv", ["benchmark", "--positions", "QB"])
    benchmark.main()
    assert recorded.call_args.kwargs["git_hash"] == SOURCE
