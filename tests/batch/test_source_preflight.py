"""Source identity must be verified before training or job allocation."""

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from botocore.exceptions import ClientError

from src.batch import launch, train
from src.shared import artifact_publication
from src.shared.registry import ALL_POSITIONS

pytestmark = pytest.mark.unit
SHA = "a" * 40


class ComputeReached(Exception):
    pass


def train_argv(monkeypatch, pos, branch):
    args = ["train", "--position", pos, "--branch", branch]
    if branch != "full":
        args += ["--split-run-id", "preflight"]
    monkeypatch.setattr("sys.argv", args)


@pytest.mark.parametrize("pos", ALL_POSITIONS)
@pytest.mark.parametrize("branch", ["full", "nn", "cpu", "merge"])
@pytest.mark.parametrize("failure", ["missing_sha", "image_mismatch", "missing_registration"])
def test_training_rejects_source_before_compute(monkeypatch, pos, branch, failure):
    train_argv(monkeypatch, pos, branch)
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", "" if failure == "missing_sha" else SHA)
    monkeypatch.setattr(
        artifact_publication,
        "image_source_sha",
        lambda: "b" * 40 if failure == "image_mismatch" else SHA,
    )
    s3 = Mock()
    s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    client = Mock(return_value=s3)
    monkeypatch.setattr(train.boto3, "client", client)
    blocked = []
    for name in (
        "_assert_gpu",
        "seed_everything",
        "sync_raw_data",
        "get_runner",
        "_merge_split_artifacts",
    ):
        stub = Mock(side_effect=AssertionError(f"Unexpected compute: {name}"))
        monkeypatch.setattr(train, name, stub)
        blocked.append(stub)
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2
    for stub in blocked:
        stub.assert_not_called()
    if failure == "missing_sha":
        client.assert_not_called()


@pytest.mark.parametrize("pos", ALL_POSITIONS)
@pytest.mark.parametrize("branch", ["full", "nn", "cpu", "merge"])
def test_verified_image_reaches_existing_compute_path(monkeypatch, pos, branch):
    train_argv(monkeypatch, pos, branch)
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", SHA)
    monkeypatch.setattr(artifact_publication, "image_source_sha", lambda: SHA)
    s3 = Mock()
    record = {"source_sha": SHA, "source_order": 1, "lineage": [SHA]}
    s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(record).encode())}
    monkeypatch.setattr(train.boto3, "client", lambda *_: s3)
    monkeypatch.setattr(train, "_assert_gpu", lambda *_a, **_k: None)
    monkeypatch.setattr(train, "seed_everything", Mock(side_effect=ComputeReached))
    with pytest.raises(ComputeReached):
        train.main()
    s3.get_object.assert_called_once()


@pytest.mark.parametrize("sha,revision", [(None, "12"), ("short", "12"), (SHA, None)])
def test_launcher_missing_source_or_revision_fails_before_aws(monkeypatch, sha, revision):
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", sha)
    monkeypatch.setattr(launch, "JOB_DEFINITION", "ff-training-job")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", revision)
    client = Mock(side_effect=AssertionError("AWS before validation"))
    monkeypatch.setattr(launch.boto3, "client", client)
    monkeypatch.setattr("sys.argv", ["launch", "--positions", "QB", "--skip-upload"])
    with pytest.raises(SystemExit) as exc:
        launch.main()
    assert exc.value.code == 2
    client.assert_not_called()


def test_split_requires_cpu_revision(monkeypatch):
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", SHA)
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", "12")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu-def")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU_REVISION", None)
    with pytest.raises(RuntimeError, match="CPU_REVISION"):
        launch.validate_submission_source(ALL_POSITIONS, split=True)
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU_REVISION", "13")
    launch.validate_submission_source(ALL_POSITIONS, split=True)


@pytest.mark.parametrize("workflow", ["train-batch.yml", "train-ec2.yml"])
def test_manual_training_requires_built_image_sha_and_automatic_uses_upstream(workflow):
    root = Path(__file__).resolve().parents[2]
    doc = yaml.safe_load((root / ".github/workflows" / workflow).read_text())
    trigger = doc.get("on", doc.get(True))
    assert trigger["workflow_dispatch"]["inputs"]["image_sha"]["required"] is True
    steps = doc["jobs"]["train"]["steps"]
    step = next(s for s in steps if "FF_TRAIN_GIT_SHA" in s.get("env", {}))
    assert (
        step["env"]["FF_TRAIN_GIT_SHA"]
        == "${{ github.event.workflow_run.head_sha || github.event.inputs.image_sha }}"
    )
    if workflow == "train-batch.yml":
        revision = next(s for s in steps if s.get("id") == "revision")
        assert "workflow_run" not in str(revision.get("if", ""))
        assert "image_sha" in revision["env"]["HEAD_SHA"]
        assert "falling back to bare" not in revision["run"]


@pytest.mark.parametrize("sha,revision", [(None, "12"), (SHA, None)])
def test_standalone_benchmark_rejects_missing_pins_before_aws(monkeypatch, sha, revision):
    from src.batch import benchmark

    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", sha)
    monkeypatch.setattr(launch, "JOB_DEFINITION", "ff-training-job")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", revision)
    client = Mock(side_effect=AssertionError("AWS before validation"))
    monkeypatch.setattr(benchmark.boto3, "client", client)
    monkeypatch.setattr("sys.argv", ["benchmark", "--positions", "QB"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    client.assert_not_called()
