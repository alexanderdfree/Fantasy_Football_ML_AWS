import json
import subprocess

import pytest

from src.orchestration import build_plan, datasets
from tests.orchestration.test_datasets import published  # noqa: F401

pytestmark = pytest.mark.unit
CODE_SHA = "d" * 40


def registered_source(s3):
    from src.artifacts.source import source_key

    s3.objects[source_key("models", CODE_SHA)] = json.dumps(
        {"source_sha": CODE_SHA, "source_order": 1, "lineage": [CODE_SHA]}
    ).encode()


class Batch:
    def get_paginator(self, name):
        assert name == "describe_job_definitions"
        return self

    def paginate(self, **kwargs):
        while True:
            page = self.describe_job_definitions(**kwargs)
            yield page
            if "nextToken" not in page:
                return
            kwargs["nextToken"] = page["nextToken"]

    def describe_job_definitions(self, **_):
        return {
            "jobDefinitions": [
                {
                    "revision": 7,
                    "status": "ACTIVE",
                    "jobDefinitionName": "train",
                    "jobDefinitionArn": "arn:aws:batch:definition/train:7",
                    "containerProperties": {"image": f"registry/train:{CODE_SHA}"},
                }
            ]
        }


def _plan(published, *, run_id=None):
    s3, source, dataset_id = published
    registered_source(s3)
    plan_id, plan = build_plan.create_plan(
        s3,
        Batch(),
        "bucket",
        dataset_id=dataset_id,
        source_id=source,
        code_sha=CODE_SHA,
        gpu_definition="train",
        positions=["QB"],
        seed=42,
        run_id=run_id,
    )
    return s3, plan_id, plan


def test_plan_round_trip_and_runtime_contract(published, monkeypatch):
    s3, plan_id, plan = _plan(published)
    monkeypatch.setenv("FF_BUILD_PLAN_ID", plan_id)
    monkeypatch.setenv("FF_DATASET_ID", plan["dataset_id"])
    monkeypatch.setenv("FF_DATA_RELEASE", plan["dataset_id"])
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", plan["git_sha"])
    monkeypatch.setenv("FF_TRAIN_IMAGE_ID", f"registry/train:{CODE_SHA}")
    assert (
        build_plan.verify_training_plan(s3, "bucket", position="QB", seed=42, branch="full") == plan
    )
    monkeypatch.setenv("FF_DATASET_ID", "f" * 64)
    with pytest.raises(datasets.DatasetError, match="dataset ID"):
        build_plan.verify_training_plan(s3, "bucket", position="QB", seed=42, branch="full")


def test_required_plan_absence_fails_before_training(monkeypatch):
    monkeypatch.setenv("FF_REQUIRE_BUILD_PLAN", "1")
    monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
    with pytest.raises(datasets.DatasetError, match="required"):
        build_plan.verify_training_plan(None, "bucket", position="QB", seed=42, branch="full")


@pytest.mark.parametrize("branch", ["full", "nn", "cpu", "merge"])
def test_launcher_pins_plan_definition_and_provenance(published, monkeypatch, branch):
    from unittest.mock import Mock

    from src.batch import launch

    s3, source, dataset_id = published
    registered_source(s3)
    plan_id, plan = build_plan.create_plan(
        s3,
        Batch(),
        "bucket",
        dataset_id=dataset_id,
        source_id=source,
        code_sha=CODE_SHA,
        gpu_definition="gpu",
        cpu_definition="cpu",
        positions=["QB"],
        seed=42,
    )
    monkeypatch.setenv("FF_BUILD_PLAN_ID", plan_id)
    monkeypatch.setenv("FF_DATASET_ID", dataset_id)
    monkeypatch.setenv("FF_DATA_RELEASE", dataset_id)
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", CODE_SHA)
    monkeypatch.setattr(launch, "S3_BUCKET", "bucket")
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu")
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", "cpu-queue")
    monkeypatch.setattr(launch.boto3, "client", lambda *_, **__: s3)
    batch = Mock()
    batch.submit_job.return_value = {"jobId": "job"}
    launch.submit_job("QB", 42, batch, branch=branch, split_run_id="run")
    call = batch.submit_job.call_args.kwargs
    assert call["jobDefinition"] == plan["job_definitions"]["gpu"]["arn"]
    environment = {
        item["name"]: item["value"] for item in call["containerOverrides"]["environment"]
    }
    assert environment["FF_BUILD_PLAN_ID"] == plan_id
    assert environment["FF_DATASET_ID"] == dataset_id
    assert environment["FF_DATA_RELEASE"] == dataset_id
    assert environment["FF_DATA_FORMAT"] == "data-release-v1"
    assert environment["FF_REQUIRE_BUILD_PLAN"] == "1"


def test_tampered_plan_is_rejected(published):
    s3, plan_id, plan = _plan(published)
    plan["seed"] = 123
    s3.objects[f"build-plans/{plan_id}.json"] = json.dumps(plan).encode()
    with pytest.raises(datasets.DatasetError, match="identity"):
        build_plan.load_plan(s3, "bucket", plan_id)


def test_plan_retry_preserves_original_intent_and_rollback_revision(published):
    from src.artifacts.intent import validate_intent
    from src.artifacts.model_sync import manifest_key

    s3, old_id, old = _plan(published, run_id="old")
    _, _, newer = _plan(published, run_id="newer")
    assert old["intents"]["QB"]["sequence"] < newer["intents"]["QB"]["sequence"]
    assert not validate_intent(s3, "bucket", "models", old["intents"]["QB"])
    # A manual rollback updates the manifest while an existing plan is retried.
    s3.objects[manifest_key("models", "QB")] = (
        b'{"revision":"rollback","rollback_epoch":"manual-a"}'
    )
    _, retry_id, retry = _plan(published, run_id="old")
    assert retry_id == old_id
    assert retry["publication_revisions"] == {"QB": None}
    _, _, fresh = _plan(published, run_id="after-rollback")
    assert fresh["publication_revisions"] == {"QB": "manual-a"}
    assert validate_intent(s3, "bucket", "models", fresh["intents"]["QB"])
    s3.objects[manifest_key("models", "QB")] = (
        b'{"revision":"gc-completed","rollback_epoch":"manual-a"}'
    )
    _, _, after_gc = _plan(published, run_id="after-gc")
    assert after_gc["publication_revisions"] == fresh["publication_revisions"]


def test_unregistered_source_cannot_reserve_or_publish_a_plan(published):
    s3, source, dataset_id = published
    with pytest.raises(RuntimeError, match="register the built image SHA"):
        build_plan.create_plan(
            s3,
            Batch(),
            "bucket",
            dataset_id=dataset_id,
            source_id=source,
            code_sha=CODE_SHA,
            gpu_definition="train",
            positions=["QB"],
            seed=42,
        )
    assert not any("intents/" in key or key.startswith("build-plans/") for key in s3.objects)


def test_plan_captures_predecessor_rollback_before_first_new_publication(published):
    from src.artifacts.model_sync import manifest_key, previous_protocol_manifest_key

    s3 = published[0]
    previous_key = previous_protocol_manifest_key("models", "QB")
    previous = {
        "schema_version": 3,
        "promotion_mode": "rollback",
        "publication_source": {"source_sha": CODE_SHA, "source_order": 1},
        "stable": None,
        "history": [],
    }
    encoded = json.dumps(previous).encode()
    s3.objects[previous_key] = encoded
    _, plan_id, plan = _plan(published, run_id="predecessor-rollback")
    fence = plan["publication_revisions"]["QB"]
    assert fence.startswith("previous-protocol:")
    assert plan["intents"]["QB"]["publication_revision"] == fence
    assert s3.objects[previous_key] == encoded
    assert manifest_key("models", "QB") not in s3.objects
    # Retrying the same identified plan cannot recapture a changed operator fence.
    s3.objects[previous_key] = json.dumps({**previous, "operator_note": "second rollback"}).encode()
    _, retry_id, retry = _plan(published, run_id="predecessor-rollback")
    assert retry_id == plan_id
    assert retry["publication_revisions"]["QB"] == fence


def test_manual_dispatch_uses_actual_image_revision_despite_newer_checkout(tmp_path, monkeypatch):
    import boto3

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Plan test")
    git("config", "user.email", "plan@example.test")
    git("config", "commit.gpgsign", "false")
    config = tmp_path / "src/config.py"
    config.parent.mkdir()
    config.write_text("SEASONS = [2025]\n")
    git("add", "src")
    git("commit", "-qm", "image source")
    image_sha = git("rev-parse", "HEAD")
    source_id = datasets.source_identity(tmp_path, image_sha)
    config.write_text("SEASONS = [2026]\n")
    git("commit", "-qam", "newer runner checkout")
    git("update-ref", "refs/remotes/origin/main", git("rev-parse", "HEAD"))
    assert git("rev-parse", "HEAD") != image_sha
    from tests.orchestration.test_datasets import MemoryS3

    s3 = MemoryS3()
    from src.scripts.wait_data_release import producer_hashes_at_revision
    from tests.orchestration.test_datasets import write_release

    dataset_id = write_release(s3, producer_hashes_at_revision(image_sha, repo_root=tmp_path))

    class ImageBatch(Batch):
        calls = []

        def describe_job_definitions(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "jobDefinitions": [
                    {
                        "revision": 7,
                        "status": "ACTIVE",
                        "jobDefinitionName": "train",
                        "jobDefinitionArn": "arn:aws:batch:definition/train:7",
                        "containerProperties": {"image": f"registry/train:{image_sha}"},
                    }
                ]
            }

    batch = ImageBatch()
    monkeypatch.setattr(boto3, "client", lambda service: s3 if service == "s3" else batch)
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    output = tmp_path / "selected_plan.json"
    build_plan.main(
        [
            "--bucket",
            "bucket",
            "--repo",
            str(tmp_path),
            "--gpu-definition",
            "train",
            "--positions",
            "QB",
            "--seed",
            "42",
            "--run-id",
            "manual-42",
            "--timeout",
            "0",
            "--output",
            str(output),
        ]
    )
    plan = json.loads(output.read_text())
    assert plan["git_sha"] == image_sha
    assert plan["source_id"] == source_id
    assert plan["dataset_id"] == dataset_id
    assert batch.calls[0]["jobDefinitionName"] == "train"
    assert batch.calls[1]["jobDefinitions"] == ["arn:aws:batch:definition/train:7"]
    assert "status" not in batch.calls[1]
    assert f"git_sha={image_sha}\n" in github_output.read_text()


def test_manual_dispatch_does_not_guess_code_from_latest_tag():
    class UntaggedBatch(Batch):
        def describe_job_definitions(self, **_):
            return {
                "jobDefinitions": [
                    {
                        "revision": 7,
                        "status": "ACTIVE",
                        "jobDefinitionName": "train",
                        "jobDefinitionArn": "arn:aws:batch:definition/train:7",
                        "containerProperties": {"image": "registry/train:latest"},
                    }
                ]
            }

    with pytest.raises(ValueError, match="full source-SHA tag"):
        build_plan.resolve_execution_source(UntaggedBatch(), "", "train", None)


def test_bare_definition_resolution_checks_every_page():
    class PagedBatch(Batch):
        def describe_job_definitions(self, **kwargs):
            revision = 7 if kwargs.get("nextToken") == "next" else 3
            response = {
                "jobDefinitions": [
                    {
                        "revision": revision,
                        "jobDefinitionArn": f"arn:train:{revision}",
                        "containerProperties": {"image": f"registry/train:{CODE_SHA}"},
                    }
                ]
            }
            if revision == 3:
                response["nextToken"] = "next"
            return response

    assert build_plan.resolve_job_definition(PagedBatch(), "train")["arn"] == "arn:train:7"


def test_workflow_requires_selection_before_submission():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/train-batch.yml").read_text())
    steps = workflow["jobs"]["train"]["steps"]
    selection = next(index for index, step in enumerate(steps) if step.get("id") == "build_plan")
    submission = next(index for index, step in enumerate(steps) if step.get("id") == "train")
    assert selection < submission
    receipt = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Verify exact published build artifacts"
    )
    cache = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Build and publish serving cache from verified artifacts"
    )
    assert submission < receipt < cache
    assert not any(step.get("name") == "Verify model artifact freshness" for step in steps)
    assert "src.artifacts.receipts" in steps[receipt]["run"]
    assert "src.orchestration.build_plan" in steps[selection]["run"]
    assert "if" not in steps[selection]  # manual dispatch is gated too
    assert steps[submission]["env"]["FF_REQUIRE_BUILD_PLAN"] == "1"
    assert "build_plan.outputs.dataset_id" in steps[submission]["env"]["FF_DATASET_ID"]
    assert "build_plan.outputs.git_sha" in steps[submission]["env"]["FF_TRAIN_GIT_SHA"]
    assert "github.sha" not in steps[selection]["env"]["CODE_SHA"]
