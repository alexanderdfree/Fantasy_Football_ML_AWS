"""Rollout gates wait for real compatible data, never mutate service state."""

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src.data.release import (
    DATA_PRODUCER_PATHS,
    SPLIT_NAMES,
    data_producer_hashes,
    producer_fingerprint,
)
from src.scripts import wait_data_release as gate

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


class S3:
    def __init__(self):
        self.objects = {}
        self.reads = []

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def publish(self, producer):
        names = [*(f"splits/{name}" for name in SPLIT_NAMES), "raw/weekly.parquet"]
        manifest = {
            "schema_version": 1,
            "producer": producer,
            "files": {name: {"sha256": "0" * 64, "bytes": 1} for name in names},
        }
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        release = hashlib.sha256(body).hexdigest()
        self.objects[f"data/releases/{release}/manifest.json"] = body
        self.objects["data/manifest.json"] = json.dumps(
            {"schema_version": 1, "release_id": release}
        ).encode()
        self.objects[f"data/by-producer/{producer_fingerprint(producer)}/manifest.json"] = (
            self.objects["data/manifest.json"]
        )
        return release


@pytest.fixture
def producers(tmp_path, monkeypatch):
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/config.py").write_text("SEASONS = [2025]\n")
    return tmp_path


def test_unrelated_docs_deploy_reuses_compatible_snapshot(producers):
    s3 = S3()
    expected = s3.publish(data_producer_hashes(producers))
    (producers / "README.md").write_text("New docs, identical data recipe")

    def no_wait(seconds):
        raise AssertionError("unrelated deploy should not wait")

    assert (
        gate.wait_for_release(s3, "bucket", revision="docs", repo_root=producers, sleep=no_wait)
        == expected
    )


def test_initial_rollout_waits_even_before_pending_marker_exists(producers):
    s3 = S3()
    waits = []
    now = [0]

    def publish_after_wait(seconds):
        waits.append(seconds)
        now[0] += seconds
        s3.publish(data_producer_hashes(producers))

    selected = gate.wait_for_release(
        s3,
        "bucket",
        revision="initial",
        repo_root=producers,
        timeout=60,
        interval=10,
        clock=lambda: now[0],
        sleep=publish_after_wait,
    )
    assert len(selected) == 64 and waits == [10]


def test_missing_pending_cannot_bypass_incompatible_snapshot(producers):
    s3 = S3()
    s3.publish({"src/config.py": "old"})
    now = [0]

    def tick(seconds):
        now[0] += seconds

    with pytest.raises(TimeoutError, match="Existing service was not changed"):
        gate.wait_for_release(
            s3,
            "bucket",
            revision="change",
            repo_root=producers,
            timeout=20,
            interval=10,
            clock=lambda: now[0],
            sleep=tick,
        )
    assert now[0] == 20


def test_failed_rebuild_aborts_without_waiting(producers):
    s3 = S3()
    for kind in ("pending", "failed"):
        s3.objects[f"splits-rebuild-markers/{kind}/change.txt"] = b"run-12"
    with pytest.raises(RuntimeError, match="rebuild run-12 failed.*refusing rollout"):
        gate.wait_for_release(
            s3,
            "bucket",
            revision="change",
            repo_root=producers,
            sleep=lambda seconds: pytest.fail("must fail immediately"),
        )


def test_retry_is_not_cancelled_by_previous_failure(producers):
    s3 = S3()
    s3.objects["splits-rebuild-markers/pending/change.txt"] = b"retry"
    s3.objects["splits-rebuild-markers/failed/change.txt"] = b"older"

    def publish(seconds):
        s3.publish(data_producer_hashes(producers))

    assert gate.wait_for_release(
        s3, "bucket", revision="change", repo_root=producers, sleep=publish
    )


def test_image_revision_hashes_are_independent_of_advanced_checkout(producers, monkeypatch):
    monkeypatch.chdir(producers)

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    git("init", "-q")
    git("add", "src/config.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "image")
    image = git("rev-parse", "HEAD")
    expected = gate.producer_hashes_at_revision(image)
    (producers / "src/config.py").write_text("SEASONS = [2026]\n")
    assert gate.producer_hashes_at_revision(image) == expected
    assert data_producer_hashes(producers) != expected


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_deploy_gate_precedes_all_running_service_changes():
    steps = workflow("deploy.yml")["jobs"]["deploy"]["steps"]
    gate_index = next(
        i
        for i, step in enumerate(steps)
        if "src.artifacts.serving_snapshot wait" in step.get("run", "")
    )
    gate = steps[gate_index]
    assert "--bind-data" in gate["run"] and "--task-definition" in gate["run"]
    changes = [
        i
        for i, step in enumerate(steps)
        if "src.artifacts.deployment deploy" in step.get("run", "")
        or "aws elbv2 modify" in step.get("run", "")
        or "amazon-ecs-deploy-task-definition" in step.get("uses", "")
    ]
    assert changes and all(i > gate_index for i in changes)
    assert gate.get("continue-on-error") is not True
    assert "--timeout 2400" in gate["run"]


@pytest.mark.parametrize("name", ["train-batch.yml", "train-ec2.yml"])
def test_training_gates_all_event_types_and_pins_before_submitting(name):
    steps = workflow(name)["jobs"]["train"]["steps"]
    command = "src.orchestration.build_plan" if name == "train-batch.yml" else "wait_data_release"
    gate_index = next(i for i, step in enumerate(steps) if command in step.get("run", ""))
    gate = steps[gate_index]
    assert "if" not in gate and gate.get("continue-on-error") is not True
    submit_index = next(i for i, step in enumerate(steps) if step.get("id") == "train")
    assert gate_index < submit_index
    if name == "train-batch.yml":
        assert "--timeout 3600" in gate["run"]
        env = steps[submit_index]["env"]
        assert env["FF_DATA_RELEASE"] == env["FF_DATASET_ID"]
        assert env["FF_DATA_FORMAT"] == "data-release-v1"
    else:
        assert "--pin-training" in gate["run"]
    assert "proceeding with current S3 splits" not in json.dumps(steps)


def test_refresh_covers_every_data_producer_and_publishes_failure_marker():
    document = workflow("refresh-splits.yml")
    triggers = document.get("on", document.get(True))
    paths = triggers["push"]["paths"]
    expected = {f"{name}/**" if (ROOT / name).is_dir() else name for name in DATA_PRODUCER_PATHS}
    assert set(paths) == expected
    marker = next(
        step
        for step in document["jobs"]["refresh"]["steps"]
        if "splits-rebuild-markers/failed/" in step.get("run", "")
    )
    assert marker["if"] == "failure()"


@pytest.mark.parametrize("name", ["train-batch.yml", "train-ec2.yml"])
def test_training_rollout_uses_verified_release_and_shared_deploy_lock(name):
    jobs = workflow(name)["jobs"]
    assert jobs["train"]["outputs"]["data_release"]
    build = next(
        step
        for step in jobs["train"]["steps"]
        if "src.prediction.build_snapshot" in step.get("run", "")
    )
    assert build.get("continue-on-error") is not True
    rollout = jobs["ecs_rollout"]
    assert rollout["needs"] == "train"
    assert rollout["concurrency"]["group"] == "deploy-fantasy-service"
    advance = next(
        step
        for step in rollout["steps"]
        if "src.scripts.advance_data_release" in step.get("run", "")
    )
    assert advance["env"]["DATA_RELEASE"] == "${{ needs.train.outputs.data_release }}"
    assert '--release-id "$DATA_RELEASE"' in advance["run"]
    assert "--state data-release-rollout.json" in advance["run"]
    restore = next(
        step
        for step in rollout["steps"]
        if "src.artifacts.deployment restore" in step.get("run", "")
    )
    assert "failure()" in restore["if"] and "cancelled()" in restore["if"]
    assert "--state data-release-rollout.json" in restore["run"]
    assert "aws ecs update-service" not in json.dumps(jobs["train"]["steps"])
