"""A raw-data refresh must preserve its deployed image and producer contract."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.artifacts import serving_snapshot
from src.data import release
from src.scripts import advance_data_release as rollout
from tests.artifacts.test_deployment_rollout import RolloutAWS, task_definition
from tests.test_data_release import FakeS3, producer  # noqa: F401 — fixture reuse

pytestmark = pytest.mark.unit


class Ecs(RolloutAWS):
    def __init__(self, release_id, recipe):
        super().__init__(response_code=200)
        self.calls = []
        self.race = False
        self.task = task_definition()
        self.task["taskDefinitionArn"] = self.current
        self.task["revision"] = 1
        env = self.task["containerDefinitions"][0]["environment"]
        env[:] = [
            {"name": "FF_MODEL_S3_BUCKET", "value": "bucket"},
            {"name": "FF_MODEL_S3_PREFIX", "value": "models"},
            {"name": "FF_ALLOW_RUNTIME_INFERENCE", "value": "0"},
            {
                "name": "FF_SERVING_SNAPSHOT_GENERATION",
                "value": publish_serving_snapshot(FakeS3(), release_id).rsplit("/", 1)[-1],
            },
            {"name": "FF_DATA_RELEASE", "value": release_id},
            {"name": "FF_DATA_PRODUCER_SHA256", "value": recipe},
            {"name": "OTHER_SETTING", "value": "keep"},
        ]

    def call(self, api, operation, **arguments):
        self.calls.append((operation, copy.deepcopy(arguments)))
        if operation == "describe-task-definition":
            return {"taskDefinition": copy.deepcopy(self.task)}
        if operation == "register-task-definition":
            self.registered = copy.deepcopy(arguments)
            if self.race:
                self.current = "arn:concurrent-deploy"
        return super().call(api, operation, **arguments)


def publish_serving_snapshot(s3, release_id, *, schema=None):
    payloads = {name: b"{}" for name in serving_snapshot.FILES}
    manifest = {
        "schema_version": 1,
        "cache_schema_version": schema or serving_snapshot.CACHE_SCHEMA_VERSION,
        "dataset_id": release_id,
        "files": {
            name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            for name, body in payloads.items()
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    generation = hashlib.sha256(encoded).hexdigest()
    base = f"models/predictions_cache/generations/{generation}"
    s3.objects.update({f"{base}/{name}": value for name, value in payloads.items()})
    s3.objects[f"{base}/manifest.json"] = encoded
    s3.objects["models/predictions_cache/current.json"] = json.dumps(
        {"schema_version": 1, "generation": generation, "manifest": f"{base}/manifest.json"}
    ).encode()
    return base


@pytest.fixture(autouse=True)
def isolated_rollout_state(tmp_path, monkeypatch):
    monkeypatch.setattr(Ecs, "state_path", tmp_path / "rollout.json", raising=False)


@pytest.fixture
def snapshots(producer):
    s3 = FakeS3()
    first = release.publish_release(s3, "bucket", **producer)
    recipe = release.resolve_release(s3, "bucket", release_id=first)[1]["data_producer_sha256"]
    pd.DataFrame({"season": [2025], "value": [99]}).to_parquet(
        producer["raw_dir"] / "weekly.parquet"
    )
    release.seal_inputs(**producer)
    second = release.publish_release(s3, "bucket", **producer)
    publish_serving_snapshot(s3, first)
    publish_serving_snapshot(s3, second)
    return s3, first, second, recipe, producer


def advance(s3, ecs, selected):
    return rollout.advance_release(
        s3,
        ecs,
        bucket="bucket",
        cluster="cluster",
        service="service",
        release_id=selected,
        state_path=ecs.state_path,
        timeout=0,
    )


def test_refresh_advances_only_data_pin_preserving_deployed_image(snapshots):
    s3, first, second, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    assert advance(s3, ecs, second)["advanced"] is True
    assert ecs.registered["containerDefinitions"][0]["image"] == "registry/app:new-image"
    env = {
        item["name"]: item["value"]
        for item in ecs.registered["containerDefinitions"][0]["environment"]
    }
    assert env == {
        "FF_MODEL_S3_BUCKET": "bucket",
        "FF_MODEL_S3_PREFIX": "models",
        "FF_ALLOW_RUNTIME_INFERENCE": "0",
        "FF_DATA_RELEASE": second,
        "FF_DATASET_ID": second,
        "FF_SERVING_SNAPSHOT_GENERATION": publish_serving_snapshot(FakeS3(), second).rsplit("/", 1)[
            -1
        ],
        "FF_DATA_PRODUCER_SHA256": recipe,
        "OTHER_SETTING": "keep",
    }
    assert "taskDefinitionArn" not in ecs.registered
    assert "revision" not in ecs.registered
    assert ecs.current == "new-revision"


def test_different_producer_waits_for_matching_code_deployment(snapshots):
    s3, first, _, recipe, producer = snapshots
    config = producer["repo_root"] / "src/config.py"
    config.parent.mkdir()
    config.write_text("SEASONS = [2026]\n")
    release.seal_inputs(**producer)
    incompatible = release.publish_release(s3, "bucket", **producer)
    ecs = Ecs(first, recipe)
    assert advance(s3, ecs, incompatible)["advanced"] is False
    assert not any(name in {"register-task-definition", "update-service"} for name, _ in ecs.calls)


def test_initial_unpinned_task_is_left_to_automatic_code_deploy(snapshots):
    s3, first, second, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    ecs.task["containerDefinitions"][0]["environment"] = []
    result = advance(s3, ecs, second)
    assert result["advanced"] is False
    assert "first compatible code deployment" in result["reason"]
    assert not any(name == "update-service" for name, _ in ecs.calls)


def test_manual_concurrent_task_change_never_gets_overwritten(snapshots):
    s3, first, second, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    ecs.race = True
    with pytest.raises(RuntimeError, match="refusing stale rollout"):
        advance(s3, ecs, second)
    assert ecs.current == "arn:concurrent-deploy"
    assert not any(name == "update-service" for name, _ in ecs.calls)


def test_queued_older_refresh_cannot_replace_a_newer_snapshot(snapshots):
    s3, first, _, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    result = advance(s3, ecs, first)
    assert not result["advanced"] and "newer release" in result["reason"]
    assert ecs.calls == []


def test_deploy_pins_gate_outputs_and_refresh_has_a_service_scoped_lock():
    root = Path(__file__).resolve().parents[2] / ".github/workflows"
    deploy = yaml.safe_load((root / "deploy.yml").read_text())
    refresh = yaml.safe_load((root / "refresh-splits.yml").read_text())
    step = next(
        s
        for s in deploy["jobs"]["deploy"]["steps"]
        if s.get("name") == "Require a compatible published serving snapshot"
    )
    assert "--bind-data" in step["run"]
    assert "--task-definition" in step["run"]
    assert refresh["jobs"]["rollout"]["concurrency"]["group"] == deploy["concurrency"]["group"]
    assert "github.ref" not in deploy["concurrency"]["group"]
    assert refresh["concurrency"]["group"] != deploy["concurrency"]["group"]


def test_offline_builders_preserve_their_selected_snapshot_across_steps():
    root = Path(__file__).resolve().parents[2] / ".github/workflows"
    refresh = yaml.safe_load((root / "refresh-splits.yml").read_text())
    publish = next(s for s in refresh["jobs"]["refresh"]["steps"] if s.get("id") == "publish")
    assert (
        'os.environ["GITHUB_ENV"]' in publish["run"]
        and "FF_DATA_RELEASE={release_id}" in publish["run"]
    )
    upcoming = yaml.safe_load((root / "refresh-upcoming-week.yml").read_text())
    job = next(iter(upcoming["jobs"].values()))
    steps = job["steps"]
    gate = next(i for i, s in enumerate(steps) if "wait_data_release" in s.get("run", ""))
    hydration = next(i for i, s in enumerate(steps) if "sync_data_from_s3()" in s.get("run", ""))
    assert gate < hydration and "--pin-training" in steps[gate]["run"]
    assert job["timeout-minutes"] >= 85


@pytest.mark.parametrize("problem", ["missing", "wrong_release", "old_schema", "corrupt"])
def test_raw_data_cannot_advance_without_matching_verified_prediction_snapshot(snapshots, problem):
    s3, first, second, recipe, _ = snapshots
    if problem == "missing":
        del s3.objects["models/predictions_cache/current.json"]
    elif problem == "wrong_release":
        publish_serving_snapshot(s3, first)
    elif problem == "old_schema":
        publish_serving_snapshot(s3, second, schema=9)
    else:
        base = publish_serving_snapshot(s3, second)
        s3.objects[f"{base}/snapshot.json"] = b"tampered"
    ecs = Ecs(first, recipe)
    assert advance(s3, ecs, second)["advanced"] is False
    assert not any(name in {"register-task-definition", "update-service"} for name, _ in ecs.calls)


def test_data_advance_rolls_back_if_new_tasks_never_become_ready(snapshots):
    s3, first, second, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    ecs.code = 503
    with pytest.raises(RuntimeError, match="did not become ready"):
        advance(s3, ecs, second)
    assert ecs.current == "old-revision"
    assert ecs.health == ecs.original_health


@pytest.mark.parametrize("producer_matches", [True, False])
def test_deploy_gate_binds_only_verified_snapshot_data_authority(
    snapshots, tmp_path, monkeypatch, producer_matches
):
    s3, first, second, recipe, _ = snapshots
    task_path = tmp_path / "rendered-task.json"
    task_path.write_text(json.dumps(Ecs(first, recipe).task))
    previous = task_path.read_bytes()
    manifest = release.resolve_release(s3, "bucket", release_id=second)[1]
    monkeypatch.setattr(
        release,
        "data_producer_hashes",
        lambda root: manifest["producer"] if producer_matches else {"different": "recipe"},
    )

    def aws_get(command, **kwargs):
        key = command[command.index("--key") + 1]
        Path(command[-1]).write_bytes(s3.objects[key])
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", aws_get)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serving_snapshot",
            "wait",
            "--task-definition",
            str(task_path),
            "--bind-data",
            "--timeout",
            "0",
        ],
    )
    if not producer_matches:
        with pytest.raises(RuntimeError, match="No compatible"):
            serving_snapshot.main()
        assert task_path.read_bytes() == previous
        return
    serving_snapshot.main()
    task = json.loads(task_path.read_text())
    env = {item["name"]: item["value"] for item in task["containerDefinitions"][0]["environment"]}
    assert env["FF_DATA_RELEASE"] == env["FF_DATASET_ID"] == second
    assert env["FF_DATA_PRODUCER_SHA256"] == recipe
    pointer = json.loads(s3.objects["models/predictions_cache/current.json"])
    assert env["FF_SERVING_SNAPSHOT_GENERATION"] == pointer["generation"]
