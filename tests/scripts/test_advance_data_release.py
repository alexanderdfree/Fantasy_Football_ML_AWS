"""A raw-data refresh must preserve its deployed image and producer contract."""

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import release
from src.scripts import advance_data_release as rollout
from tests.test_data_release import FakeS3, producer  # noqa: F401 — fixture reuse

pytestmark = pytest.mark.unit


class Ecs:
    def __init__(self, release_id, recipe):
        self.current = "arn:old"
        self.calls = []
        self.race = False
        self.task = {
            "taskDefinitionArn": self.current,
            "revision": 1,
            "status": "ACTIVE",
            "family": "fantasy-predictor",
            "cpu": "1024",
            "memory": "8192",
            "containerDefinitions": [
                {
                    "name": "fantasy-predictor",
                    "image": "registry/app:sha-A",
                    "environment": [
                        {"name": "FF_DATA_RELEASE", "value": release_id},
                        {"name": "FF_DATA_PRODUCER_SHA256", "value": recipe},
                        {"name": "OTHER_SETTING", "value": "keep"},
                    ],
                }
            ],
        }

    def call(self, operation, arguments):
        self.calls.append((operation, copy.deepcopy(arguments)))
        if operation == "describe-services":
            return {"services": [{"taskDefinition": self.current}]}
        if operation == "describe-task-definition":
            return {"taskDefinition": copy.deepcopy(self.task)}
        if operation == "register-task-definition":
            self.registered = copy.deepcopy(arguments)
            if self.race:
                self.current = "arn:concurrent-deploy"
            return {"taskDefinition": {"taskDefinitionArn": "arn:new"}}
        if operation == "update-service":
            self.current = arguments["taskDefinition"]
            return {}
        raise AssertionError(operation)


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
    return s3, first, second, recipe, producer


def advance(s3, ecs, selected):
    return rollout.advance_release(
        s3, ecs, bucket="bucket", cluster="cluster", service="service", release_id=selected
    )


def test_refresh_advances_only_data_pin_preserving_deployed_image(snapshots):
    s3, first, second, recipe, _ = snapshots
    ecs = Ecs(first, recipe)
    assert advance(s3, ecs, second)["advanced"] is True
    assert ecs.registered["containerDefinitions"][0]["image"] == "registry/app:sha-A"
    env = {
        item["name"]: item["value"]
        for item in ecs.registered["containerDefinitions"][0]["environment"]
    }
    assert env == {
        "FF_DATA_RELEASE": second,
        "FF_DATA_PRODUCER_SHA256": recipe,
        "OTHER_SETTING": "keep",
    }
    assert "taskDefinitionArn" not in ecs.registered
    assert "revision" not in ecs.registered
    assert ecs.current == "arn:new"


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
        if s.get("name") == "Pull current task definition"
    )
    assert "steps.data-release.outputs.release_id" in step["env"]["DATA_RELEASE"]
    assert "steps.data-release.outputs.producer_sha256" in step["env"]["DATA_PRODUCER"]
    assert "FF_DATA_RELEASE" in step["run"] and "FF_DATA_PRODUCER_SHA256" in step["run"]
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
