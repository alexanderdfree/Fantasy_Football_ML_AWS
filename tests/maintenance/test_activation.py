"""Exercise staged data, real manifest validation, and ambiguous activation recovery."""

import copy
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.artifacts import serving_snapshot
from src.data import release
from src.maintenance import control, storage, worker
from tests.maintenance.test_contracts import S3, Dynamo, models
from tests.test_data_release import producer  # noqa: F401

pytestmark = pytest.mark.unit
SHA = "a" * 40


class Ecs:
    def __init__(self, release_id, recipe, generation):
        self.current = "arn:old"
        self.fail = None
        self.registrations = 0
        self.health = {"HealthCheckPath": "/ready", "Matcher": {"HttpCode": "200"}}
        self.attributes = [{"Key": "deregistration_delay.timeout_seconds", "Value": "300"}]
        self.definitions = {
            self.current: {
                "taskDefinitionArn": self.current,
                "family": "fantasy-predictor",
                "containerDefinitions": [
                    {
                        "name": "fantasy-predictor",
                        "image": "registry/app:" + SHA,
                        "healthCheck": {"command": ["CMD-SHELL", "curl -f http://localhost/ready"]},
                        "environment": [
                            {"name": "FF_DATA_RELEASE", "value": release_id},
                            {"name": "FF_DATASET_ID", "value": release_id},
                            {"name": "FF_DATA_PRODUCER_SHA256", "value": recipe},
                            {"name": "FF_SERVING_SNAPSHOT_GENERATION", "value": generation},
                            {"name": "FF_MODEL_S3_BUCKET", "value": "bucket"},
                        ],
                    }
                ],
            }
        }

    def describe_services(self, **kwargs):
        return {
            "services": [
                {
                    "taskDefinition": self.current,
                    "runningCount": 2,
                    "desiredCount": 2,
                    "pendingCount": 0,
                    "loadBalancers": [
                        {"containerName": "fantasy-predictor", "targetGroupArn": "target"}
                    ],
                    "deployments": [
                        {
                            "status": "PRIMARY",
                            "rolloutState": "COMPLETED",
                            "taskDefinition": self.current,
                        }
                    ],
                }
            ]
        }

    def describe_task_definition(self, *, taskDefinition):
        return {"taskDefinition": copy.deepcopy(self.definitions[taskDefinition])}

    def register_task_definition(self, **definition):
        self.registrations += 1
        arn = f"arn:new-{self.registrations}"
        self.definitions[arn] = dict(definition, taskDefinitionArn=arn)
        return {"taskDefinition": self.definitions[arn]}

    def update_service(self, *, taskDefinition, **kwargs):
        if self.fail == "before":
            self.fail = None
            raise OSError("API unavailable")
        self.current = taskDefinition
        if self.fail == "after":
            self.fail = None
            raise OSError("response lost after acceptance")
        return {}

    def list_tasks(self, **kwargs):
        return {"taskArns": ["task-1", "task-2"]}

    def describe_tasks(self, **kwargs):
        container = self.definitions[self.current]["containerDefinitions"][0]
        return {
            "tasks": [
                {
                    "taskDefinitionArn": self.current,
                    "lastStatus": "RUNNING",
                    "containers": [
                        {
                            "name": "fantasy-predictor",
                            "image": container["image"],
                            "healthStatus": "HEALTHY",
                        }
                    ],
                }
                for _ in range(2)
            ]
        }

    def describe_target_groups(self, **kwargs):
        return {"TargetGroups": [{"TargetGroupArn": "target", **copy.deepcopy(self.health)}]}

    def describe_target_group_attributes(self, **kwargs):
        return {"Attributes": copy.deepcopy(self.attributes)}

    def modify_target_group(self, *, TargetGroupArn, **kwargs):
        self.health.update(copy.deepcopy(kwargs))
        return {}

    def modify_target_group_attributes(self, *, Attributes, **kwargs):
        self.attributes = copy.deepcopy(Attributes)
        return {}


def cache_files(label):
    return {
        "predictions.parquet": label,
        "metrics.json": b"{}",
        "fingerprint.json": json.dumps(
            {"schema_version": serving_snapshot.CACHE_SCHEMA_VERSION}
        ).encode(),
        "snapshot.json": b"{}",
    }


def staged(s3, request, data_release):
    build = serving_snapshot.SnapshotBuild(
        request["cache_pointer"]["etag"],
        tuple((p, pin["etag"], pin["artifact"]["key"]) for p, pin in request["models"].items()),
        None,
        data_release,
    )
    return {
        "build": json.loads(json.dumps(asdict(build))),
        "files": {
            name: worker._record(
                s3, "bucket", request["run_id"], name, body, "application/octet-stream"
            )
            for name, body in cache_files(b"new-cache").items()
        },
    }


@pytest.fixture
def system(producer, tmp_path):
    s3 = S3()
    old = release.publish_release(s3, "bucket", **producer)
    _, manifest = release.resolve_release(s3, "bucket", release_id=old)
    pins = models(s3)
    for name, body in cache_files(b"old-cache").items():
        (tmp_path / name).write_bytes(body)
    token = serving_snapshot.SnapshotBuild(
        None, tuple((p, x["etag"], x["artifact"]["key"]) for p, x in pins.items()), None, old
    )
    pointer = serving_snapshot.publish(s3, "bucket", tmp_path, token)
    ecs = Ecs(old, manifest["data_producer_sha256"], pointer["generation"])
    settings = {
        "bucket": "bucket",
        "table": "table",
        "cluster": "cluster",
        "service": "service",
        "url": "https://example.test",
        "mode": "active",
        "elbv2": ecs,
    }
    metadata = {"source_sha": SHA, "data_producer_sha256": manifest["data_producer_sha256"]}
    begun = control.begin(
        s3, ecs, settings, {"kind": "weekly", "execution_id": "first", "request": {}}, metadata
    )
    request, _ = storage.get_json(s3, "bucket", begun["request_key"])
    pd.DataFrame({"season": [2025], "value": [2]}).to_parquet(
        producer["raw_dir"] / "weekly.parquet"
    )
    release.seal_inputs(**producer)
    candidate = release.publish_release(s3, "bucket", promote=False, **producer)
    assert release.resolve_release(s3, "bucket")[0] == old
    result = {
        "run_id": request["run_id"],
        "kind": "weekly",
        "request": request,
        "data_release": candidate,
        "cache": staged(s3, request, candidate),
    }
    storage.put_json(s3, "bucket", storage.run_key(request["run_id"], "result.json"), result)
    return s3, ecs, settings, metadata, request, result, old


@pytest.mark.parametrize("mode", ["shadow", "active"])
@pytest.mark.parametrize("kind", ["inference", "weekly"])
def test_begin_accepts_canonical_publisher_models_in_every_workflow(system, mode, kind):
    s3, ecs, settings, metadata, *_ = system
    settings = {**settings, "mode": mode}
    begun = control.begin(
        s3,
        ecs,
        settings,
        {"kind": kind, "execution_id": f"canonical-{mode}-{kind}", "request": {}},
        metadata,
    )
    request, _ = storage.get_json(s3, "bucket", begun["request_key"])
    assert set(request["models"]) == set(storage.POSITIONS)
    assert request["mode"] == mode
    assert request["kind"] == kind


def test_staging_leaves_current_data_and_cache_unchanged(system):
    s3, ecs, settings, metadata, request, result, old = system
    assert release.resolve_release(s3, "bucket")[0] == old
    assert (
        storage.get_json(s3, "bucket", "models/predictions_cache/current.json")[0]
        == request["cache_pointer"]["value"]
    )
    assert ecs.current == "arn:old"


def test_daily_scheduled_delivery_pins_once_and_reuses_request(system):
    s3, ecs, settings, metadata, request, result, old = system
    event = {
        "kind": "inference",
        "execution_id": "first",
        "request": {"scheduled_at": "2026-09-10T10:15:00Z"},
    }
    first = control.begin(s3, ecs, settings, event, metadata)
    s3.objects["models/releases/v3/QB/manifest.json"] = b"invalid after the snapshot"
    event["execution_id"] = "duplicate"
    assert control.begin(s3, ecs, settings, event, metadata) == first
    saved, _ = storage.get_json(s3, "bucket", first["request_key"])
    assert saved["data_release"] == old
    assert set(saved["models"]) == set(storage.POSITIONS)


def test_active_run_refuses_an_image_different_from_serving(system):
    s3, ecs, settings, metadata, request, result, old = system
    metadata["source_sha"] = "b" * 40
    with pytest.raises(ValueError, match="deployed source"):
        control.begin(s3, ecs, settings, {"kind": "inference", "execution_id": "other"}, metadata)


def test_shadow_completion_does_not_activate_or_publish_to_serving(system):
    s3, ecs, settings, metadata, request, result, old = system
    settings["mode"] = result["request"]["mode"] = "shadow"
    storage.put_json(s3, "bucket", storage.run_key(request["run_id"], "result.json"), result)
    before = {k: v for k, v in s3.objects.items() if not k.startswith("maintenance/")}
    outcome = control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert outcome["mode"] == "shadow"
    assert {k: v for k, v in s3.objects.items() if not k.startswith("maintenance/")} == before
    assert ecs.current == "arn:old"


def test_failed_activation_restores_previous_data_and_cache(system):
    s3, ecs, settings, metadata, request, result, old = system
    ecs.fail = "before"
    with pytest.raises(OSError, match="API unavailable"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert release.resolve_release(s3, "bucket")[0] == old
    assert (
        storage.get_json(s3, "bucket", "models/predictions_cache/current.json")[0]
        == request["cache_pointer"]["value"]
    )
    assert ecs.current == "arn:old"


def test_lost_update_response_rolls_back_before_retry(system):
    s3, ecs, settings, metadata, request, result, old = system
    ecs.fail = "after"
    with pytest.raises(OSError, match="response lost"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    receipt = control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert receipt["data_release"] == result["data_release"]
    assert ecs.registrations == 2
    assert control.verify(s3, ecs, settings, request["run_id"])["verified_at"]


def test_unchanged_data_still_validates_and_publishes_its_cache(system):
    s3, ecs, settings, metadata, request, result, old = system
    result["data_release"] = old
    result["cache"] = staged(s3, request, old)
    storage.put_json(s3, "bucket", storage.run_key(request["run_id"], "result.json"), result)
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    pointer, _ = storage.get_json(s3, "bucket", "models/predictions_cache/current.json")
    assert pointer != request["cache_pointer"]["value"]
    assert ecs.current == "arn:new-1"


def test_rollout_failure_can_restore_only_its_own_publication(system):
    s3, ecs, settings, metadata, request, result, old = system
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert control.restore_activation(s3, ecs, settings, result)["restored"]
    assert ecs.current == "arn:old"
    assert release.resolve_release(s3, "bucket")[0] == old
    assert (
        storage.get_json(s3, "bucket", "models/predictions_cache/current.json")[0]
        == request["cache_pointer"]["value"]
    )


def test_rollback_preserves_a_later_code_deployment(system):
    s3, ecs, settings, metadata, request, result, old = system
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    ecs.definitions[ecs.current]["containerDefinitions"][0]["image"] = "registry/app:" + "b" * 40
    before = copy.deepcopy(s3.objects)
    with pytest.raises(RuntimeError, match="later code deployment"):
        control.restore_activation(s3, ecs, settings, result)
    assert s3.objects == before


def test_model_change_blocks_activation_before_any_pointer_moves(system):
    s3, ecs, settings, metadata, request, result, old = system
    storage.put_json(
        s3,
        "bucket",
        "models/releases/v3/QB/manifest.json",
        {"stable": {"key": "models/QB/new", "bytes": 3}},
    )
    with pytest.raises(RuntimeError, match="QB changed"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert release.resolve_release(s3, "bucket")[0] == old
    assert (
        storage.get_json(s3, "bucket", "models/predictions_cache/current.json")[0]
        == request["cache_pointer"]["value"]
    )


def test_newer_same_producer_publication_wins_before_activation(system, producer):
    s3, ecs, settings, metadata, request, result, old = system
    pd.DataFrame({"season": [2025], "value": [3]}).to_parquet(
        producer["raw_dir"] / "weekly.parquet"
    )
    release.seal_inputs(**producer)
    newer = release.publish_release(s3, "bucket", **producer)
    before = copy.deepcopy(s3.objects)
    with pytest.raises(RuntimeError, match="newer data publication"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert release.resolve_release(s3, "bucket")[0] == newer
    assert s3.objects == before
    assert ecs.current == "arn:old"


def test_rollback_preserves_later_same_image_and_data_deployment(system):
    s3, ecs, settings, metadata, request, result, old = system
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    later = copy.deepcopy(ecs.definitions[ecs.current])
    later.update(taskDefinitionArn="arn:later", cpu="8192")
    ecs.definitions["arn:later"] = later
    ecs.current = "arn:later"
    before = copy.deepcopy(s3.objects)
    with pytest.raises(RuntimeError, match="later data deployment"):
        control.restore_activation(s3, ecs, settings, result)
    assert ecs.current == "arn:later"
    assert s3.objects == before


def test_activation_intent_is_durable_before_ecs_update(system, monkeypatch):
    s3, ecs, settings, metadata, request, result, old = system
    original = ecs.update_service

    def update(**arguments):
        backup, _ = storage.get_json(
            s3, "bucket", storage.run_key(request["run_id"], "activation-backup.json")
        )
        assert backup["activated_task_definition"] == arguments["taskDefinition"]
        return original(**arguments)

    monkeypatch.setattr(ecs, "update_service", update)
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])


def test_explicit_resume_reuses_preparation_after_rollback(system, tmp_path, monkeypatch):
    s3, ecs, settings, metadata, request, result, old = system
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    control.restore_activation(s3, ecs, settings, result)
    resumed = control.begin(
        s3,
        ecs,
        settings,
        {
            "kind": "weekly",
            "execution_id": "recovery",
            "request": {"resume_run_id": request["run_id"]},
        },
        metadata,
    )
    assert resumed["run_id"] == request["run_id"]
    (tmp_path / "maintenance-image.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        worker,
        "prepare_release",
        lambda *a, **k: pytest.fail("completed preparation must be reused"),
    )
    assert (
        worker.execute(s3, Dynamo(), bucket="bucket", table="table", request=request, root=tmp_path)
        == result
    )
    receipt = control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert receipt["data_release"] == result["data_release"]
    assert not receipt.get("rolled_back")
    assert ecs.current == "arn:new-2"


def test_expired_or_wrong_workflow_resume_requires_fresh_run(system):
    s3, ecs, settings, metadata, request, result, old = system
    event = {
        "kind": "inference",
        "execution_id": "recovery",
        "request": {"resume_run_id": request["run_id"]},
    }
    with pytest.raises(ValueError, match="does not match"):
        control.begin(s3, ecs, settings, event, metadata)
    event["kind"] = "weekly"
    request["created_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    storage.put_json(s3, "bucket", storage.run_key(request["run_id"], "request.json"), request)
    with pytest.raises(ValueError, match="expired"):
        control.begin(s3, ecs, settings, event, metadata)


def test_rollback_does_not_partially_revert_a_newer_data_publication(system, producer):
    s3, ecs, settings, metadata, request, result, old = system
    control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    pd.DataFrame({"season": [2025], "value": [4]}).to_parquet(
        producer["raw_dir"] / "weekly.parquet"
    )
    release.seal_inputs(**producer)
    release.publish_release(s3, "bucket", **producer)
    before = copy.deepcopy(s3.objects)
    task = ecs.current
    with pytest.raises(RuntimeError, match="later data publication"):
        control.restore_activation(s3, ecs, settings, result)
    assert s3.objects == before
    assert ecs.current == task


def test_lost_final_receipt_recovers_verified_rollout_without_redeploying(system, monkeypatch):
    s3, ecs, settings, metadata, request, result, old = system
    original = s3.put_object
    receipt_key = storage.run_key(request["run_id"], "published.json")
    failed = False

    def put(**kwargs):
        nonlocal failed
        if kwargs["Key"] == receipt_key and not failed:
            failed = True
            raise OSError("receipt response unavailable")
        return original(**kwargs)

    monkeypatch.setattr(s3, "put_object", put)
    with pytest.raises(OSError, match="receipt response"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert ecs.current == "arn:new-1"
    receipt = control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert receipt["task_definition"] == "arn:new-1"
    assert ecs.registrations == 1
    assert control.verify(s3, ecs, settings, request["run_id"])["verified_at"]


def test_newer_snapshot_prevents_activation_before_any_data_write(system):
    s3, ecs, settings, metadata, request, result, old = system
    storage.put_json(s3, "bucket", "models/predictions_cache/current.json", {"newer": True})
    before = copy.deepcopy(s3.objects)
    with pytest.raises(RuntimeError, match="newer serving snapshot"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert s3.objects == before
    assert ecs.current == "arn:old"


def test_rollout_budget_is_checked_before_publication(system):
    s3, ecs, settings, metadata, request, result, old = system
    settings["rollout_timeout"] = 0
    before = copy.deepcopy(s3.objects)
    with pytest.raises(RuntimeError, match="Insufficient invocation time"):
        control.finish(s3, Dynamo(), ecs, settings, request["run_id"])
    assert s3.objects == before
    assert ecs.current == "arn:old"
