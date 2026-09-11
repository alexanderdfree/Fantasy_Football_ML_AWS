"""Advance an ECS task's data pin only within its deployed producer recipe.

Run in a job sharing deploy.yml's concurrency group. A new producer recipe is
left to its code deployment; unchanged-code refreshes preserve the exact image
and advance only alongside its verified prediction snapshot and readiness rollout.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.artifacts import deployment, serving_snapshot
from src.data.release import resolve_release
from src.scripts.wait_data_release import AwsS3

_CONTAINER = "fantasy-predictor"
_DROP_TASK_FIELDS = {
    "taskDefinitionArn",
    "revision",
    "status",
    "requiresAttributes",
    "compatibilities",
    "registeredAt",
    "registeredBy",
    "deregisteredAt",
}


class AwsEcs(deployment.AWSCLI):
    """Reuse the ECS/ALB transaction transport for data-only release advances."""


def _service_task(ecs, cluster, service):
    result = ecs.call("ecs", "describe-services", cluster=cluster, services=[service])
    if result.get("failures") or len(result.get("services", [])) != 1:
        raise RuntimeError(f"Cannot resolve running ECS service {cluster}/{service}")
    return result["services"][0]["taskDefinition"]


def advance_release(
    s3,
    ecs,
    *,
    bucket,
    cluster,
    service,
    release_id,
    state_path="data-release-rollout.json",
    timeout=1200,
    on_state=None,
    expected_task_definition=None,
):
    selected, manifest = resolve_release(s3, bucket, release_id=release_id)
    recipe = manifest["data_producer_sha256"]
    latest = json.loads(
        s3.get_object(Bucket=bucket, Key=f"data/by-producer/{recipe}/manifest.json")["Body"].read()
    )
    if latest.get("release_id") != selected:
        return {
            "advanced": False,
            "reason": "a newer release of this producer is already published",
        }
    original_task = _service_task(ecs, cluster, service)
    if expected_task_definition is not None and original_task != expected_task_definition:
        raise RuntimeError("Running ECS task changed during maintenance preparation")
    task = ecs.call("ecs", "describe-task-definition", taskDefinition=original_task)[
        "taskDefinition"
    ]
    container = next(c for c in task["containerDefinitions"] if c["name"] == _CONTAINER)
    env = {item["name"]: item["value"] for item in container.get("environment", [])}
    old_release = env.get("FF_DATA_RELEASE")
    old_recipe = env.get("FF_DATA_PRODUCER_SHA256")
    if not old_release or not old_recipe or not env.get("FF_SERVING_SNAPSHOT_GENERATION"):
        return {
            "advanced": False,
            "reason": "running task awaits its first compatible code deployment",
        }
    if env.get("FF_DATASET_ID", old_release) != old_release:
        raise RuntimeError("Running task dataset and data release identities disagree")
    _, old_manifest = resolve_release(s3, bucket, release_id=old_release)
    if old_manifest.get("data_producer_sha256") != old_recipe:
        raise RuntimeError("Running task data pin and producer metadata disagree")
    if old_recipe != recipe:
        return {
            "advanced": False,
            "reason": "new data recipe requires its matching code deployment",
        }
    serving_bucket, prefix = deployment.serving_coordinates(task)
    if serving_bucket != bucket:
        raise RuntimeError("Running serving task reads a different bucket")
    try:
        # Rollback must be able to bootstrap replacement tasks after current
        # has advanced, not merely rely on the old workers' local cache.
        serving_snapshot.verify_remote(
            s3,
            bucket,
            prefix,
            expected_dataset_id=old_release,
            generation=env["FF_SERVING_SNAPSHOT_GENERATION"],
        )
        snapshot_pointer, snapshot = serving_snapshot.verify_remote(
            s3, bucket, prefix, expected_dataset_id=selected
        )
        serving_snapshot.verify_data_release(s3, bucket, snapshot, expected_producer=recipe)
    except (ValueError, FileNotFoundError) as error:
        return {"advanced": False, "reason": f"matching serving snapshot is not ready: {error}"}
    if (
        old_release == selected
        and env["FF_SERVING_SNAPSHOT_GENERATION"] == snapshot_pointer["generation"]
    ):
        return {"advanced": False, "reason": "running task already pins this release"}
    latest = json.loads(
        s3.get_object(Bucket=bucket, Key=f"data/by-producer/{recipe}/manifest.json")["Body"].read()
    )
    if latest.get("release_id") != selected:
        return {"advanced": False, "reason": "a newer release arrived during snapshot verification"}
    definition = {
        key: copy.deepcopy(value) for key, value in task.items() if key not in _DROP_TASK_FIELDS
    }
    definition = deployment.bind_data_release(
        definition, selected, recipe, snapshot_generation=snapshot_pointer["generation"]
    )
    new_task = deployment.deploy(
        ecs,
        definition,
        cluster=cluster,
        service=service,
        state_path=state_path,
        expected_current_task=original_task,
        timeout=timeout,
        on_state=on_state,
    )
    return {"advanced": True, "release_id": selected, "task_definition": new_task}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--state", type=Path, default=Path("data-release-rollout.json"))
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    print(
        json.dumps(
            advance_release(
                AwsS3(),
                AwsEcs(),
                bucket=args.bucket,
                cluster=args.cluster,
                service=args.service,
                release_id=args.release_id,
                state_path=args.state,
                timeout=args.timeout,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
