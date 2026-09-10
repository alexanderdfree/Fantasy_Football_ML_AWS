"""Advance an ECS task's data pin only within its deployed producer recipe.

Run in a job sharing deploy.yml's concurrency group. A new producer recipe is
left to its code deployment; unchanged-code refreshes preserve the exact image
and update only the verified data pin.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile

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


class AwsEcs:
    def call(self, operation, arguments):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as request:
            json.dump(arguments, request)
            request.flush()
            result = subprocess.run(
                [
                    "aws",
                    "ecs",
                    operation,
                    "--cli-input-json",
                    f"file://{request.name}",
                    "--no-cli-pager",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"ECS {operation} failed: {result.stderr.strip()}")
        return json.loads(result.stdout)


def _service_task(ecs, cluster, service):
    result = ecs.call("describe-services", {"cluster": cluster, "services": [service]})
    if result.get("failures") or len(result.get("services", [])) != 1:
        raise RuntimeError(f"Cannot resolve running ECS service {cluster}/{service}")
    return result["services"][0]["taskDefinition"]


def advance_release(s3, ecs, *, bucket, cluster, service, release_id):
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
    task = ecs.call("describe-task-definition", {"taskDefinition": original_task})["taskDefinition"]
    container = next(c for c in task["containerDefinitions"] if c["name"] == _CONTAINER)
    env = {item["name"]: item["value"] for item in container.get("environment", [])}
    old_release = env.get("FF_DATA_RELEASE")
    old_recipe = env.get("FF_DATA_PRODUCER_SHA256")
    if not old_release or not old_recipe:
        return {
            "advanced": False,
            "reason": "running task awaits its first compatible code deployment",
        }
    _, old_manifest = resolve_release(s3, bucket, release_id=old_release)
    if old_manifest.get("data_producer_sha256") != old_recipe:
        raise RuntimeError("Running task data pin and producer metadata disagree")
    if old_recipe != recipe:
        return {
            "advanced": False,
            "reason": "new data recipe requires its matching code deployment",
        }
    if old_release == selected:
        return {"advanced": False, "reason": "running task already pins this release"}
    definition = {
        key: copy.deepcopy(value) for key, value in task.items() if key not in _DROP_TASK_FIELDS
    }
    updated = next(c for c in definition["containerDefinitions"] if c["name"] == _CONTAINER)
    updated["environment"] = [
        item
        for item in updated.get("environment", [])
        if item["name"] not in {"FF_DATA_RELEASE", "FF_DATA_PRODUCER_SHA256"}
    ]
    updated["environment"].extend(
        [
            {"name": "FF_DATA_RELEASE", "value": selected},
            {"name": "FF_DATA_PRODUCER_SHA256", "value": recipe},
        ]
    )
    new_task = ecs.call("register-task-definition", definition)["taskDefinition"][
        "taskDefinitionArn"
    ]
    if _service_task(ecs, cluster, service) != original_task:
        raise RuntimeError("Running ECS task changed during preparation; refusing stale rollout")
    ecs.call("update-service", {"cluster": cluster, "service": service, "taskDefinition": new_task})
    return {"advanced": True, "release_id": selected, "task_definition": new_task}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--release-id", required=True)
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
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
