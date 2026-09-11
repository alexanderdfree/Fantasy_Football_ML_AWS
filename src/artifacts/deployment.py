"""Compose owned serving settings with the deployment's existing ARNs/secrets."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
import time
from pathlib import Path

READ_ONLY = (
    "taskDefinitionArn",
    "revision",
    "status",
    "requiresAttributes",
    "compatibilities",
    "registeredAt",
    "registeredBy",
)


def render_task_definition(live: dict, desired: dict) -> dict:
    result = copy.deepcopy(live)
    for key in READ_ONLY:
        result.pop(key, None)
    for key in ("cpu", "memory", "runtimePlatform", "networkMode", "requiresCompatibilities"):
        result[key] = copy.deepcopy(desired[key])
    wanted = next(c for c in desired["containerDefinitions"] if c["name"] == "fantasy-predictor")
    targets = [c for c in result["containerDefinitions"] if c["name"] == wanted["name"]]
    if len(targets) != 1:
        raise ValueError("Expected exactly one fantasy-predictor container")
    target = targets[0]
    target["healthCheck"] = copy.deepcopy(wanted["healthCheck"])
    # Only this owned switch is copied from the template. Real bucket/region,
    # application environment, secrets and role identifiers remain live values.
    environment = [
        item
        for item in target.get("environment", [])
        if item["name"] != "FF_ALLOW_RUNTIME_INFERENCE"
    ]
    policy = next(
        item for item in wanted["environment"] if item["name"] == "FF_ALLOW_RUNTIME_INFERENCE"
    )
    target["environment"] = [*environment, copy.deepcopy(policy)]
    return result


def serving_coordinates(task_definition: dict) -> tuple[str, str]:
    """Read the same bucket/prefix that the rendered container will consume."""
    container = _application_container(task_definition)
    environment = {item["name"]: item["value"] for item in container.get("environment", [])}
    bucket = environment.get("FF_MODEL_S3_BUCKET", "").strip()
    if not bucket or bucket.startswith("__"):
        raise ValueError("Rendered serving task has no concrete FF_MODEL_S3_BUCKET")
    return bucket, environment.get("FF_MODEL_S3_PREFIX", "models").strip("/")


def bind_data_release(task_definition, release_id, producer, *, snapshot_generation=None):
    """Bind task authority to the data underlying its verified prediction snapshot."""
    result = copy.deepcopy(task_definition)
    container = _application_container(result)
    fields = {
        "FF_DATA_RELEASE": release_id,
        "FF_DATASET_ID": release_id,
        "FF_DATA_PRODUCER_SHA256": producer,
    }
    if snapshot_generation is not None:
        fields["FF_SERVING_SNAPSHOT_GENERATION"] = snapshot_generation
    container["environment"] = [
        item for item in container.get("environment", []) if item["name"] not in fields
    ] + [{"name": name, "value": value} for name, value in fields.items()]
    return result


def _application_container(task_definition):
    containers = [
        c for c in task_definition["containerDefinitions"] if c["name"] == "fantasy-predictor"
    ]
    if len(containers) != 1:
        raise ValueError("Expected exactly one fantasy-predictor container")
    return containers[0]


class AWSCLI:
    """Thin stdlib transport; deployment runners need no Python AWS dependency."""

    def __init__(self, region=None):
        self.region = region

    def call(self, api, operation, **request):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
            json.dump(request, stream)
            stream.flush()
            command = [
                "aws",
                api,
                operation,
                "--cli-input-json",
                f"file://{stream.name}",
                "--output",
                "json",
                "--no-cli-pager",
            ]
            if self.region:
                command.extend(["--region", self.region])
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"AWS {api} {operation} failed: {result.stderr.strip()}")
        return json.loads(result.stdout) if result.stdout.strip() else {}


_HEALTH_FIELDS = (
    "HealthCheckProtocol",
    "HealthCheckPort",
    "HealthCheckEnabled",
    "HealthCheckPath",
    "HealthCheckIntervalSeconds",
    "HealthCheckTimeoutSeconds",
    "HealthyThresholdCount",
    "UnhealthyThresholdCount",
    "Matcher",
)
_COMPATIBILITY_HEALTH = {
    # Legacy images ignore this query and keep their existing /health response;
    # current images delegate it to /ready, rejecting unhydrated replacements.
    "HealthCheckPath": "/health?readiness=1",
    "Matcher": {"HttpCode": "200"},
    "HealthyThresholdCount": 2,
    "HealthCheckIntervalSeconds": 10,
}
_DRAIN_ATTRIBUTE = "deregistration_delay.timeout_seconds"


def _service(aws, cluster, service):
    response = aws.call("ecs", "describe-services", cluster=cluster, services=[service])
    if response.get("failures") or len(response.get("services", [])) != 1:
        raise RuntimeError(f"Cannot resolve serving service {service}")
    return response["services"][0]


def _save_state(path, state, on_state=None):
    if on_state is not None:
        # Remote controllers must durably record intent before the next mutation.
        on_state(copy.deepcopy(state))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _running_tasks(aws, cluster, service):
    arns, request = [], {"cluster": cluster, "serviceName": service, "desiredStatus": "RUNNING"}
    while True:
        response = aws.call("ecs", "list-tasks", **request)
        arns.extend(response.get("taskArns", []))
        if not response.get("nextToken"):
            break
        request["nextToken"] = response["nextToken"]
    tasks = []
    for start in range(0, len(arns), 100):
        response = aws.call(
            "ecs", "describe-tasks", cluster=cluster, tasks=arns[start : start + 100]
        )
        if response.get("failures"):
            raise RuntimeError("Cannot inspect expected serving tasks")
        tasks.extend(response.get("tasks", []))
    return tasks


def _expected_release_ready(aws, state):
    service = _service(aws, state["cluster"], state["service"])
    expected = state["expected_task_definition"]
    if service.get("taskDefinition") != expected:
        raise RuntimeError("Serving rollout was rolled back or superseded by a different revision")
    deployments = service.get("deployments", [])
    if any(
        d.get("taskDefinition") == expected and d.get("rolloutState") == "FAILED"
        for d in deployments
    ):
        raise RuntimeError("Expected serving deployment failed")
    desired = service.get("desiredCount", 0)
    if (
        desired < 1
        or service.get("runningCount") != desired
        or service.get("pendingCount") != 0
        or len(deployments) != 1
        or deployments[0].get("taskDefinition") != expected
        or deployments[0].get("rolloutState") != "COMPLETED"
    ):
        return False
    tasks = _running_tasks(aws, state["cluster"], state["service"])
    if len(tasks) != desired:
        return False
    for task in tasks:
        if task.get("taskDefinitionArn") != expected or task.get("lastStatus") != "RUNNING":
            return False
        containers = [c for c in task.get("containers", []) if c.get("name") == "fantasy-predictor"]
        if (
            len(containers) != 1
            or containers[0].get("image") != state["expected_image"]
            or containers[0].get("healthStatus") != "HEALTHY"
        ):
            return False
    return True


def restore_rollout(aws, state, *, state_path, on_state=None):
    """Restore only this deployment's settings; never overwrite another revision."""
    if state.get("phase") in {"complete", "restored"}:
        return
    current = _service(aws, state["cluster"], state["service"])["taskDefinition"]
    superseded = current not in {state["prior_task_definition"], state["expected_task_definition"]}
    failures = []
    if current == state["expected_task_definition"]:
        try:
            aws.call(
                "ecs",
                "update-service",
                cluster=state["cluster"],
                service=state["service"],
                taskDefinition=state["prior_task_definition"],
                forceNewDeployment=True,
            )
        except Exception as error:
            failures.append(str(error))
    for group in state["target_groups"]:
        try:
            health, attributes = group["health"], group["attributes"]
            if superseded:
                # Another operator owns the service now. Restore only fields
                # still equal to our compatibility settings, retaining any
                # independent health-policy changes they have made.
                observed = aws.call(
                    "elbv2", "describe-target-groups", TargetGroupArns=[group["arn"]]
                )["TargetGroups"][0]
                health = {
                    key: group["health"][key]
                    for key, value in _COMPATIBILITY_HEALTH.items()
                    if key in group["health"] and observed.get(key) == value
                }
                observed_attributes = aws.call(
                    "elbv2", "describe-target-group-attributes", TargetGroupArn=group["arn"]
                )["Attributes"]
                owns_drain = any(
                    item.get("Key") == _DRAIN_ATTRIBUTE and item.get("Value") == "30"
                    for item in observed_attributes
                )
                attributes = attributes if owns_drain else []
            if health:
                aws.call("elbv2", "modify-target-group", TargetGroupArn=group["arn"], **health)
            if attributes:
                aws.call(
                    "elbv2",
                    "modify-target-group-attributes",
                    TargetGroupArn=group["arn"],
                    Attributes=attributes,
                )
        except Exception as error:
            failures.append(str(error))
    state["phase"] = "restore-failed" if failures else "restored"
    _save_state(state_path, state, on_state)
    if failures:
        raise RuntimeError("Serving rollback restoration failed: " + "; ".join(failures))


def deploy(
    aws,
    task_definition,
    *,
    cluster,
    service,
    state_path,
    timeout=1200,
    poll=10,
    clock=time.monotonic,
    sleep=time.sleep,
    expected_current_task=None,
    on_state=None,
):
    """Migrate readiness without making the still-running legacy image unhealthy."""
    container = _application_container(task_definition)
    if "/ready" not in " ".join(container.get("healthCheck", {}).get("command", [])):
        raise ValueError("Serving deployment requires the container /ready health check")
    before = _service(aws, cluster, service)
    if expected_current_task is not None and before["taskDefinition"] != expected_current_task:
        raise RuntimeError("Running ECS task changed during preparation; refusing stale rollout")
    if before.get("desiredCount", 0) < 1:
        raise ValueError("Serving deployment needs a positive desired count to verify readiness")
    arns = sorted(
        {
            lb["targetGroupArn"]
            for lb in before.get("loadBalancers", [])
            if lb.get("targetGroupArn")
            and lb.get("containerName", "fantasy-predictor") == "fantasy-predictor"
        }
    )
    if not arns:
        raise ValueError("Serving deployment has no application target group")
    groups = aws.call("elbv2", "describe-target-groups", TargetGroupArns=arns)["TargetGroups"]
    recorded = []
    for group in groups:
        health = {key: copy.deepcopy(group[key]) for key in _HEALTH_FIELDS if key in group}
        if "HttpCode" not in health.get("Matcher", {}):
            raise ValueError("Serving readiness migration requires HTTP target-group health checks")
        attributes = aws.call(
            "elbv2", "describe-target-group-attributes", TargetGroupArn=group["TargetGroupArn"]
        )["Attributes"]
        # Only the owned drain attribute is changed/restored.
        attributes = [
            item for item in attributes if item["Key"] == "deregistration_delay.timeout_seconds"
        ]
        recorded.append(
            {"arn": group["TargetGroupArn"], "health": health, "attributes": attributes}
        )
    registered = aws.call("ecs", "register-task-definition", **task_definition)
    expected = registered["taskDefinition"]["taskDefinitionArn"]
    if _service(aws, cluster, service)["taskDefinition"] != before["taskDefinition"]:
        raise RuntimeError("Running ECS task changed during preparation; refusing stale rollout")
    state = {
        "schema_version": 1,
        "cluster": cluster,
        "service": service,
        "prior_task_definition": before["taskDefinition"],
        "expected_task_definition": expected,
        "expected_image": container["image"],
        "target_groups": recorded,
        "phase": "prepared",
    }
    _save_state(state_path, state, on_state)
    try:
        for group in recorded:
            aws.call(
                "elbv2",
                "modify-target-group",
                TargetGroupArn=group["arn"],
                **_COMPATIBILITY_HEALTH,
            )
            aws.call(
                "elbv2",
                "modify-target-group-attributes",
                TargetGroupArn=group["arn"],
                Attributes=[{"Key": "deregistration_delay.timeout_seconds", "Value": "30"}],
            )
        state["phase"] = "compatibility"
        _save_state(state_path, state, on_state)
        # UpdateService has no revision CAS. Recheck after the intervening ALB
        # calls; a final read/update window still requires operator coordination.
        if _service(aws, cluster, service)["taskDefinition"] != before["taskDefinition"]:
            raise RuntimeError(
                "Running ECS task changed during preparation; refusing stale rollout"
            )
        aws.call(
            "ecs",
            "update-service",
            cluster=cluster,
            service=service,
            taskDefinition=expected,
            forceNewDeployment=True,
        )
        # A controller can supply its remaining invocation budget. Resolve it
        # after remote preparation/update, not before those calls consume time.
        wait_budget = timeout() if callable(timeout) else timeout
        deadline = clock() + wait_budget
        while not _expected_release_ready(aws, state):
            remaining = deadline - clock()
            if remaining <= 0:
                raise RuntimeError("Expected serving revision did not become ready before timeout")
            sleep(min(poll, remaining))
        for group in recorded:
            aws.call(
                "elbv2",
                "modify-target-group",
                TargetGroupArn=group["arn"],
                HealthCheckPath="/ready",
                Matcher={"HttpCode": "200"},
            )
        state["phase"] = "complete"
        _save_state(state_path, state, on_state)
        return expected
    except BaseException:
        restore_rollout(aws, state, state_path=state_path, on_state=on_state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", choices=["render", "deploy", "restore"], default="render"
    )
    parser.add_argument("--live", type=Path)
    parser.add_argument("--desired", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-definition", type=Path)
    parser.add_argument("--cluster")
    parser.add_argument("--service")
    parser.add_argument("--expected-current-task")
    parser.add_argument("--region")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.action == "render":
        if not all((args.live, args.desired, args.output)):
            parser.error("render requires --live, --desired and --output")
        rendered = render_task_definition(
            json.loads(args.live.read_text()), json.loads(args.desired.read_text())
        )
        args.output.write_text(json.dumps(rendered, indent=2) + "\n")
        return
    if args.state is None:
        parser.error("deploy/restore requires --state")
    aws = AWSCLI(args.region)
    if args.action == "restore":
        if args.state.exists():
            restore_rollout(aws, json.loads(args.state.read_text()), state_path=args.state)
        return
    if not all((args.task_definition, args.cluster, args.service)):
        parser.error("deploy requires --task-definition, --cluster and --service")
    arn = deploy(
        aws,
        json.loads(args.task_definition.read_text()),
        cluster=args.cluster,
        service=args.service,
        state_path=args.state,
        timeout=args.timeout,
        expected_current_task=args.expected_current_task,
    )
    print(f"Serving revision ready with strict ALB /ready health: {arn}")


if __name__ == "__main__":
    main()
