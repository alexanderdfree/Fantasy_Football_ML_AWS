"""First artifact-only cutover preserves legacy tasks and verifies the new revision."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.artifacts import deployment, serving_snapshot

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def task_definition():
    document = json.loads((ROOT / "infra/aws/task-definition.json").read_text())
    container = document["containerDefinitions"][0]
    container["image"] = "registry/app:new-image"
    container["environment"] = [
        {"name": "FF_MODEL_S3_BUCKET", "value": "actual-bucket"},
        {"name": "FF_MODEL_S3_PREFIX", "value": "actual/prefix"},
        {"name": "FF_ALLOW_RUNTIME_INFERENCE", "value": "0"},
    ]
    return document


class RolloutAWS:
    def __init__(self, *, response_code=503, wrong_image=False, auto_rollback=False):
        self.current = "old-revision"
        self.code = response_code
        self.wrong_image = wrong_image
        self.auto_rollback = auto_rollback
        self.health = {
            "HealthCheckPath": "/health",
            "Matcher": {"HttpCode": "200"},
            "HealthCheckIntervalSeconds": 30,
            "HealthyThresholdCount": 5,
            "HealthCheckTimeoutSeconds": 5,
        }
        self.original_health = copy.deepcopy(self.health)
        self.attributes = [{"Key": "deregistration_delay.timeout_seconds", "Value": "300"}]
        self.events = []
        self.fail_update = False

    def call(self, api, operation, **request):
        self.events.append((api, operation, copy.deepcopy(request)))
        if operation == "describe-services":
            if self.auto_rollback and self.current == "new-revision":
                self.current = "old-revision"
            ready = self.current == "old-revision" or self.code == 200
            return {
                "services": [
                    {
                        "taskDefinition": self.current,
                        "desiredCount": 1,
                        "runningCount": 1,
                        "pendingCount": 0,
                        "loadBalancers": [
                            {"containerName": "fantasy-predictor", "targetGroupArn": "target-group"}
                        ],
                        "deployments": [
                            {
                                "taskDefinition": self.current,
                                "rolloutState": "COMPLETED" if ready else "IN_PROGRESS",
                            }
                        ],
                    }
                ]
            }
        if operation == "describe-target-groups":
            return {"TargetGroups": [{"TargetGroupArn": "target-group", **self.health}]}
        if operation == "describe-target-group-attributes":
            return {"Attributes": copy.deepcopy(self.attributes)}
        if operation == "register-task-definition":
            assert request["containerDefinitions"][0]["image"] == "registry/app:new-image"
            return {"taskDefinition": {"taskDefinitionArn": "new-revision"}}
        if operation == "modify-target-group":
            self.health.update(
                {
                    key: copy.deepcopy(value)
                    for key, value in request.items()
                    if key != "TargetGroupArn"
                }
            )
            return {}
        if operation == "modify-target-group-attributes":
            self.attributes = copy.deepcopy(request["Attributes"])
            return {}
        if operation == "update-service":
            if request["taskDefinition"] == "new-revision":
                # The old image's /ready returns404. It must remain healthy
                # while replacement tasks return503 and hydrate their cache.
                assert self.health["HealthCheckPath"] == "/ready"
                assert "404" in self.health["Matcher"]["HttpCode"].split(",")
                if self.fail_update:
                    raise RuntimeError("update failed")
            self.current = request["taskDefinition"]
            return {}
        if operation == "list-tasks":
            return {"taskArns": ["new-task"]}
        if operation == "describe-tasks":
            return {
                "tasks": [
                    {
                        "taskDefinitionArn": self.current,
                        "lastStatus": "RUNNING",
                        "containers": [
                            {
                                "name": "fantasy-predictor",
                                "image": "registry/app:wrong"
                                if self.wrong_image
                                else "registry/app:new-image",
                                "healthStatus": "HEALTHY" if self.code == 200 else "UNHEALTHY",
                            }
                        ],
                    }
                ]
            }
        raise AssertionError((api, operation, request))


def test_deploy_refuses_a_service_revision_changed_since_rendering(tmp_path):
    aws = RolloutAWS(response_code=200)
    with pytest.raises(RuntimeError, match="refusing stale rollout"):
        deployment.deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=tmp_path / "state.json",
            expected_current_task="superseded-revision",
        )
    assert not any(
        operation in {"register-task-definition", "update-service", "modify-target-group"}
        for _, operation, _ in aws.events
    )


@pytest.mark.parametrize("operator_changes_health", [False, True])
def test_operator_revision_during_health_migration_is_preserved_and_own_health_restored(
    tmp_path, operator_changes_health
):
    class OperatorRace(RolloutAWS):
        def __init__(self):
            super().__init__(response_code=200)
            self.intervened = False

        def call(self, api, operation, **request):
            result = super().call(api, operation, **request)
            if operation == "modify-target-group-attributes" and not self.intervened:
                self.intervened = True
                self.current = "operator-revision"
                if operator_changes_health:
                    self.health.update(
                        HealthCheckPath="/operator-ready",
                        Matcher={"HttpCode": "204"},
                        HealthCheckTimeoutSeconds=12,
                    )
                    self.attributes = [
                        {"Key": "deregistration_delay.timeout_seconds", "Value": "60"}
                    ]
            return result

    aws = OperatorRace()
    with pytest.raises(RuntimeError, match="refusing stale rollout"):
        deployment.deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=tmp_path / "state.json",
            expected_current_task="old-revision",
            timeout=0,
        )
    assert aws.intervened is True
    assert aws.current == "operator-revision"
    assert not any(operation == "update-service" for _, operation, _ in aws.events)
    expected_health = dict(aws.original_health)
    if operator_changes_health:
        expected_health.update(
            HealthCheckPath="/operator-ready",
            Matcher={"HttpCode": "204"},
            HealthCheckTimeoutSeconds=12,
        )
    assert aws.health == expected_health
    assert aws.attributes == [
        {
            "Key": "deregistration_delay.timeout_seconds",
            "Value": "60" if operator_changes_health else "300",
        }
    ]
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "restored"


def test_legacy_404_stays_eligible_while_new_503_waits_then_only_new_200_is_accepted(tmp_path):
    aws = RolloutAWS(response_code=503)
    now = [0]

    def warm(seconds):
        assert aws.health["Matcher"] == {"HttpCode": "200,404"}
        assert "503" not in aws.health["Matcher"]["HttpCode"].split(",")
        now[0] += seconds
        aws.code = 200

    result = deployment.deploy(
        aws,
        task_definition(),
        cluster="cluster",
        service="service",
        state_path=tmp_path / "state.json",
        timeout=2,
        poll=1,
        clock=lambda: now[0],
        sleep=warm,
    )
    assert result == "new-revision"
    assert now[0] == 1
    assert aws.health["HealthCheckPath"] == "/ready"
    assert aws.health["Matcher"] == {"HttpCode": "200"}
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "complete"
    events = [
        request["Matcher"]["HttpCode"]
        for _, operation, request in aws.events
        if operation == "modify-target-group"
    ]
    assert events == ["200,404", "200"]


@pytest.mark.parametrize("response_code", [503, 404])
def test_unready_new_revision_times_out_and_restores_old_health(response_code, tmp_path):
    aws = RolloutAWS(response_code=response_code)
    with pytest.raises(RuntimeError, match="timeout"):
        deployment.deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=tmp_path / "state.json",
            timeout=0,
        )
    assert aws.current == "old-revision"
    assert aws.health == aws.original_health
    assert aws.attributes == [{"Key": "deregistration_delay.timeout_seconds", "Value": "300"}]
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "restored"


@pytest.mark.parametrize("failure", ["wrong_image", "auto_rollback", "update_failure"])
def test_false_service_stability_or_update_failure_cannot_leave_compatibility_matcher(
    failure, tmp_path
):
    aws = RolloutAWS(
        response_code=200,
        wrong_image=failure == "wrong_image",
        auto_rollback=failure == "auto_rollback",
    )
    aws.fail_update = failure == "update_failure"
    with pytest.raises(RuntimeError):
        deployment.deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=tmp_path / "state.json",
            timeout=0,
        )
    assert aws.current == "old-revision"
    assert aws.health == aws.original_health


def test_interrupted_rollout_restores_health_and_saved_recovery_is_idempotent(tmp_path):
    aws = RolloutAWS(response_code=503)

    def interrupted(_):
        raise KeyboardInterrupt()

    state_file = tmp_path / "state.json"
    with pytest.raises(KeyboardInterrupt):
        deployment.deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=state_file,
            sleep=interrupted,
        )
    state = json.loads(state_file.read_text())
    assert state["phase"] == "restored"
    before = len(aws.events)
    deployment.restore_rollout(aws, state, state_path=state_file)
    assert len(aws.events) == before
    assert aws.health == aws.original_health


def test_readiness_uses_the_rendered_task_source(monkeypatch, tmp_path):
    from tests.shared.test_model_sync import _generation_cache_fixture

    _, objects, _ = _generation_cache_fixture("actual/prefix")
    document = task_definition()
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(document))
    assert deployment.serving_coordinates(document) == ("actual-bucket", "actual/prefix")

    def fake_aws(command, **_):
        assert command[command.index("--bucket") + 1] == "actual-bucket"
        key = command[command.index("--key") + 1]
        assert key.startswith("actual/prefix/")
        Path(command[-1]).write_bytes(objects[key])
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_aws)
    monkeypatch.setattr(
        sys,
        "argv",
        ["serving_snapshot", "wait", "--task-definition", str(task_file), "--timeout", "0"],
    )
    serving_snapshot.main()


def test_workflow_uses_transactional_deploy_and_cleanup_and_bootstrap_is_strict():
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    readiness = next(
        step
        for step in steps
        if step.get("name") == "Require a compatible published serving snapshot"
    )
    deploy = next(step for step in steps if step.get("name") == "Deploy to ECS")
    restore = next(
        step
        for step in steps
        if step.get("name") == "Restore health settings after incomplete rollout"
    )
    assert steps.index(readiness) < steps.index(deploy) < steps.index(restore)
    assert "--task-definition" in readiness["run"] and "--bucket" not in readiness["run"]
    assert "src.artifacts.deployment deploy" in deploy["run"]
    assert "failure()" in restore["if"] and "cancelled()" in restore["if"]
    assert "src.artifacts.deployment restore" in restore["run"]
    assert "github.ref" not in workflow["concurrency"]["group"]
    bootstrap = (ROOT / "infra/aws/bootstrap.sh").read_text()
    assert "--health-check-path /ready --matcher HttpCode=200" in bootstrap
