"""Infrastructure wiring, standalone packaging, and CI handoff contracts."""

import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from src.maintenance import package, update_runtime

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def template():
    return yaml.safe_load((ROOT / "infra/maintenance/template.yaml").read_text())


def test_two_small_workflows_wait_for_workers_and_verify_publication():
    resources = template()["Resources"]
    machines = [r for r in resources.values() if r["Type"] == "AWS::StepFunctions::StateMachine"]
    assert len(machines) == 2
    for machine in machines:
        props = machine["Properties"]
        assert props["StateMachineType"] == "STANDARD"
        states = props["Definition"]["States"]
        assert states["Infer"]["Resource"].endswith("ecs:runTask.sync")
        assert states["Infer"]["Next"] == "Publish"
        assert states["Publish"]["Next"] == "Verify"
        assert states["Verify"]["Parameters"]["Payload"]["action"] == "verify"
        for state in states.values():
            if state["Type"] == "Task":
                assert state["Resource"] != "arn:aws:states:::ecs:runTask"
    weekly = resources["WeeklyWorkflow"]["Properties"]["Definition"]["States"]
    assert weekly["Prepare"]["Next"] == "Activate"
    assert weekly["Activate"]["Next"] == "VerifyActivation"
    assert weekly["Activate"]["Catch"][0]["Next"] == "Rollback"
    assert any("States.TaskFailed" in r["ErrorEquals"] for r in weekly["Activate"]["Retry"])
    assert weekly["VerifyActivation"]["Catch"][0]["Next"] == "Rollback"
    assert (
        weekly["BeginInference"]["Parameters"]["Payload"]["request"]["expected_data_release.$"]
        == "$.activation.value.data_release"
    )


def test_default_stack_is_shadow_and_schedules_are_disabled():
    doc = template()
    assert doc["Parameters"]["Mode"]["Default"] == "shadow"
    assert doc["Parameters"]["EnableSchedules"]["Default"] == "false"
    resources = doc["Resources"]
    for value in resources.values():
        if value["Type"] == "AWS::Scheduler::Schedule":
            assert value["Properties"]["State"]["Fn::If"][-1] == "DISABLED"
    assert (
        resources["DailySchedule"]["Properties"]["ScheduleExpressionTimezone"] == "America/New_York"
    )
    assert resources["WeeklySchedule"]["Properties"]["ScheduleExpression"] == "cron(15 7 ? * THU *)"
    # Only the control role can change production pointers/service; workers stage outputs.
    for role in ["InferenceRole", "PreparationRole"]:
        policies = json.dumps(resources[role]["Properties"]["Policies"])
        assert "ecs:UpdateService" not in policies
        assert "models/predictions_cache" not in policies
        assert "data/manifest.json" not in policies


def test_activation_has_time_and_permissions_for_the_canonical_rollout():
    resources = template()["Resources"]
    states = resources["WeeklyWorkflow"]["Properties"]["Definition"]["States"]
    assert resources["Control"]["Properties"]["Timeout"] == 900
    assert states["Activate"]["TimeoutSeconds"] > 900
    assert states["Rollback"]["TimeoutSeconds"] > 900
    policy = json.dumps(resources["ControlRole"]["Properties"]["Policies"])
    for value in (
        "current.json",
        "generations/*",
        "elasticloadbalancing:ModifyTargetGroup",
        "elasticloadbalancing:DescribeTargetGroups",
        "ecs:DescribeTasks",
    ):
        assert value in policy
    assert "cache.tar.gz" not in policy


def test_receipt_probes_have_bucket_permission_to_distinguish_absence_from_denial():
    resources = template()["Resources"]
    for role in ("InferenceRole", "PreparationRole", "ControlRole"):
        statements = resources[role]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        listing = next(s for s in statements if s.get("Action") == ["s3:ListBucket"])
        assert listing["Resource"] == {"Fn::Sub": "arn:${AWS::Partition}:s3:::${ArtifactBucket}"}
        assert "Condition" not in listing  # GetObject does not supply a ListObjects prefix.


def test_lambda_zip_imports_in_an_isolated_process_without_ml_dependencies(tmp_path):
    output = tmp_path / "control.zip"
    package.package("a" * 40, output, root=ROOT)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(tmp_path / "runtime")
        metadata = json.loads(archive.read("maintenance-image.json"))
    assert "src/data/maintenance_build.py" in metadata["producer"]
    assert ".github/workflows/refresh-splits.yml" in metadata["producer"]
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from src.maintenance.control import handler
from src.scripts.advance_data_release import advance_release
assert not any(n in sys.modules for n in ('torch','pandas','pyarrow','nflreadpy'))
print('control import OK')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "runtime")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "control import OK"


def test_legacy_forecast_cron_does_not_duplicate_aws_schedule():
    doc = yaml.safe_load((ROOT / ".github/workflows/refresh-upcoming-week.yml").read_text())
    assert "AWS_MAINTENANCE_ACTIVE != 'true'" in doc["jobs"]["refresh"]["if"]
    condition = doc["jobs"]["aws_refresh"]["if"]
    assert "AWS_MAINTENANCE_ACTIVE == 'true'" in condition
    assert "github.event_name != 'schedule'" in condition


@pytest.mark.parametrize(
    "name,jobs",
    [
        ("deploy", ["deploy"]),
        ("refresh-splits", ["refresh", "rollout"]),
        ("train-batch", ["ecs_rollout"]),
        ("train-ec2", ["ecs_rollout"]),
    ],
)
def test_existing_rollouts_participate_in_the_shared_publication_lease(name, jobs):
    doc = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
    assert "FF_MAINTENANCE_LOCK_TABLE" in doc["env"]
    for job in jobs:
        steps = doc["jobs"][job]["steps"]
        acquired = next(
            i
            for i, s in enumerate(steps)
            if s.get("name") == "Acquire maintenance publication lease"
        )
        released = next(
            i
            for i, s in enumerate(steps)
            if s.get("name") == "Release maintenance publication lease"
        )
        assert acquired < released
        assert "always()" in steps[released]["if"]


def test_runtime_update_preserves_mode_schedules_and_network_and_rejects_old_deploy():
    parameters = {
        "ClusterName": "cluster",
        "ServiceName": "service",
        "WorkerImage": "old",
        "LambdaCodeKey": "old.zip",
        "Mode": "shadow",
        "EnableSchedules": "false",
        "Subnets": "subnet-one",
    }
    calls = []
    deployed = ["b" * 40]

    def call(service, op, args):
        calls.append((op, copy.deepcopy(args)))
        if op == "list-stacks":
            return {"StackSummaries": [{"StackName": "stack", "StackStatus": "CREATE_COMPLETE"}]}
        if op == "describe-stacks":
            return {
                "Stacks": [
                    {
                        "Parameters": [
                            {"ParameterKey": k, "ParameterValue": v} for k, v in parameters.items()
                        ]
                    }
                ]
            }
        if op == "describe-services":
            return {"services": [{"taskDefinition": "task"}]}
        if op == "describe-task-definition":
            return {
                "taskDefinition": {
                    "containerDefinitions": [
                        {"name": "fantasy-predictor", "image": "registry/app:" + deployed[0]}
                    ]
                }
            }
        return {}

    assert not update_runtime.update("stack", "image", "a" * 40, call=call)["updated"]
    deployed[0] = "a" * 40
    assert update_runtime.update("stack", "image", "a" * 40, call=call)["updated"]
    changed = next(args for op, args in calls if op == "update-stack")
    preserved = {x["ParameterKey"] for x in changed["Parameters"] if x.get("UsePreviousValue")}
    assert {"Mode", "EnableSchedules", "Subnets"} <= preserved
