"""The serving role consumes published artifacts; publishers use separate roles."""

import json
from pathlib import Path

import pytest

from src.artifacts.deployment import render_task_definition

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_serving_role_can_read_snapshots_and_history_without_write_access():
    policy = json.loads((ROOT / "infra/aws/task-role-policy.json").read_text())
    statements = policy["Statement"]
    reads = [s for s in statements if "s3:GetObject" in s["Action"]]
    resources = {
        resource
        for s in reads
        for resource in ([s["Resource"]] if isinstance(s["Resource"], str) else s["Resource"])
    }
    assert "arn:aws:s3:::ff-predictor-training/models/*" in resources
    listing = next(s for s in statements if "s3:ListBucket" in s["Action"])
    assert listing["Condition"]["StringLike"]["s3:prefix"] == [
        "models/benchmark_history/",
        "models/benchmark_history/*",
    ]
    assert {action for s in statements for action in s["Action"]} <= {
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetObjectVersion",
        "s3:ListBucketVersions",
    }
    versions = next(s for s in statements if s["Sid"] == "ReadUpcomingArtifactVersions")
    assert versions["Resource"].endswith("/models/predictions_cache/upcoming_week.json")
    listing = next(s for s in statements if s["Sid"] == "ListUpcomingArtifactVersions")
    assert listing["Condition"] == {
        "StringEquals": {"s3:prefix": "models/predictions_cache/upcoming_week.json"}
    }


def test_production_runtime_and_deployment_require_ready_artifacts():
    docker = (ROOT / "Dockerfile").read_text()
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
    assert "ENV FF_ALLOW_RUNTIME_INFERENCE=0" in docker
    assert "http://localhost:8000/ready" in docker
    assert workflow.index("serving_snapshot wait") < workflow.index("- name: Deploy to ECS")


def test_deployment_carries_readiness_and_preserves_live_secrets_and_arns():
    desired = json.loads((ROOT / "infra/aws/task-definition.json").read_text())
    live = json.loads(json.dumps(desired))
    live["taskRoleArn"] = "actual-role"
    live["revision"] = 12
    container = live["containerDefinitions"][0]
    container["image"] = "existing-image"
    container["healthCheck"]["command"] = ["CMD-SHELL", "old-health"]
    container["secrets"] = [{"name": "TOKEN", "valueFrom": "actual-secret-arn"}]
    container["environment"] = [
        {"name": "FF_MODEL_S3_BUCKET", "value": "real-bucket"},
        {"name": "FF_ALLOW_RUNTIME_INFERENCE", "value": "1"},
    ]
    result = render_task_definition(live, desired)
    output = result["containerDefinitions"][0]
    assert result["taskRoleArn"] == "actual-role" and "revision" not in result
    assert output["image"] == "existing-image"
    assert output["secrets"] == container["secrets"]
    assert output["healthCheck"] == desired["containerDefinitions"][0]["healthCheck"]
    assert "/ready" in output["healthCheck"]["command"][1]
    assert {v["name"]: v["value"] for v in output["environment"]} == {
        "FF_MODEL_S3_BUCKET": "real-bucket",
        "FF_ALLOW_RUNTIME_INFERENCE": "0",
    }
    assert container["healthCheck"]["command"][1] == "old-health"
