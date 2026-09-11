"""Exercise startup permissions and traffic readiness using configured callers."""

import fnmatch
import json
from pathlib import Path

import pytest
import yaml
from botocore.exceptions import ClientError

from src.artifacts import model_sync

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_actual_history_sync_request_is_allowed_by_serving_task_policy(monkeypatch, tmp_path):
    policy = json.loads((ROOT / "infra/aws/task-role-policy.json").read_text())
    listing = next(
        statement for statement in policy["Statement"] if "s3:ListBucket" in statement["Action"]
    )
    patterns = listing["Condition"]["StringLike"]["s3:prefix"]
    requested = []

    class PolicyS3:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, Bucket, Prefix):  # noqa: N803
            requested.append(Prefix)
            if not any(fnmatch.fnmatchcase(Prefix, pattern) for pattern in patterns):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": Prefix}}, "ListObjectsV2"
                )
            yield {"Contents": []}

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "ff-predictor-training")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr("boto3.client", lambda *_: PolicyS3())
    assert model_sync.sync_benchmark_history_from_s3()["files"] == 0
    assert requested == ["models/benchmark_history/"]
    assert not any(fnmatch.fnmatchcase("data/raw/", pattern) for pattern in patterns)


def test_alb_probe_excludes_a_live_but_not_ready_worker(monkeypatch, tmp_path):
    from src.serving import core, state
    from src.serving.app import create_app

    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "empty-cache"))
    application = create_app(
        serving_state=state.ServingState(), config={"ALLOW_RUNTIME_INFERENCE": False}
    )
    client = application.test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503

    from src.artifacts.deployment import deploy
    from tests.artifacts.test_deployment_rollout import RolloutAWS, task_definition

    aws = RolloutAWS(response_code=503)
    with pytest.raises(RuntimeError, match="timeout"):
        deploy(
            aws,
            task_definition(),
            cluster="cluster",
            service="service",
            state_path=tmp_path / "rollout.json",
            timeout=0,
        )
    updates = [
        request for _, operation, request in aws.events if operation == "modify-target-group"
    ]
    compatibility = updates[0]
    response = client.get(compatibility["HealthCheckPath"])
    assert str(response.status_code) not in compatibility["Matcher"]["HttpCode"].split(",")
    assert compatibility["HealthCheckPath"] == "/health?readiness=1"
    assert compatibility["Matcher"] == {"HttpCode": "200"}

    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    deployment_step = next(
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Deploy to ECS"
    )
    assert "src.artifacts.deployment deploy" in deployment_step["run"]
