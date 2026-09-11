"""Scheduling and recovery permissions are part of the availability contract."""

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_only_eligible_refreshes_share_the_pending_slot():
    workflow = yaml.safe_load((ROOT / ".github/workflows/refresh-upcoming-week.yml").read_text())
    # YAML 1.1 interprets the unquoted GitHub `on` key as True.
    triggers = workflow[True]
    assert triggers["schedule"] == [{"cron": "17 * * * *"}]
    assert "workflow_dispatch" in triggers
    eligibility = (
        "github.event_name != 'workflow_run' || github.event.workflow_run.conclusion == 'success'"
    )
    assert workflow["jobs"]["refresh"]["if"] == (
        "${{ vars.AWS_MAINTENANCE_ACTIVE != 'true' && (" + eligibility + ") }}"
    )
    assert workflow["jobs"]["aws_refresh"]["if"] == (
        "${{ vars.AWS_MAINTENANCE_ACTIVE == 'true' && github.event_name != 'schedule' && ("
        + eligibility
        + ") }}"
    )
    assert workflow["concurrency"]["group"] == (
        "${{ (" + eligibility + ") && 'refresh-upcoming-week' || "
        "format('refresh-upcoming-week-skipped-{0}', github.run_id) }}"
    )
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["concurrency"].get("queue", "single") == "single"


def test_version_recovery_permission_is_scoped_to_the_upcoming_snapshot():
    policy = json.loads((ROOT / "infra/aws/task-role-policy.json").read_text())
    statements = {statement["Sid"]: statement for statement in policy["Statement"]}
    read = statements["ReadUpcomingArtifactVersions"]
    assert read["Effect"] == "Allow" and read["Action"] == ["s3:GetObjectVersion"]
    key = "models/predictions_cache/upcoming_week.json"
    bucket = "arn:aws:s3:::ff-predictor-training"
    assert read["Resource"] == f"{bucket}/{key}"
    listing = statements["ListUpcomingArtifactVersions"]
    assert listing["Effect"] == "Allow" and listing["Action"] == ["s3:ListBucketVersions"]
    assert listing["Resource"] == bucket
    assert listing["Condition"] == {"StringEquals": {"s3:prefix": key}}
