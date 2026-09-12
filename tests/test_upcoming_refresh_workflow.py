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


@pytest.mark.parametrize("corrupt_capture", [False, True])
def test_builder_materializes_every_pinned_release_file_before_models(
    monkeypatch, tmp_path, corrupt_capture
):
    import boto3

    from src.artifacts import model_sync
    from src.orchestration.datasets import DatasetError
    from tests.orchestration.test_datasets import MemoryS3, write_release

    workflow = yaml.safe_load((ROOT / ".github/workflows/refresh-upcoming-week.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["refresh"]["steps"]
        if step.get("name") == "Build + upload upcoming-week artifact"
    )
    code = step["run"].split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    s3 = MemoryS3()
    capture = f"raw/provider_sources/{'a' * 64}"
    files = {
        **{f"splits/{name}.parquet": name.encode() for name in ("train", "val", "test")},
        "raw/weekly.parquet": b"selected historical rows",
        "raw/weekly_evaluation_reference_v1.parquet": b"selected reference",
        f"{capture}.json": b'{"status":"observed"}',
        f"{capture}.parquet": b"selected provider response",
    }
    release_id = write_release(s3, files=files)
    # Legacy objects cannot replace any part of the explicitly selected release.
    s3.objects["data/train.parquet"] = b"stale mutable split"
    s3.objects["data/raw/weekly.parquet"] = b"stale mutable raw rows"
    if corrupt_capture:
        s3.objects[f"data/releases/{release_id}/{capture}.parquet"] = b"corrupt"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "bucket")
    monkeypatch.setenv("FF_DATA_RELEASE", release_id)
    monkeypatch.setattr(boto3, "client", lambda name: s3)
    synced = []

    def sync_models():
        for name, body in files.items():
            assert (tmp_path / "data" / name).read_bytes() == body
        marker = json.loads((tmp_path / "data/raw/.release.json").read_text())
        assert marker == {"release_id": release_id, "provider_sources": "captured"}
        synced.append(True)

    monkeypatch.setattr(model_sync, "sync_models_from_s3", sync_models)
    if corrupt_capture:
        with pytest.raises(DatasetError, match="checksum mismatch"):
            exec(compile(code, str(ROOT / ".github/workflows/refresh-upcoming-week.yml"), "exec"))
        assert not synced
        assert not (tmp_path / "data/raw/.release.json").exists()
    else:
        exec(compile(code, str(ROOT / ".github/workflows/refresh-upcoming-week.yml"), "exec"))
        assert synced == [True]
