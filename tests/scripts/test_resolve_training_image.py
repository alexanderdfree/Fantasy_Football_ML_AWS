"""Read-only image resolution pins the code whose producer contract is checked."""

import io
from unittest.mock import Mock

import pytest

from src.scripts.resolve_training_image import main, resolve_batch, resolve_ec2

pytestmark = pytest.mark.unit
SHA_A = "a" * 40
SHA_B = "b" * 40
DIGEST = "sha256:" + "1" * 64
REPO = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ff-training"


def definition(revision, sha=SHA_A, name="ff-training-job"):
    return {
        "jobDefinitionName": name,
        "revision": revision,
        "status": "ACTIVE",
        "containerProperties": {"image": f"{REPO}:{sha}"},
    }


def test_manual_batch_pins_actual_current_revision_when_latest_changes(monkeypatch):
    from src.batch import launch

    batch = Mock()
    batch.get_paginator.return_value.paginate.return_value = [
        {"jobDefinitions": [definition(3, SHA_B)]},
        {"jobDefinitions": [definition(14)]},
    ]
    resolved = resolve_batch(batch, Mock(), "bucket")
    # A new build arriving after verification must not change this argument.
    batch.get_paginator.return_value.paginate.return_value = [
        {"jobDefinitions": [definition(15, SHA_B)]}
    ]
    assert resolved == {"image_sha": SHA_A, "revision": "14", "cpu_revision": ""}
    monkeypatch.setattr(launch, "JOB_DEFINITION", "ff-training-job")
    monkeypatch.setattr(launch, "JOB_DEFINITION_REVISION", resolved["revision"])
    monkeypatch.setattr(launch, "TRAIN_GIT_SHA", resolved["image_sha"])
    monkeypatch.setattr(launch, "data_release_environment", lambda: [])
    batch.submit_job.return_value = {"jobId": "pinned-job"}
    launch.submit_job("QB", branch="nn", split_run_id="pinned-run", batch_client=batch)
    call = batch.submit_job.call_args.kwargs
    assert call["jobDefinition"] == "ff-training-job:14"
    assert {"name": "FF_TRAIN_GIT_SHA", "value": SHA_A} in call["containerOverrides"]["environment"]


def test_split_cpu_pin_matches_resolved_image_not_latest_cpu_revision():
    batch, s3 = Mock(), Mock()
    batch.get_paginator.return_value.paginate.return_value = [{"jobDefinitions": [definition(14)]}]
    s3.get_object.return_value = {"Body": io.BytesIO(b"8")}
    batch.describe_job_definitions.return_value = {
        "jobDefinitions": [definition(8, name="ff-training-cpu-job")]
    }
    resolved = resolve_batch(batch, s3, "bucket", split=True)
    assert resolved["cpu_revision"] == "8"
    s3.get_object.assert_called_once_with(Bucket="bucket", Key=f"job-def-revisions/cpu/{SHA_A}.txt")
    batch.describe_job_definitions.assert_called_once_with(jobDefinitions=["ff-training-cpu-job:8"])


@pytest.mark.parametrize("image_sha", [SHA_B, "latest"])
def test_requested_revision_must_contain_the_verified_sha(image_sha):
    batch, s3 = Mock(), Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(b"14")}
    batch.describe_job_definitions.return_value = {"jobDefinitions": [definition(14, image_sha)]}
    with pytest.raises(ValueError, match="source|source-SHA"):
        resolve_batch(batch, s3, "bucket", sha=SHA_A)


def test_split_rejects_cpu_image_different_from_gpu():
    batch, s3 = Mock(), Mock()
    batch.get_paginator.return_value.paginate.return_value = [{"jobDefinitions": [definition(14)]}]
    s3.get_object.return_value = {"Body": io.BytesIO(b"8")}
    batch.describe_job_definitions.return_value = {
        "jobDefinitions": [definition(8, SHA_B, name="ff-training-cpu-job")]
    }
    with pytest.raises(ValueError, match="different source"):
        resolve_batch(batch, s3, "bucket", split=True)


def test_ec2_resolves_requested_sha_to_digest_without_reading_latest():
    ecr = Mock()
    ecr.describe_images.return_value = {"imageDetails": [{"imageDigest": DIGEST}]}
    ecr.describe_repositories.return_value = {"repositories": [{"repositoryUri": REPO}]}
    assert resolve_ec2(ecr, SHA_A) == {"image_sha": SHA_A, "image_uri": f"{REPO}@{DIGEST}"}
    ecr.describe_images.assert_called_once_with(
        repositoryName="ff-training", imageIds=[{"imageTag": SHA_A}]
    )


def test_ec2_unavailable_digest_fails_before_training():
    ecr = Mock()
    ecr.describe_images.return_value = {"imageDetails": []}
    with pytest.raises(ValueError, match="unambiguous"):
        resolve_ec2(ecr, SHA_A)


def test_manual_ec2_derives_source_from_current_image_and_retains_digest_if_latest_moves():
    ecr = Mock()
    ecr.describe_images.return_value = {
        "imageDetails": [{"imageDigest": DIGEST, "imageTags": ["latest", SHA_A]}]
    }
    ecr.describe_repositories.return_value = {"repositories": [{"repositoryUri": REPO}]}
    resolved = resolve_ec2(ecr)
    ecr.describe_images.return_value = {
        "imageDetails": [{"imageDigest": "sha256:" + "2" * 64, "imageTags": ["latest", SHA_B]}]
    }
    assert resolved == {"image_sha": SHA_A, "image_uri": f"{REPO}@{DIGEST}"}


def test_manual_ec2_requires_explicit_sha_if_current_digest_has_ambiguous_source_tags():
    ecr = Mock()
    ecr.describe_images.return_value = {
        "imageDetails": [{"imageDigest": DIGEST, "imageTags": ["latest", SHA_A, SHA_B]}]
    }
    with pytest.raises(ValueError, match="pass image_sha"):
        resolve_ec2(ecr)


def test_cli_forwards_verified_pin_to_workflow_output(monkeypatch, tmp_path):
    ecr = Mock()
    ecr.describe_images.return_value = {"imageDetails": [{"imageDigest": DIGEST}]}
    ecr.describe_repositories.return_value = {"repositories": [{"repositoryUri": REPO}]}
    monkeypatch.setattr("src.scripts.resolve_training_image.boto3.client", lambda *a, **kw: ecr)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["ec2", "--sha", SHA_A])
    assert output.read_text().splitlines() == [
        f"image_sha={SHA_A}",
        f"image_uri={REPO}@{DIGEST}",
    ]
