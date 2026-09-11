"""Resolve the actual training image before checking its data compatibility.

Read-only: snapshots existing Batch revisions, or an ECR digest for EC2. Manual
Batch dispatch derives its source SHA from the selected revision's image instead
of labeling whatever is latest with the workflow checkout's unrelated SHA.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import boto3

_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _image_sha(definition: dict) -> str:
    image = definition.get("containerProperties", {}).get("image", "")
    sha = image.rsplit(":", 1)[-1]
    if "@" in image or not _SHA.fullmatch(sha):
        raise ValueError(f"Batch image must have an explicit full source-SHA tag: {image!r}")
    return sha


def _revision(s3, bucket: str, sha: str, *, cpu: bool = False) -> int:
    key = f"job-def-revisions/{'cpu/' if cpu else ''}{sha}.txt"
    value = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"Invalid Batch revision at s3://{bucket}/{key}")
    return int(value)


def _definition(batch, name: str, revision: int) -> dict:
    # AWS forbids combining jobDefinitions with the status filter; verify the
    # returned revision's state locally instead.
    values = batch.describe_job_definitions(jobDefinitions=[f"{name}:{revision}"])
    matches = [
        d
        for d in values["jobDefinitions"]
        if d["jobDefinitionName"] == name and d["revision"] == revision and d["status"] == "ACTIVE"
    ]
    if len(matches) != 1:
        raise ValueError(f"Batch revision is unavailable: {name}:{revision}")
    return matches[0]


def resolve_definition(batch, identifier: str, *, include_image=False) -> dict[str, str]:
    """Read one already-selected immutable Batch revision and its source tag."""
    values = batch.describe_job_definitions(jobDefinitions=[identifier]).get("jobDefinitions", [])
    active = [value for value in values if value.get("status") == "ACTIVE"]
    if len(active) != 1:
        raise ValueError(f"Batch definition is not one active revision: {identifier}")
    value = active[0]
    result = {
        "image_sha": _image_sha(value),
        "job_definition": value.get("jobDefinitionArn")
        or f"{value['jobDefinitionName']}:{value['revision']}",
    }
    if include_image:
        result["image"] = value["containerProperties"]["image"]
    return result


def resolve_batch(
    batch,
    s3,
    bucket: str,
    *,
    sha: str = "",
    split: bool = False,
    name: str = "ff-training-job",
    cpu_name: str = "ff-training-cpu-job",
    revision: str = "",
    cpu_revision: str = "",
    primary_cpu: bool = False,
    include_definition: bool = False,
) -> dict[str, str]:
    if revision:
        if not str(revision).isdigit() or int(revision) < 1:
            raise ValueError("Batch revision must be a positive integer")
        definition = _definition(batch, name, int(revision))
    elif sha:
        if not _SHA.fullmatch(sha):
            raise ValueError("Training image SHA must be a full 40-character commit")
        definition = _definition(batch, name, _revision(s3, bucket, sha, cpu=primary_cpu))
    else:
        definitions = [
            d
            for page in batch.get_paginator("describe_job_definitions").paginate(
                jobDefinitionName=name, status="ACTIVE"
            )
            for d in page["jobDefinitions"]
        ]
        if not definitions:
            raise ValueError(f"No active Batch definition for {name}")
        definition = max(definitions, key=lambda d: d["revision"])
    actual_sha = _image_sha(definition)
    if sha and actual_sha != sha:
        raise ValueError(f"Batch image source {actual_sha} differs from requested {sha}")
    result = {"image_sha": actual_sha, "revision": str(definition["revision"]), "cpu_revision": ""}
    if include_definition:
        result.update(
            job_definition=definition.get("jobDefinitionArn") or f"{name}:{definition['revision']}",
            image=definition["containerProperties"]["image"],
        )
    if split:
        revision = (
            int(cpu_revision) if cpu_revision else _revision(s3, bucket, actual_sha, cpu=True)
        )
        cpu = _definition(batch, cpu_name, revision)
        if _image_sha(cpu) != actual_sha:
            raise ValueError("GPU and CPU Batch revisions use different source images")
        result["cpu_revision"] = str(revision)
        if include_definition:
            result["cpu_job_definition"] = cpu.get("jobDefinitionArn") or f"{cpu_name}:{revision}"
            result["cpu_image"] = cpu["containerProperties"]["image"]
    return result


def resolve_ec2(ecr, sha: str = "", *, repository: str = "ff-training") -> dict[str, str]:
    if sha and not _SHA.fullmatch(sha):
        raise ValueError("Training image SHA must be a full 40-character commit")
    details = ecr.describe_images(
        repositoryName=repository, imageIds=[{"imageTag": sha or "latest"}]
    )["imageDetails"]
    if len(details) != 1 or not _DIGEST.fullmatch(details[0].get("imageDigest", "")):
        raise ValueError(f"No unambiguous ECR digest for source {sha}")
    if not sha:
        tags = {tag for tag in details[0].get("imageTags", []) if _SHA.fullmatch(tag)}
        if len(tags) != 1:
            raise ValueError("Current ECR image lacks one unambiguous source SHA; pass image_sha")
        sha = tags.pop()
    uri = ecr.describe_repositories(repositoryNames=[repository])["repositories"][0][
        "repositoryUri"
    ]
    return {"image_sha": sha, "image_uri": f"{uri}@{details[0]['imageDigest']}"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("batch", "ec2"))
    parser.add_argument("--sha", default="")
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    args = parser.parse_args(argv)
    if args.backend == "batch":
        result = resolve_batch(
            boto3.client("batch", region_name=args.region),
            boto3.client("s3", region_name=args.region),
            args.bucket,
            sha=args.sha,
            split=args.split,
        )
    else:
        result = resolve_ec2(boto3.client("ecr", region_name=args.region), args.sha)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in result.items())
    print(
        "Resolved training image: " + ", ".join(f"{key}={value}" for key, value in result.items())
    )


if __name__ == "__main__":
    main()
