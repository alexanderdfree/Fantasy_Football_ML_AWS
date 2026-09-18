"""Recover hash-verified PR1564 inputs without changing a production pointer."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

BUCKET = "ff-predictor-training"
FIXED_RELEASE = "556115711494d5f7c10af9fe3e97b94b200e97a661078a3a1e4a874ca4b20340"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def relative_path(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe artifact name: {name}")
    return Path(*value.parts)


def recover_object(s3, bucket: str, task: dict, output: Path) -> dict:
    """Verify bytes before installing; retain exact S3 object identity."""
    request = {"Bucket": bucket, "Key": task["key"]}
    if task.get("version") is not None:
        request["VersionId"] = task["version"]
    record = dict(task)
    try:
        response = s3.get_object(**request)
        raw = response["Body"].read()
        actual = digest(raw)
        record.update(
            actual_sha256=actual,
            actual_bytes=len(raw),
            resolved_version=response.get("VersionId"),
            modified=str(response.get("LastModified")),
        )
        if actual != task["sha256"]:
            raise ValueError("Archived content hash does not match recovered bytes")
        if len(raw) != task["bytes"]:
            raise ValueError("Archived size does not match recovered bytes")
        path = output / relative_path(task["destination"])
        if path.exists() and digest(path.read_bytes()) != actual:
            raise ValueError("Refusing to overwrite different local evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        record["ok"] = True
    except Exception as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}")
    return record


def recover(s3, archive: Path, output: Path, bucket=BUCKET, release=FIXED_RELEASE) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    versions = {
        item["key"]: item["version"]
        for item in json.loads((archive / "raw_s3_versions.json").read_text())
    }
    tasks = []
    for item in json.loads((archive / "s3_manifest.json").read_text()):
        # The initial audit's split copies predated #1551. Its follow-up
        # latest_s3_splits.json explicitly replaces those three observations.
        if not item["Key"].startswith("data/raw/"):
            continue
        tasks.append(
            dict(
                arm="archived_baseline",
                key=item["Key"],
                version=versions.get(item["Key"]),
                bytes=item["Size"],
                sha256=item["sha256"],
                destination="baseline/" + item["Key"],
            )
        )
    for item in json.loads((archive / "latest_s3_splits.json").read_text()):
        key = f"data/{item['split']}.parquet"
        head = s3.head_object(Bucket=bucket, Key=key, VersionId=item["version"])
        tasks.append(
            dict(
                arm="archived_baseline",
                key=key,
                version=item["version"],
                bytes=head["ContentLength"],
                sha256=item["sha256"],
                destination=f"baseline/data/splits/{item['split']}.parquet",
            )
        )
    key = f"data/releases/{release}/manifest.json"
    response = s3.get_object(Bucket=bucket, Key=key)
    raw = response["Body"].read()
    if digest(raw) != release:
        raise ValueError("Fixed release manifest is not content-addressed by its ID")
    manifest = json.loads(raw)
    (output / "fixed-release-manifest.json").write_bytes(raw)
    for name, metadata in manifest["files"].items():
        tasks.append(
            dict(
                arm="published_fixed",
                key=f"data/releases/{release}/{name}",
                version=None,
                bytes=metadata["bytes"],
                sha256=metadata["sha256"],
                destination=f"fixed/data/{name}",
            )
        )
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda task: recover_object(s3, bucket, task, output), tasks))
    report = {
        "bucket": bucket,
        "fixed_release": release,
        "fixed_source": manifest["git_sha"],
        "records": records,
        "recovered": sum(record["ok"] for record in records),
        "missing": [record["destination"] for record in records if not record["ok"]],
        "scope": "Archived production snapshot; original local CPU experiment identity unverified",
    }
    (output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import boto3

    report = recover(boto3.client("s3"), args.archive, args.output)
    print(json.dumps({key: report[key] for key in ("recovered", "missing", "fixed_source")}))
    if report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
