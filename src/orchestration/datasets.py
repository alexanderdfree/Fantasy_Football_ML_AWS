"""Content-addressed training datasets and explicit source-to-dataset selection.

The source reference is mutable so a scheduled refresh can publish newer data
for unchanged code. Once selected, a dataset ID names an immutable manifest and
every object is checked against its recorded SHA256 before training begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from src.data.release import DATA_PRODUCER_PATHS as SOURCE_PATHS

SCHEMA_VERSION = 1
SPLITS = ("train.parquet", "val.parquet", "test.parquet")
# Kept in sync with refresh-splits.yml push.paths by a contract test. These
# cover the actual regeneration imports and its schema-verification inputs.

DATA_RELEASE_FORMAT = "data-release-v1"
LEGACY_DATASET_FORMAT = "dataset-v1"


class DatasetError(RuntimeError):
    """A requested dataset is unavailable, unidentified or inconsistent."""


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def content_id(value: dict) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetError(f"Invalid {label}: expected a full SHA256 identifier")
    return value


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(repo: Path, revision: str = "HEAD") -> str:
    """Use the canonical sealed-release recipe identity for the selected image."""
    from src.data.release import producer_fingerprint
    from src.scripts.wait_data_release import producer_hashes_at_revision

    return producer_fingerprint(producer_hashes_at_revision(revision, repo_root=repo))


def assert_source_checkout(repo: Path, revision: str) -> None:
    changed = subprocess.run(["git", "diff", "--quiet", revision, "--", *SOURCE_PATHS], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS], cwd=repo
    )
    if changed.returncode or untracked.strip():
        raise DatasetError("Dataset source checkout differs from the declared producing revision")


def _missing(error: Exception) -> bool:
    from botocore.exceptions import ClientError

    return isinstance(error, ClientError) and error.response.get("Error", {}).get("Code") in {
        "NoSuchKey",
        "404",
        "NotFound",
    }


def read_json(s3, bucket: str, key: str) -> dict:
    document = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if not isinstance(document, dict):
        raise DatasetError(f"Expected a JSON object at {key}")
    return document


def put_immutable_json(s3, bucket: str, key: str, document: dict) -> None:
    """Idempotent create; a conflicting body can never replace an immutable ID."""
    from botocore.exceptions import ClientError

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=canonical_bytes(document),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        if read_json(s3, bucket, key) != document:
            raise DatasetError(f"Conflicting immutable document at {key}") from error


def _validate_path(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise DatasetError(f"Unsafe dataset destination: {value!r}")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or len(path.parts) < 3
        or path.parts[0] != "data"
        or path.parts[1] not in {"raw", "splits"}
        or (path.parts[1] == "splits" and (len(path.parts) != 3 or path.name not in SPLITS))
    ):
        raise DatasetError(f"Unsafe dataset destination: {value!r}")
    return path


def validate_manifest(document: dict, dataset_id: str) -> dict:
    require_id(dataset_id, "dataset ID")
    if content_id(document) != dataset_id or document.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("Dataset manifest identity/schema mismatch")
    require_id(document.get("source_id"), "source ID")
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise DatasetError("Dataset manifest has no files")
    seen: set[str] = set()
    for entry in entries:
        path = str(_validate_path(entry.get("path")))
        digest = require_id(entry.get("sha256"), "object digest")
        if path in seen or entry.get("key") != f"datasets/objects/{digest}":
            raise DatasetError("Duplicate destination or non-addressed dataset object")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            raise DatasetError("Invalid dataset object size")
        seen.add(path)
    if not {f"data/splits/{name}" for name in SPLITS}.issubset(seen):
        raise DatasetError("Dataset lacks required train/val/test splits")
    if not any(path.startswith("data/raw/") for path in seen):
        raise DatasetError("Dataset lacks raw inputs required by K/DST and weather features")
    return document


def load_dataset(s3, bucket: str, dataset_id: str, *, data_format=DATA_RELEASE_FORMAT) -> dict:
    """Read one explicitly identified format; never infer a second authority."""
    require_id(dataset_id, "dataset ID")
    if data_format == LEGACY_DATASET_FORMAT:
        return validate_manifest(
            read_json(s3, bucket, f"datasets/manifests/{dataset_id}.json"), dataset_id
        )
    if data_format != DATA_RELEASE_FORMAT:
        raise DatasetError(f"Unsupported dataset format: {data_format}")
    from src.data.release import resolve_release

    try:
        selected, manifest = resolve_release(s3, bucket, release_id=dataset_id)
    except ValueError as error:
        raise DatasetError(str(error)) from error
    source_id = require_id(manifest.get("data_producer_sha256"), "producer identity")
    return {
        "schema_version": 1,
        "data_format": DATA_RELEASE_FORMAT,
        "release_id": selected,
        "source_id": source_id,
        "release_manifest": manifest,
        "files": [
            {"path": f"data/{name}", "key": f"data/releases/{selected}/{name}", **info}
            for name, info in manifest["files"].items()
        ],
    }


def publish_dataset(s3, bucket: str, repo: Path, source_id: str, *, workers: int = 4) -> str:
    """Compatibility entrypoint for the sole sealed-release publisher.

    It never writes datasets/sources or silently seals arbitrary local files.
    The producer must finish all inputs and seal them before calling upload.
    """
    from src.data.release import data_producer_hashes, producer_fingerprint, publish_release

    require_id(source_id, "source ID")
    if producer_fingerprint(data_producer_hashes(repo)) != source_id:
        raise DatasetError("Declared dataset producer differs from the local source")
    return publish_release(
        s3, bucket, raw_dir=repo / "data/raw", splits_dir=repo / "data/splits", repo_root=repo
    )


def select_dataset(
    s3,
    bucket: str,
    source_id: str,
    *,
    expected_hashes=None,
    revision=None,
    timeout: float = 1200,
    poll: float = 15,
    clock=time.monotonic,
    sleep=time.sleep,
) -> str:
    """Select one release from the canonical producer index; missing fails closed."""
    from src.data.release import producer_fingerprint, resolve_release

    require_id(source_id, "source ID")
    if expected_hashes is not None and producer_fingerprint(expected_hashes) != source_id:
        raise DatasetError("Expected producer hashes differ from the selected source identity")
    if expected_hashes is not None:
        from src.scripts.wait_data_release import wait_for_release

        class Reader:
            def get_object(self, **kwargs):
                try:
                    return s3.get_object(**kwargs)
                except Exception as error:
                    if _missing(error):
                        raise FileNotFoundError(kwargs["Key"]) from error
                    raise

        return wait_for_release(
            Reader(),
            bucket,
            revision=revision or source_id,
            expected_hashes=expected_hashes,
            timeout=timeout,
            interval=poll,
            clock=clock,
            sleep=sleep,
        )
    deadline = clock() + timeout
    while True:
        try:
            pointer = read_json(s3, bucket, f"data/by-producer/{source_id}/manifest.json")
            if pointer.get("schema_version") != 1:
                raise DatasetError("Invalid data release producer index")
            selected, manifest = resolve_release(
                s3, bucket, release_id=require_id(pointer.get("release_id"), "data release ID")
            )
            if manifest.get("data_producer_sha256") != source_id:
                raise DatasetError("Selected release was built from a different producer")
            return selected
        except Exception as error:
            if not _missing(error):
                raise
        remaining = deadline - clock()
        if remaining <= 0:
            raise DatasetError(
                f"No completed data release for source {source_id}; refresh-splits must complete "
                "before training. Mutable data/ fallback is disabled."
            )
        sleep(min(poll, remaining))


def materialize_dataset(
    s3,
    bucket: str,
    dataset_id: str,
    *,
    splits_dir: Path,
    raw_dir: Path,
    workers: int = 4,
    data_format=DATA_RELEASE_FORMAT,
) -> dict:
    """Verify every snapshot object before replacing isolated training inputs."""
    document = load_dataset(s3, bucket, dataset_id, data_format=data_format)
    for destination in (splits_dir, raw_dir):
        if destination.is_symlink():
            raise DatasetError(f"Refusing to replace a symlinked data destination: {destination}")
    if data_format == DATA_RELEASE_FORMAT:
        from src.data.release import download_release

        try:
            download_release(
                s3, bucket, raw_dir=raw_dir, splits_dir=splits_dir, release_id=dataset_id
            )
        except ValueError as error:
            raise DatasetError(str(error)) from error
        return document
    with tempfile.TemporaryDirectory(prefix="dataset-verified-") as temp:
        stage = Path(temp)

        def download(entry: dict) -> None:
            target = stage / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, entry["key"], str(target))
            if target.stat().st_size != entry["bytes"] or file_digest(target) != entry["sha256"]:
                raise DatasetError(f"Dataset object checksum mismatch: {entry['path']}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(download, document["files"]))
        # Called before any loader/model. A failure aborts the job; nothing can
        # train on an unverified or partially downloaded snapshot.
        for name, destination in (("splits", splits_dir), ("raw", raw_dir)):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(stage / "data" / name), str(destination))
    return document


def bind_data_release(release_id: str) -> str:
    """Bind the canonical release alias once; conflicting selections fail closed."""
    dataset_id = os.environ.get("FF_DATASET_ID")
    selected = os.environ.get("FF_DATA_RELEASE")
    if release_id == "legacy":
        if dataset_id or os.environ.get("FF_REQUIRE_BUILD_PLAN") == "1":
            raise DatasetError(
                "Mutable legacy data cannot satisfy an identified dataset/build plan"
            )
    else:
        require_id(release_id, "data release ID")
        if dataset_id and dataset_id != release_id:
            raise DatasetError("FF_DATA_RELEASE and FF_DATASET_ID disagree")
    if selected and selected != release_id:
        raise DatasetError("A different data release was already selected")
    os.environ["FF_DATA_RELEASE"] = release_id
    if release_id != "legacy":
        os.environ["FF_DATASET_ID"] = release_id
        os.environ["FF_DATA_FORMAT"] = DATA_RELEASE_FORMAT
    return release_id


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "publish", "select"])
    parser.add_argument("--bucket")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        from src.data.release import prewarm_training_dependencies

        prewarm_training_dependencies()
        return
    if not args.bucket:
        parser.error("--bucket is required for publish/select")
    import boto3

    s3 = boto3.client("s3")
    source_id = source_identity(args.repo, args.revision)
    if args.action == "publish":
        assert_source_checkout(args.repo, args.revision)
        dataset_id = publish_dataset(s3, args.bucket, args.repo, source_id)
    else:
        from src.scripts.wait_data_release import producer_hashes_at_revision

        dataset_id = select_dataset(
            s3,
            args.bucket,
            source_id,
            timeout=args.timeout,
            expected_hashes=producer_hashes_at_revision(args.revision, repo_root=args.repo),
        )
    result = {
        "source_id": source_id,
        "dataset_id": dataset_id,
        "data_release": dataset_id,
        "data_format": DATA_RELEASE_FORMAT,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"dataset_id={dataset_id}\nsource_id={source_id}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
