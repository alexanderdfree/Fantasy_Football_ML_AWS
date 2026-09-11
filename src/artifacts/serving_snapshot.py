"""Publish and consume one immutable generation of the historical serving cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.artifacts.model_sync import load_manifest_snapshot

FILES = ("predictions.parquet", "metrics.json", "fingerprint.json", "snapshot.json")
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
CACHE_SCHEMA_VERSION = 11


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(s3, bucket, key):
    from botocore.exceptions import ClientError

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return None, None
        raise
    return json.loads(response["Body"].read()), response["ETag"]


def _is_digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_pointer(pointer, prefix="models"):
    """Resolve only a supported pointer inside the configured generation prefix."""
    if (
        not isinstance(pointer, dict)
        or type(pointer.get("schema_version")) is not int
        or pointer.get("schema_version") != 1
        or not _is_digest(pointer.get("generation"))
    ):
        raise ValueError("Invalid serving snapshot pointer")
    generation = pointer["generation"]
    base = f"{prefix}/predictions_cache/generations/{generation}"
    if pointer.get("manifest") != f"{base}/manifest.json":
        raise ValueError("Serving snapshot manifest key mismatch")
    return generation, base


def validate_generation(pointer, manifest, prefix="models"):
    """Shared compatibility and identity contract for sync and deployment gates."""
    generation, base = validate_pointer(pointer, prefix)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or type(manifest.get("cache_schema_version")) is not int
        or manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
    ):
        raise ValueError("Serving snapshot cache schema is incompatible")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError("Serving snapshot file inventory mismatch")
    for name, metadata in files.items():
        if (
            not isinstance(metadata, dict)
            or type(metadata.get("bytes")) is not int
            or metadata["bytes"] < 0
            or not _is_digest(metadata.get("sha256"))
        ):
            raise ValueError(f"Invalid serving snapshot file metadata: {name}")
    if hashlib.sha256(_bytes(manifest)).hexdigest() != generation:
        raise ValueError("Serving snapshot identity mismatch")
    return generation, base


def validate_payload(name, payload, metadata):
    if (
        payload is None
        or len(payload) != metadata["bytes"]
        or hashlib.sha256(payload).hexdigest() != metadata["sha256"]
    ):
        raise ValueError(f"Serving snapshot checksum mismatch: {name}")


def verify_remote(s3, bucket, prefix="models", *, expected_dataset_id=None, generation=None):
    """Verify one captured release and every byte before any deployment mutation."""

    def read(key):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as error:
            if getattr(error, "response", {}).get("Error", {}).get("Code") in {
                "NoSuchKey",
                "404",
                "NotFound",
            }:
                raise FileNotFoundError(key) from error
            raise

    if generation is None:
        pointer = json.loads(read(f"{prefix}/predictions_cache/current.json"))
    else:
        pointer = {
            "schema_version": 1,
            "generation": generation,
            "manifest": f"{prefix}/predictions_cache/generations/{generation}/manifest.json",
        }
    validate_pointer(pointer, prefix)
    manifest = json.loads(read(pointer["manifest"]))
    _, base = validate_generation(pointer, manifest, prefix)
    if expected_dataset_id is not None and manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("Serving snapshot belongs to a different data release")
    for name, metadata in manifest["files"].items():
        validate_payload(name, read(f"{base}/{name}"), metadata)
    return pointer, manifest


def verify_data_release(s3, bucket, manifest, *, expected_producer=None):
    """A deployable snapshot must identify a canonical coherent training release."""
    from src.data.release import resolve_release

    dataset_id = manifest.get("dataset_id")
    if not _is_digest(dataset_id):
        raise ValueError("Serving snapshot lacks a canonical data release")
    selected, release = resolve_release(s3, bucket, release_id=dataset_id)
    producer = release["data_producer_sha256"]
    if selected != dataset_id or (expected_producer is not None and producer != expected_producer):
        raise ValueError("Serving snapshot data producer differs from deployed code")
    return selected, producer


@dataclass(frozen=True)
class SnapshotBuild:
    pointer_etag: str | None
    model_generations: tuple[tuple[str, str, str], ...]
    plan_id: str | None
    dataset_id: str | None


def begin_build(s3, bucket, prefix="models", *, expected_model_keys=None):
    dataset_id = os.environ.get("FF_DATASET_ID") or os.environ.get("FF_DATA_RELEASE")
    if os.environ.get("FF_DATA_RELEASE") and dataset_id != os.environ["FF_DATA_RELEASE"]:
        raise ValueError("Serving dataset and data release identities disagree")
    _, pointer = _read(s3, bucket, f"{prefix}/predictions_cache/current.json")
    models = []
    for position in POSITIONS:
        manifest, etag = load_manifest_snapshot(s3, bucket, prefix, position)
        stable = (manifest or {}).get("stable") or {}
        if not stable.get("key") or not etag:
            raise RuntimeError(f"No approved {position} model for serving snapshot")
        expected = (expected_model_keys or {}).get(position)
        if expected is not None and stable["key"] != expected:
            raise RuntimeError(
                f"Build plan {position} model was superseded before cache construction"
            )
        models.append((position, etag, stable["key"]))
    return SnapshotBuild(pointer, tuple(models), os.environ.get("FF_BUILD_PLAN_ID"), dataset_id)


def publish(
    s3,
    bucket,
    directory,
    build: SnapshotBuild,
    prefix="models",
    *,
    generation=None,
    before_publish=None,
):
    root = Path(directory)
    captured = None
    if generation is not None or (root / "current.json").exists():
        captured, content = read_generation(root, generation)
    else:
        # Offline producers may supply an isolated staging directory. Runtime
        # consumers never accept this uncommitted shape.
        content = {name: (root / name).read_bytes() for name in FILES}
    files = {}
    for name in FILES:
        payload = content[name]
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    cache_schema = json.loads(content["fingerprint.json"])["schema_version"]
    if cache_schema != CACHE_SCHEMA_VERSION:
        raise ValueError("Serving cache producer schema is incompatible")
    manifest = {
        "schema_version": 1,
        "cache_schema_version": cache_schema,
        "files": files,
        "models": {pos: key for pos, _, key in build.model_generations},
        "build_plan_id": build.plan_id,
        "dataset_id": build.dataset_id,
    }
    generation = hashlib.sha256(_bytes(manifest)).hexdigest()
    base = f"{prefix}/predictions_cache/generations/{generation}"
    for name, payload in content.items():
        s3.put_object(Bucket=bucket, Key=f"{base}/{name}", Body=payload)
    s3.put_object(
        Bucket=bucket,
        Key=f"{base}/manifest.json",
        Body=_bytes(manifest),
        ContentType="application/json",
    )
    # Avoid superseded build work. The snapshot pointer is the serving release
    # boundary; a later model-head advance alone does not mutate this release.
    for position, expected, _ in build.model_generations:
        _, actual = load_manifest_snapshot(s3, bucket, prefix, position)
        if actual != expected:
            raise RuntimeError(f"{position} model advanced during serving cache construction")
    pointer = {"schema_version": 1, "generation": generation, "manifest": f"{base}/manifest.json"}
    if captured is not None and is_invalidated(root, captured.name):
        raise ValueError("Serving snapshot generation was invalidated before upload")
    condition = {"IfMatch": build.pointer_etag} if build.pointer_etag else {"IfNoneMatch": "*"}
    if before_publish is not None:
        before_publish(pointer)
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/predictions_cache/current.json",
        Body=_bytes(pointer),
        ContentType="application/json",
        **condition,
    )
    return pointer


def is_invalidated(directory, generation):
    """A generation revocation survives worker replacement and pointer advances."""
    if not _is_digest(generation):
        raise ValueError("Invalid serving snapshot generation")
    return (Path(directory) / "invalidated" / generation).exists()


def invalidate_generation(directory, generation):
    if not _is_digest(generation):
        raise ValueError("Invalid serving snapshot generation")
    target = Path(directory) / "invalidated"
    target.mkdir(parents=True, exist_ok=True)
    (target / generation).touch()


def _local_manifest(root, generation):
    if is_invalidated(root, generation):
        raise ValueError("Serving snapshot generation was invalidated")
    directory = Path(root) / "generations" / generation
    manifest = json.loads((directory / "manifest.json").read_bytes())
    # Remote key validation happens during sync. Local readers validate the
    # content-addressed manifest independently of the remote bucket prefix.
    pointer = {
        "schema_version": 1,
        "generation": generation,
        "manifest": f"models/predictions_cache/generations/{generation}/manifest.json",
    }
    validate_generation(pointer, manifest)
    return directory, manifest


def active_directory(directory) -> Path:
    root = Path(directory)
    pointer = root / "current.json"
    if not pointer.exists():
        return root
    value = json.loads(pointer.read_text())
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("Invalid serving snapshot pointer")
    generation = value.get("generation")
    if value.get("schema_version") != 1 or not _is_digest(generation):
        raise ValueError("Invalid serving snapshot generation")
    return _local_manifest(root, generation)[0]


def read_generation(directory, generation=None, *, expected_dataset_id=None):
    """Capture and verify bytes once; callers must deserialize these exact bytes."""
    root = Path(directory)
    if generation is not None and not _is_digest(generation):
        raise ValueError("Invalid serving snapshot generation")
    selected = active_directory(root) if generation is None else root / "generations" / generation
    if selected == root:
        raise ValueError("No committed serving snapshot generation")
    selected, manifest = _local_manifest(root, selected.name)
    if expected_dataset_id is not None and manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("Serving snapshot belongs to a different data release")
    files = {}
    for name in FILES:
        payload = (selected / name).read_bytes()
        validate_payload(name, payload, manifest["files"][name])
        files[name] = payload
    if is_invalidated(root, selected.name):
        raise ValueError("Serving snapshot generation was invalidated during read")
    return selected, files


def _install_generation(root, pointer, manifest, files):
    root = Path(root)
    generation = pointer["generation"]
    if is_invalidated(root, generation):
        raise ValueError("Cannot install an invalidated serving snapshot generation")
    parent = root / "generations"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / generation
    stage = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=parent))
    temporary = None
    try:
        for name in FILES:
            validate_payload(name, files[name], manifest["files"][name])
            (stage / name).write_bytes(files[name])
        (stage / "manifest.json").write_bytes(_bytes(manifest))
        try:
            os.rename(stage, final)
        except OSError:
            if not final.is_dir():
                raise
        # An existing directory (including a concurrent identical install) is
        # trusted only after verifying all its actual bytes, not just its name.
        read_generation(root, generation)
        fd, temporary = tempfile.mkstemp(prefix=".current-", dir=root)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_bytes(pointer))
        if is_invalidated(root, generation):
            raise ValueError("Serving snapshot generation was invalidated before publication")
        os.replace(temporary, root / "current.json")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return final


def publish_local(directory, files):
    """Commit one complete local computation without granting remote publication."""
    if set(files) != set(FILES):
        raise ValueError("Serving snapshot file inventory mismatch")
    fingerprint = json.loads(files["fingerprint.json"])
    if fingerprint.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Serving cache producer schema is incompatible")
    manifest = {
        "schema_version": 1,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "files": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in files.items()
        },
        "models": {},
        "build_plan_id": os.environ.get("FF_BUILD_PLAN_ID"),
        "dataset_id": os.environ.get("FF_DATASET_ID"),
    }
    generation = hashlib.sha256(_bytes(manifest)).hexdigest()
    pointer = {
        "schema_version": 1,
        "generation": generation,
        "manifest": f"models/predictions_cache/generations/{generation}/manifest.json",
    }
    return _install_generation(directory, pointer, manifest, files)


def sync(
    s3, bucket, directory, prefix="models", *, expected_dataset_id=None, pinned_generation=None
):
    """Return None only for pre-generation buckets; other failures retain old data."""
    pointer, etag = _read(s3, bucket, f"{prefix}/predictions_cache/current.json")
    if pointer is None and etag is None and pinned_generation is None:
        return None
    manifest = None
    if pointer is not None:
        validate_pointer(pointer, prefix)
        manifest, _ = _read(s3, bucket, pointer["manifest"])
        validate_generation(pointer, manifest, prefix)
    if pinned_generation is not None and (
        manifest is None or manifest.get("dataset_id") != expected_dataset_id
    ):
        if expected_dataset_id is None or not _is_digest(pinned_generation):
            raise ValueError("Serving snapshot fallback requires valid data and generation pins")
        pointer = {
            "schema_version": 1,
            "generation": pinned_generation,
            "manifest": f"{prefix}/predictions_cache/generations/{pinned_generation}/manifest.json",
        }
        manifest, _ = _read(s3, bucket, pointer["manifest"])
    generation, base = validate_generation(pointer, manifest, prefix)
    if expected_dataset_id is not None and manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("Serving snapshot belongs to a different data release")
    current = Path(directory) / "current.json"
    try:
        installed = json.loads(current.read_bytes())
    except (OSError, ValueError):
        installed = None
    if installed == pointer:
        # Remote metadata still establishes the selected release. Revalidate
        # local bytes and revocations before reusing them; a directory name
        # alone is not evidence of a complete or unmodified installation.
        read_generation(directory, generation, expected_dataset_id=expected_dataset_id)
        return {"generation": generation, "files": len(FILES)}
    files = {
        name: s3.get_object(Bucket=bucket, Key=f"{base}/{name}")["Body"].read() for name in FILES
    }
    _install_generation(directory, pointer, manifest, files)
    return {"generation": generation, "files": len(FILES)}


def main():
    """Deployment readiness gate using the already configured AWS CLI."""
    import argparse
    import subprocess
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["wait"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bucket")
    source.add_argument("--task-definition", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--bind-data", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.bind_data and args.task_definition is None:
        parser.error("--bind-data requires --task-definition")
    if args.task_definition:
        if args.prefix is not None:
            parser.error("--prefix cannot override the rendered task definition")
        from src.artifacts.deployment import serving_coordinates

        args.bucket, args.prefix = serving_coordinates(json.loads(args.task_definition.read_text()))
    elif args.prefix is None:
        args.prefix = "models"
    deadline = time.monotonic() + args.timeout
    with tempfile.TemporaryDirectory(prefix="snapshot-readiness-") as temporary:
        destination = Path(temporary) / "response.json"

        def get_bytes(key):
            result = subprocess.run(
                [
                    "aws",
                    "s3api",
                    "get-object",
                    "--bucket",
                    args.bucket,
                    "--key",
                    key,
                    str(destination),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                if "NoSuchKey" in result.stderr or "404" in result.stderr:
                    return None
                raise RuntimeError(f"Cannot verify serving snapshot: {result.stderr.strip()}")
            return destination.read_bytes()

        def get(key):
            payload = get_bytes(key)
            return json.loads(payload) if payload is not None else None

        class Reader:
            def get_object(self, *, Bucket, Key):
                import io

                payload = get_bytes(Key)
                if payload is None:
                    raise FileNotFoundError(Key)
                return {"Body": io.BytesIO(payload)}

        while True:
            try:
                pointer = get(f"{args.prefix}/predictions_cache/current.json")
                validate_pointer(pointer, args.prefix)
                manifest = get(pointer["manifest"])
                generation, base = validate_generation(pointer, manifest, args.prefix)
                for name, expected in manifest["files"].items():
                    validate_payload(name, get_bytes(f"{base}/{name}"), expected)
                if args.bind_data:
                    from src.artifacts.deployment import bind_data_release
                    from src.data.release import data_producer_hashes, producer_fingerprint

                    release_id, producer = verify_data_release(
                        Reader(),
                        args.bucket,
                        manifest,
                        expected_producer=producer_fingerprint(data_producer_hashes(Path.cwd())),
                    )
            except (ValueError, FileNotFoundError):
                # A missing/incompatible generation may be replaced while this
                # deployment waits. AWS authorization/transport errors propagate.
                pass
            else:
                if args.bind_data:
                    task = bind_data_release(
                        json.loads(args.task_definition.read_text()),
                        release_id,
                        producer,
                        snapshot_generation=generation,
                    )
                    args.task_definition.write_text(json.dumps(task, indent=2) + "\n")
                print(f"Serving snapshot ready: {generation}")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "No compatible published serving snapshot; keeping current deployment"
                )
            time.sleep(min(15, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
