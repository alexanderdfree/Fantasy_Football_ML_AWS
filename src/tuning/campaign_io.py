"""Campaign-owned files, immutable input snapshots and conditional journals."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from src.tuning.campaign_contracts import PREFIX, canonical, identity

MAX_JSON_BYTES = 1024 * 1024


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".campaign-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def local_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".running.lock").open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This local campaign is already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Journal:
    """A unit owns its progress; submit intents use ETag compare-and-swap."""

    def __init__(self, root, *, s3=None, bucket=None, campaign_id=None):
        self.root = Path(root)
        self.s3, self.bucket = s3, bucket
        self.prefix = f"{PREFIX}/{campaign_id}" if campaign_id else None

    def _path(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe campaign path")
        return self.root / relative

    def read(self, name):
        if self.s3 is None:
            try:
                data = self._path(name).read_bytes()
            except FileNotFoundError:
                return None, None
            return json.loads(data), hashlib.sha256(data).hexdigest()
        from botocore.exceptions import ClientError

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{name}")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
            raise
        body = response["Body"]
        try:
            data = body.read(MAX_JSON_BYTES + 1)
        finally:
            body.close()
        if len(data) > MAX_JSON_BYTES:
            raise ValueError("Campaign journal exceeds its size limit")
        return json.loads(data), response["ETag"]

    def write(self, name, value, etag=None):
        if self.s3 is None:
            _, current = self.read(name)
            if current != etag:
                raise RuntimeError("Campaign journal changed concurrently")
            atomic_json(self._path(name), value)
            return hashlib.sha256(canonical(value)).hexdigest()
        response = self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{name}",
            Body=canonical(value),
            ContentType="application/json",
            **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
        )
        atomic_json(self._path(name), value)
        return response["ETag"]

    def immutable(self, name, value):
        existing, _ = self.read(name)
        if existing is not None:
            if existing != value:
                raise ValueError("Campaign identity changed; choose a new campaign ID")
            return
        try:
            self.write(name, value)
        except Exception:
            existing, _ = self.read(name)
            if existing != value:
                raise

    def upload_outputs(self, directory, prefix):
        directory = Path(directory)
        records = {}
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or "data" in path.relative_to(directory).parts
            ):
                continue
            # Live SQLite files use the backup service, including their WAL.
            if path.suffix == ".db" or path.name.endswith(("-wal", "-shm")):
                continue
            relative = str(path.relative_to(directory)).replace(os.sep, "/")
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            records[relative] = {"sha256": sha, "bytes": path.stat().st_size}
            if self.s3 is not None:
                self.s3.upload_file(
                    str(path),
                    self.bucket,
                    f"{self.prefix}/{prefix}/{relative}",
                    ExtraArgs={"Metadata": {"sha256": sha}},
                )
        return records

    def outputs_valid(self, directory, prefix, records):
        if not records:
            return False
        for name, expected in records.items():
            if self.s3 is None:
                path = Path(directory) / name
                if not path.is_file() or path.stat().st_size != expected["bytes"]:
                    return False
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                        return False
            else:
                from botocore.exceptions import ClientError

                try:
                    head = self.s3.head_object(
                        Bucket=self.bucket, Key=f"{self.prefix}/{prefix}/{name}"
                    )
                except ClientError as exc:
                    if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                        return False
                    raise
                if (
                    head["ContentLength"] != expected["bytes"]
                    or head.get("Metadata", {}).get("sha256") != expected["sha256"]
                ):
                    return False
        return True

    def restore_outputs(self, directory, prefix, records):
        """Restore verified completed cells after a Spot worker loses its disk."""
        directory = Path(directory)
        for name, expected in records.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe output checkpoint path")
            path = directory / relative
            if self.s3 is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, f"{self.prefix}/{prefix}/{name}", str(path))
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Output checkpoint checksum mismatch: {name}")


class CellCheckpoint:
    """A frozen campaign owns each completed A/B unit or benchmark fold."""

    def __init__(self, journal, prefix, manifest_id):
        self.journal, self.prefix, self.manifest_id = journal, prefix, manifest_id

    def _name(self, key):
        return f"{self.prefix}/{identity(key)}.json"

    def load(self, key, output=None):
        value, _ = self.journal.read(self._name(key))
        if value is None:
            return None
        if value["manifest_id"] != self.manifest_id or value["key"] != key:
            raise ValueError("Cell checkpoint belongs to a different campaign")
        if identity(value["result"]) != value["sha256"]:
            raise ValueError("Cell checkpoint checksum mismatch")
        if output is not None:
            prefix = f"{self.prefix}/artifacts/{identity(key)}"
            if not self.journal.outputs_valid(output, prefix, value["outputs"]):
                return None
            self.journal.restore_outputs(output, prefix, value["outputs"])
        return value["result"]

    def save(self, key, result, output=None):
        result = clean(result)
        _, token = self.journal.read(self._name(key))
        files = (
            self.journal.upload_outputs(output, f"{self.prefix}/artifacts/{identity(key)}")
            if output is not None
            else {}
        )
        self.journal.write(
            self._name(key),
            {
                "manifest_id": self.manifest_id,
                "key": key,
                "result": result,
                "sha256": identity(result),
                "outputs": files,
            },
            token,
        )


def verify_snapshot(root, manifest):
    root = Path(root)
    for name, entry in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in {"raw", "splits"}
        ):
            raise ValueError("Unsafe input snapshot path")
        path = root / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Input snapshot mismatch: {name}")


def snapshot_local_inputs(source, destination, *, expected_producer):
    """Copy a completed producer release without writing through shared links."""
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "splits/release-inputs.json").read_text())
    if any(manifest["producer"].get(name) != sha for name, sha in expected_producer.items()):
        raise ValueError(
            "Local data producer differs from current code; rebuild and seal inputs first"
        )
    dataset_id = identity(manifest)
    if destination.exists():
        selected = json.loads((destination / "splits/release-inputs.json").read_text())
        if selected != manifest:
            raise ValueError("Existing campaign input snapshot differs")
        verify_snapshot(destination, manifest)
        return dataset_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    verify_snapshot(source, manifest)
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "inputs"
        stage.mkdir()

        def copy(name):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy, manifest["files"]))
        verify_snapshot(stage, manifest)
        atomic_json(stage / "splits/release-inputs.json", manifest)
        atomic_json(
            stage / "raw/.release.json",
            {
                "release_id": dataset_id,
                "provider_sources": "captured"
                if any(k.startswith("raw/provider_sources/") for k in manifest["files"])
                else "derived_only",
            },
        )
        stage.rename(destination)
    return dataset_id
