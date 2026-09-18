"""Checksum-verified, write-once experiment cache; never a serving publisher.

Payloads use the project's existing trusted Python/model serializers. Only the
owner's local cache and configured training bucket are accepted cache sources.
Durable run results must be materialized outside this disposable store.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

LOG = logging.getLogger(__name__)
DEFAULT_MAX_BYTES = 20 * 1024**3
MAX_AGE_SECONDS = 30 * 86400
S3_PREFIX = "experiment-cache/v1"


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def default_root() -> Path:
    return Path(
        os.environ.get(
            "FF_RESULT_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache/results"
        )
    ).resolve()


class ResultStore:
    def __init__(self, root=None, *, s3=None, bucket=None, max_bytes=DEFAULT_MAX_BYTES):
        self.root = Path(root or default_root())
        self.s3 = s3
        self.bucket = bucket
        self.max_bytes = max_bytes

    @classmethod
    def configured(cls):
        bucket = os.environ.get(
            "FF_RESULT_CACHE_BUCKET",
            os.environ.get("S3_BUCKET") if os.environ.get("AWS_BATCH_JOB_ID") else None,
        )
        if bucket:
            import boto3

            return cls(s3=boto3.client("s3"), bucket=bucket)
        return cls()

    def path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Result cache key must be a SHA-256 digest")
        return self.root / key

    def _manifest(self, path, key):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["key"] != key or manifest["version"] != 1:
            raise ValueError("Cache identity mismatch")
        if time.time() - manifest["created_at"] > MAX_AGE_SECONDS:
            raise ValueError("Expired cache entry")
        if not manifest["files"] or manifest["bytes"] > self.max_bytes:
            raise ValueError("Empty or oversized cache entry")
        actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
        if actual != {*manifest["files"], "manifest.json"}:
            raise ValueError("Cache inventory mismatch")
        for name, expected in manifest["files"].items():
            item = path / name
            if not item.resolve().is_relative_to(path.resolve()) or item.is_symlink():
                raise ValueError("Unsafe cache member")
            if file_digest(item) != expected:
                raise ValueError("Cache checksum mismatch")
        return manifest

    def lookup(self, key):
        path = self.path(key)
        try:
            if not path.exists() and self.s3 is not None:
                self._download(key)
            manifest = self._manifest(path, key)
            os.utime(path, None)
            return path, manifest
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            if path.exists():
                LOG.warning("Ignoring invalid result cache %s: %s", key[:12], exc)
            return None

    def publish(self, key, writer, *, source_run_id, identity):
        """A complete entry appears in one rename; another complete writer wins."""
        path = self.path(key)
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".building-", dir=self.root) as temporary:
            stage = Path(temporary) / "entry"
            stage.mkdir()
            writer(stage)
            files = {
                str(p.relative_to(stage)): file_digest(p) for p in stage.rglob("*") if p.is_file()
            }
            size = sum((stage / name).stat().st_size for name in files)
            if size > self.max_bytes:
                raise ValueError("Result exceeds cache size limit")
            manifest = {
                "version": 1,
                "key": key,
                "identity": identity,
                "source_run_id": source_run_id,
                "created_at": time.time(),
                "files": files,
                "bytes": size,
            }
            (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            if path.exists() and self.lookup(key) is None:
                # Move a corrupt entry out of the published namespace first.
                quarantine = Path(temporary) / "invalid"
                with contextlib.suppress(FileNotFoundError):
                    path.rename(quarantine)
            try:
                stage.rename(path)
            except OSError:
                if self.lookup(key) is None:
                    raise
            if self.s3 is not None:
                self._upload(key, path)
        self.prune(protect=key)

    def _upload(self, key, path):
        from botocore.exceptions import ClientError

        with tempfile.TemporaryFile() as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for item in path.rglob("*"):
                    if item.is_file():
                        archive.write(item, str(item.relative_to(path)))
            stream.seek(0)
            try:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=f"{S3_PREFIX}/{key}.zip",
                    Body=stream,
                    IfNoneMatch="*",
                    ContentType="application/zip",
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                    raise

    def _download(self, key):
        from botocore.exceptions import ClientError

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{S3_PREFIX}/{key}.zip")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return
            raise
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=self.root) as temporary:
            stage = Path(temporary) / "entry"
            stage.mkdir()
            with tempfile.TemporaryFile() as stream:
                size = 0
                body = response["Body"]
                try:
                    for chunk in iter(lambda: body.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("Oversized remote cache entry")
                        stream.write(chunk)
                finally:
                    body.close()
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    if sum(item.file_size for item in archive.infolist()) > self.max_bytes:
                        raise ValueError("Oversized expanded cache entry")
                    names = archive.namelist()
                    if len(names) != len(set(names)):
                        raise ValueError("Duplicate cache members")
                    for item in archive.infolist():
                        destination = stage / item.filename
                        if not destination.resolve().is_relative_to(stage.resolve()):
                            raise ValueError("Unsafe remote cache path")
                        if item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError("Unexpected directory or symlink in cache")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as target, archive.open(item) as source:
                            shutil.copyfileobj(source, target)
            self._manifest(stage, key)
            try:
                stage.rename(self.path(key))
            except OSError:
                if not self.path(key).is_dir():
                    raise

    def prune(self, *, protect=None):
        """Evict old/least recently accessed entries, never durable outputs."""
        entries = []
        for path in self.root.iterdir():
            if not path.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", path.name):
                continue
            try:
                manifest = json.loads((path / "manifest.json").read_text())
                entries.append(
                    (path.stat().st_mtime, path, manifest["bytes"], manifest["created_at"])
                )
            except (OSError, ValueError, KeyError):
                continue
        total = sum(item[2] for item in entries)
        for _, path, size, created in sorted(entries):
            if path.name != protect and (
                total > self.max_bytes or time.time() - created > MAX_AGE_SECONDS
            ):
                shutil.rmtree(path, ignore_errors=True)
                total -= size


def configure_s3_expiration(s3, bucket):
    """Install only our disposable-prefix rule, preserving every existing rule."""
    from botocore.exceptions import ClientError

    try:
        rules = s3.get_bucket_lifecycle_configuration(Bucket=bucket)["Rules"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
            raise
        rules = []
    desired = {
        "ID": "ff-experiment-cache-v1",
        "Status": "Enabled",
        "Filter": {"Prefix": f"{S3_PREFIX}/"},
        "Expiration": {"Days": 30},
    }
    if any(rule == desired for rule in rules):
        return False
    rules = [rule for rule in rules if rule.get("ID") != desired["ID"]] + [desired]
    s3.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configure disposable result-cache retention")
    parser.add_argument("--configure-s3-expiration", action="store_true", required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    import boto3

    changed = configure_s3_expiration(boto3.client("s3"), args.bucket)
    print(f"Result-cache expiration {'updated' if changed else 'already configured'}")
