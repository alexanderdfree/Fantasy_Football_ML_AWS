"""Bounded S3 transfers and last-good recovery for the CI-built live snapshot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionError, HTTPClientError, IncompleteReadError

from src.serving import upcoming_status
from src.serving.serialization import _MODEL_PRED_PREFIXES, _ROW_PRED_PREFIXES, _VALID_SCORING

_MAX_BYTES = 10 * 1024 * 1024
_VERSION_LIMIT = 6  # Latest plus at most five older versions, never an unbounded scan.
_sync_lock = threading.Lock()
_etags: dict[tuple[str, str, str], tuple[str, str]] = {}


def _client():
    # One retry budget includes reading the response body (SDK retries alone do
    # not cover interrupted StreamingBody reads). Never multiply nested retries.
    return boto3.client(
        "s3", config=Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 1})
    )


def _retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except (ConnectionError, HTTPClientError, IncompleteReadError, ClientError) as exc:
            if isinstance(exc, ClientError):
                response = exc.response
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                code = response.get("Error", {}).get("Code")
                if status not in (408, 429, 500, 502, 503, 504) and code not in (
                    "SlowDown",
                    "RequestTimeout",
                    "InternalError",
                    "ServiceUnavailable",
                ):
                    raise
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite value in projection artifact")
    return result


def decode_artifact(body: bytes) -> dict:
    """Validate the serving contract, allowing stale and optional-model data.

    Timestamps and week are never repaired or advanced by a download. Optional
    source metadata and pre-K/DST snapshots remain backwards compatible.
    """
    if len(body) > _MAX_BYTES:
        raise ValueError("Projection artifact exceeds download size limit")
    payload = json.loads(body, parse_float=_finite_float, parse_constant=_finite_float)
    if not isinstance(payload, dict) or type(payload.get("available")) is not bool:
        raise ValueError("Missing projection availability")
    timestamp = datetime.fromisoformat(str(payload.get("generated_at")).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Projection timestamp requires a timezone")
    for key in ("sources", "source_status", "data_quality"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"Invalid projection {key}")
    if "injuries" in payload.get("sources", {}) and not isinstance(
        payload["sources"]["injuries"], dict
    ):
        raise ValueError("Invalid projection injury metadata")
    try:
        # Exercise the same optional metadata paths that the endpoint reads.
        upcoming_status.freshness(payload)
        upcoming_status.data_quality(payload.get("sources", {}))
        quality = payload.get("data_quality", {})
        if not isinstance(quality.get("issues", []), list) or any(
            not isinstance(issue, dict) or not isinstance(issue.get("message", ""), str)
            for issue in quality.get("issues", [])
        ):
            raise ValueError("Invalid projection coverage issues")
        status = payload.get("source_status", {})
        if not isinstance(status.get("weather", []), list) or any(
            not isinstance(item, dict) for item in status.get("weather", [])
        ):
            raise ValueError("Invalid projection weather metadata")
        if not isinstance(status.get("missing_kicker_teams", []), list) or any(
            not isinstance(team, str) for team in status.get("missing_kicker_teams", [])
        ):
            raise ValueError("Invalid missing kicker teams")
    except (AttributeError, TypeError, KeyError) as exc:
        raise ValueError("Invalid projection source metadata") from exc
    if not payload["available"]:
        if payload.get("reason") != "offseason":
            raise ValueError("Unavailable artifact is not a verified offseason")
        return payload
    if type(payload.get("season")) is not int or type(payload.get("week")) is not int:
        raise ValueError("Missing projection season/week")
    if payload["season"] < 2000 or not 1 <= payload["week"] <= 22:
        raise ValueError("Invalid projection season/week")
    scoring = payload.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("Missing scoring formats")
    identities = None
    for fmt in _VALID_SCORING:
        rows = scoring.get(fmt)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty projection scoring format: {fmt}")
        keys = set()
        positions = set()
        for row in rows:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key]
                for key in ("player_id", "name", "position", "team")
            ):
                raise ValueError("Invalid projection player identity")
            if row["position"] not in ("QB", "RB", "WR", "TE", "K", "DST"):
                raise ValueError("Invalid projection position")
            for prefix in _ROW_PRED_PREFIXES:
                value = row.get(f"{prefix}_pred")
                if value is not None and type(value) not in (int, float):
                    raise ValueError("Invalid projection value")
            if not any(row.get(f"{prefix}_pred") is not None for prefix in _MODEL_PRED_PREFIXES):
                raise ValueError("Player has no usable model projection")
            keys.add((row["player_id"], row["position"], row["team"]))
            positions.add(row["position"])
        if len(keys) != len(rows) or (identities is not None and keys != identities):
            raise ValueError("Inconsistent projection player rows")
        declared = payload.get("positions", sorted(positions))
        if (
            not isinstance(declared, list)
            or any(not isinstance(pos, str) for pos in declared)
            or set(declared) != positions
        ):
            raise ValueError("Missing declared projection positions")
        identities = keys
    return payload


def upload(path: str, bucket: str, key: str) -> bool:
    """Retry the same validated bytes; a failed upload never reruns inference."""
    try:
        body = Path(path).read_bytes()
        decode_artifact(body)
        s3 = _client()
        _retry(
            lambda: s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        )
        print(f"[upcoming_week] uploaded artifact -> s3://{bucket}/{key}")
        return True
    except Exception as exc:  # noqa: BLE001 - caller makes exhausted publication fail CI
        print(f"[upcoming_week] S3 artifact upload failed: {exc!r}")
        return False


@contextlib.contextmanager
def _exclusive_sync(path: Path):
    """Skip overlapping worker polls; OS locks are released even after a crash."""
    if not _sync_lock.acquire(blocking=False):
        yield False
        return
    try:
        with path.with_suffix(".sync.lock").open("a+b") as handle:
            try:
                if os.name == "nt":
                    import msvcrt

                    if not handle.tell():
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                # Gunicorn can fork while its master poller holds this fd.
                # close() alone leaves flock held by an inherited worker fd.
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        _sync_lock.release()


def _download(s3, **request):
    def read():
        response = s3.get_object(**request)
        with contextlib.closing(response["Body"]) as stream:
            data = bytearray()
            # StreamingBody checks Content-Length at EOF, not after a single
            # nonempty bounded read. Reach EOF so short responses are retried.
            while chunk := stream.read(min(64 * 1024, _MAX_BYTES + 1 - len(data))):
                data.extend(chunk)
                if len(data) > _MAX_BYTES:
                    raise ValueError("Projection artifact exceeds download size limit")
            body = bytes(data)
        decode_artifact(body)
        return body, response.get("ETag")

    return _retry(read)


def _install(path, body, etag, identity):
    # Keep the original bytes and metadata. Readers see either complete file.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f"{path.name}.", delete=False
    ) as handle:
        tmp = handle.name
        try:
            handle.write(body)
            handle.close()
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    if etag:
        _etags[identity] = (hashlib.sha256(body).hexdigest(), etag)


def sync(path: str, bucket: str, key: str) -> bool:
    """Install valid updates only; recover a cold start from recent S3 versions."""
    path = Path(path)
    identity = (str(path), bucket, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_sync(path) as acquired:
            if not acquired:
                return False
            local_valid = False
            local_digest = None
            with contextlib.suppress(OSError, ValueError, TypeError):
                local_body = path.read_bytes()
                decode_artifact(local_body)
                local_valid = True
                # Workers share a file, not their ETag maps. An identical-byte
                # installation by a peer must not invalidate this worker's ETag.
                local_digest = hashlib.sha256(local_body).hexdigest()
            s3 = _client()
            request = {"Bucket": bucket, "Key": key}
            known = _etags.get(identity)
            if local_valid and known and known[0] == local_digest:
                request["IfNoneMatch"] = known[1]
            try:
                body, etag = _download(s3, **request)
                _install(path, body, etag, identity)
                return True
            except Exception as exc:  # noqa: BLE001 - preserve valid disk snapshot on any failure
                if (
                    isinstance(exc, ClientError)
                    and exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 304
                ):
                    return False
                print(f"[upcoming_week] S3 artifact sync skipped: {exc!r}")
                if local_valid:
                    return False
            versions = _retry(
                lambda: s3.list_object_versions(Bucket=bucket, Prefix=key, MaxKeys=_VERSION_LIMIT)
            )
            for version in versions.get("Versions", []):
                if version["Key"] != key or version.get("IsLatest"):
                    continue
                try:
                    body, etag = _download(
                        s3, Bucket=bucket, Key=key, VersionId=version["VersionId"]
                    )
                    _install(path, body, etag, identity)
                    print(
                        f"[upcoming_week] recovered prior artifact version {version['VersionId']}; retained original timestamps"
                    )
                    return True
                except Exception as exc:  # noqa: BLE001 - one bad version need not prevent recovery
                    print(f"[upcoming_week] prior artifact version skipped: {exc!r}")
    except Exception as exc:  # noqa: BLE001 - serving keeps responding during S3 failures
        print(f"[upcoming_week] S3 artifact recovery unavailable: {exc!r}")
    return False
