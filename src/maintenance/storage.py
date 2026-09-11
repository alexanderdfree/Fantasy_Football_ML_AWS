"""Small shared contracts for workers and the import-light control Lambda."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
FORMATS = ("ppr", "half_ppr", "standard")
MODELS = ("ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred")


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include its timezone")
    return result.astimezone(UTC)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def run_key(run_id: str, filename: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{64}", run_id):
        raise ValueError("Invalid maintenance run ID")
    if not re.fullmatch(r"[a-z][a-z0-9_.-]*", filename):
        raise ValueError("Invalid receipt filename")
    return f"maintenance/runs/{run_id}/{filename}"


def missing(error: Exception) -> bool:
    return getattr(error, "response", {}).get("Error", {}).get("Code") in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def get_json(s3, bucket: str, key: str, *, optional=False):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        if optional and missing(error):
            return None, None
        raise
    body = response["Body"]
    try:
        payload = body.read(8 * 1024 * 1024 + 1)
    finally:
        body.close()
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError(f"Oversized JSON object: {key}")
    return json.loads(payload), response.get("ETag")


def put_json(s3, bucket: str, key: str, value, **conditions):
    return s3.put_object(
        Bucket=bucket, Key=key, Body=json_bytes(value), ContentType="application/json", **conditions
    )


def model_pins(s3, bucket: str) -> dict:
    """Resolve every stable artifact once, including its consumed manifest ETag."""
    from src.artifacts.model_sync import load_manifest_snapshot, manifest_key

    pins = {}
    for pos in POSITIONS:
        key = manifest_key("models", pos)
        manifest, etag = load_manifest_snapshot(s3, bucket, "models", pos)
        entry = (manifest or {}).get("stable") or {}
        if not str(entry.get("key", "")).startswith(f"models/{pos}/") or not etag:
            raise ValueError(f"No verified stable manifest for {pos}")
        pins[pos] = {"artifact": entry, "manifest_key": key, "etag": etag}
    return pins


def assert_models_current(s3, bucket: str, pins: dict) -> None:
    from src.artifacts.model_sync import load_manifest_snapshot, manifest_key

    for pos, pin in pins.items():
        manifest, etag = load_manifest_snapshot(s3, bucket, "models", pos)
        if (
            pin["manifest_key"] != manifest_key("models", pos)
            or etag != pin["etag"]
            or (manifest or {}).get("stable") != pin["artifact"]
        ):
            raise RuntimeError(f"Model {pos} changed during maintenance; start a fresh run")
