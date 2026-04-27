"""Sync position model tarballs + inference data from S3 at container boot.

Opt-in via FF_MODEL_S3_BUCKET env var. Unset/empty -> no-op (dev, tests).

``sync_models_from_s3`` reads each position's ``manifest.json`` and prefers
the smoke-test-validated ``stable`` artifact, falling back to ``current``
then ``previous`` only when ``stable`` is missing or fails to load. For
pre-manifest buckets it falls back to the legacy
``s3://{bucket}/{prefix}/{POS}/model.tar.gz`` key (migration compat).
See ``src/shared/artifact_gc.py`` for retention; ``stable`` is exempted from GC.

``sync_data_from_s3`` pulls the splits + raw weekly parquets that inference
needs (K reconstructs kicker stats from data/raw/, all positions read
schedules for weather features). These were previously baked into the
Docker image via the deploy workflow; fetching at boot shrinks the image
and decouples deploy.yml from data changes.

Fail-loud: if every manifest entry (``stable``, ``current``, ``previous``)
fails, the raise propagates and gunicorn --preload aborts before binding
:8000, blocking a broken rollout. Per-position graceful degradation in the
Flask layer lives in ``app.py``.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import tarfile
import time
from pathlib import Path

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_ENV_BUCKET = "FF_MODEL_S3_BUCKET"
_ENV_PREFIX = "FF_MODEL_S3_PREFIX"

_SPLIT_KEYS = ("train.parquet", "val.parquet", "test.parquet")
_RAW_PREFIX = "data/raw/"
_RAW_EXCLUDE_SUFFIX = "_2023_2023.parquet"

# Manifest schema lives in a single place so producer + consumer can't drift.
# Schema v2:
#   {
#     "schema_version": 2,
#     "current":  {"key": "...", "sha7": "...", "bytes": int, "uploaded_at": "..."},
#     "stable":   <same shape> | null,    // last upload that PASSED smoke test
#     "previous": <same shape> | null,    // the prior current (forensics)
#     "history":  ["key0", "key1", ...]   // newest-first, capped at HISTORY_KEEP_N
#   }
# v1 manifests (no "stable") are read transparently — the consumer falls
# through to "current" until the next successful smoke test populates
# "stable". This keeps the migration window safe.
MANIFEST_SCHEMA_VERSION = 2
HISTORY_KEEP_N = 5


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def manifest_key(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/manifest.json"


def legacy_model_key(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/model.tar.gz"


def history_prefix(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/history/"


def new_history_key(prefix: str, pos: str, ts: str, sha7: str) -> str:
    return f"{history_prefix(prefix, pos)}{ts}-{sha7}/model.tar.gz"


def load_manifest(s3_client, bucket: str, prefix: str, pos: str) -> dict | None:
    """Return the parsed manifest.json for ``pos``, or ``None`` if absent.

    Re-raises any other S3 error so the caller can't silently confuse a
    missing manifest with a permissions / transient failure.
    """
    from botocore.exceptions import ClientError

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=manifest_key(prefix, pos))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read())


def write_manifest(s3_client, bucket: str, prefix: str, pos: str, manifest: dict) -> None:
    """Publish a ``manifest.json``. This write IS the atomic promotion —
    after the put returns, subsequent consumer syncs will pull the new
    artifact. Earlier steps in the producer (validation, history upload)
    only raise; a raise leaves the old manifest in place and the site keeps
    serving the previous good artifact.
    """
    body = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key(prefix, pos),
        Body=body,
        ContentType="application/json",
    )


def build_manifest(
    new_key: str,
    sha7: str,
    bytes_: int,
    uploaded_at: str,
    old_manifest: dict | None = None,
    smoke_passed: bool = False,
    keep_history: int = HISTORY_KEEP_N,
) -> dict:
    """Pure helper: return the dict for the new manifest given the new upload
    and the old manifest (``None`` on first write). ``previous`` becomes the
    old ``current``; ``history`` prepends the new key and caps at ``keep_history``
    newest-first. Duplicates of the new key in the old history are stripped
    (idempotent on retry).

    ``stable`` advances to the new entry when ``smoke_passed=True`` and
    otherwise carries forward the old ``stable`` (or ``None`` if there was
    none). The ``stable`` pointer is the only reason to consult the old
    manifest beyond ``previous``.
    """
    new_entry = {
        "key": new_key,
        "sha7": sha7,
        "bytes": bytes_,
        "uploaded_at": uploaded_at,
    }
    old = old_manifest or {}
    old_current = old.get("current")
    old_stable = old.get("stable")
    old_history = [k for k in (old.get("history") or []) if k != new_key]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "current": new_entry,
        "stable": new_entry if smoke_passed else old_stable,
        "previous": old_current,
        "history": [new_key] + old_history[: keep_history - 1],
    }


def _extract_tarball(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest_resolved not in target.parents and target != dest_resolved:
                raise RuntimeError(f"Tarball escape attempt: {member.name}")
        tar.extractall(dest, filter="data")


def _try_key(s3_client, bucket: str, key: str, dest: Path) -> dict:
    """Download + extract one tarball key. Raises on any S3 / tar / extract
    error — the caller decides whether to fall back."""
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    _extract_tarball(data, dest)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def _sync_one(s3_client, bucket: str, prefix: str, pos: str, root: Path) -> dict:
    """Sync one position with manifest-driven fallback.

    Order: manifest.stable → manifest.current → manifest.previous → legacy
    ``model.tar.gz`` (only when manifest is absent; a manifest-present-but-
    all-entries-broken state is a real bug, not something to paper over with
    a stale legacy copy). The frontend prefers ``stable`` because that's the
    last artifact a smoke test confirmed actually loads + predicts; serving
    ``current`` means stable was missing or unreadable, which is page-worthy.

    A v1-shaped manifest (no ``stable`` slot) reads as ``stable=None`` and
    falls through to ``current`` automatically — that's the migration window.

    Each fall-through is logged with the grep-able tag
    ``source=stable|current|previous|legacy`` so on-call can tell from
    CloudWatch which artifact tier we ended up on.
    """
    from botocore.exceptions import ClientError

    # S3 keys are uppercase POS (set by the producer in src/batch/train.py);
    # the local layout is lowercase to match the registry's model_dir entries
    # (``src/qb/outputs/models``). On case-sensitive Linux these diverge, so
    # the destination must be normalized here even though macOS APFS papered
    # over it during the rename refactor.
    dest = root / pos.lower() / "outputs" / "models"
    manifest = load_manifest(s3_client, bucket, prefix, pos)

    if manifest is None:
        # Pre-migration buckets, first boot after the producer change.
        r = _try_key(s3_client, bucket, legacy_model_key(prefix, pos), dest)
        print(f"[model_sync] {pos}: source=legacy ({r['key']})", flush=True)
        return {"pos": pos, "source": "legacy", **r}

    tried: list[tuple[str, str, str]] = []
    for label in ("stable", "current", "previous"):
        entry = manifest.get(label)
        if not entry or not entry.get("key"):
            continue
        try:
            r = _try_key(s3_client, bucket, entry["key"], dest)
        except (ClientError, tarfile.TarError, RuntimeError, OSError, EOFError) as e:
            # ``EOFError`` covers truncated-gzip from ``gzip.py``; ``tarfile.TarError``
            # covers "not a gzip file" / corrupt header; ``ClientError`` handles
            # S3-side issues (NoSuchKey, throttling, etc.). Anything else is a
            # real bug and should fail loud.
            tried.append((label, entry["key"], repr(e)))
            print(
                f"[model_sync] {pos} {label} ({entry['key']}) FAILED: {e!r} — falling through",
                flush=True,
            )
            continue
        print(f"[model_sync] {pos}: source={label} ({r['key']})", flush=True)
        return {"pos": pos, "source": label, **r}

    raise RuntimeError(f"[model_sync] {pos}: all manifest entries failed: {tried!r}")


def sync_models_from_s3() -> dict | None:
    """Download+extract all six position tarballs in parallel, preferring
    the manifest-pointed ``current`` with automatic fallback to ``previous``.

    Returns a summary dict, or ``None`` if ``FF_MODEL_S3_BUCKET`` is unset/empty.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(f"[model_sync] {_ENV_BUCKET} unset — skipping S3 sync, using on-disk models.")
        return None

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    root = _repo_root()
    import boto3

    s3 = boto3.client("s3")

    print(f"[model_sync] syncing s3://{bucket}/{prefix}/{{POS}}/ -> {root}")
    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(POSITIONS)) as pool:
        futs = [pool.submit(_sync_one, s3, bucket, prefix, pos, root) for pos in POSITIONS]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total = round(time.time() - t0, 2)
    print(f"[model_sync] done in {total}s: {results}")
    return {"total_secs": total, "positions": results}


def _download_file(s3_client, bucket: str, key: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    dest.write_bytes(data)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def sync_data_from_s3() -> dict | None:
    """Download inference data parquets from S3 in parallel.

    Pulls s3://{bucket}/data/{train,val,test}.parquet into data/splits/ and
    every data/raw/*.parquet except the 2023-only duplicates already covered
    by the 2012-2025 range files. Returns a summary dict, or None if
    FF_MODEL_S3_BUCKET is unset/empty.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(f"[data_sync] {_ENV_BUCKET} unset — skipping S3 sync, using on-disk data.")
        return None

    root = _repo_root()
    import boto3

    s3 = boto3.client("s3")

    jobs: list[tuple[str, Path]] = [
        (f"data/{name}", root / "data" / "splits" / name) for name in _SPLIT_KEYS
    ]

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet") or key.endswith(_RAW_EXCLUDE_SUFFIX):
                continue
            jobs.append((key, root / key))

    print(f"[data_sync] syncing {len(jobs)} files from s3://{bucket}/data/ -> {root / 'data'}")
    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futs = [pool.submit(_download_file, s3, bucket, key, dest) for key, dest in jobs]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    print(f"[data_sync] done in {total}s, {total_bytes / 1e6:.1f} MB across {len(results)} files")
    return {"total_secs": total, "total_bytes": total_bytes, "files": len(results)}
