"""Sync position model tarballs + inference data from S3 at container boot.

Opt-in via FF_MODEL_S3_BUCKET env var. Unset/empty -> no-op (dev, tests).

``sync_models_from_s3`` reads each position's ``manifest.json`` and prefers
the smoke-test-validated ``stable`` artifact, falling back to ``current``
then ``previous`` only when ``stable`` is missing or fails to load. The
legacy ``models/{POS}/model.tar.gz`` mirror is gone (removed in the
parallel-train-batch race fix); manifest absence in production now raises.
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
import contextlib
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
    # __file__ is at <repo>/src/shared/model_sync.py — three parents up.
    # ``parent.parent`` returned <repo>/src/ before, so the data sync silently
    # wrote splits/raw under src/data/ instead of <repo>/data/ where the Flask
    # app reads them (CWD = /app in the container).
    return Path(__file__).resolve().parent.parent.parent


def manifest_key(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/manifest.json"


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

    Order: manifest.stable → manifest.current → manifest.previous. There is
    no legacy ``model.tar.gz`` fallback anymore — the producer
    (``src/batch/train.py::upload_artifacts``) stopped writing that key; every
    production-trained position has a manifest after the schema-v2 migration.
    A manifest-absent state in production is a real bug (the producer
    crashed before the atomic manifest put, or someone hand-deleted the
    manifest); fall back through the chain only, then raise loud.

    The frontend prefers ``stable`` because that's the last artifact a smoke
    test confirmed actually loads + predicts; serving ``current`` means
    stable was missing or unreadable, which is page-worthy.

    A v1-shaped manifest (no ``stable`` slot) reads as ``stable=None`` and
    falls through to ``current`` automatically — that's the migration window.

    Each fall-through is logged with the grep-able tag
    ``source=stable|current|previous`` so on-call can tell from CloudWatch
    which artifact tier we ended up on.
    """
    from botocore.exceptions import ClientError

    # S3 keys are uppercase POS (set by the producer in src/batch/train.py);
    # the local layout is lowercase to match the registry's model_dir entries
    # (``src/qb/outputs/models``). On case-sensitive Linux these diverge, so
    # the destination must be normalized here even though macOS APFS papered
    # over it during the rename refactor.
    dest = root / "src" / pos.lower() / "outputs" / "models"
    manifest = load_manifest(s3_client, bucket, prefix, pos)

    if manifest is None:
        # Pre-migration buckets are no longer expected. Pre-schema-v2 buckets
        # would have hit the legacy ``model.tar.gz`` fallback we removed; any
        # production bucket past that migration has a manifest for every
        # trained position. Raise rather than silently serving a stale local
        # copy (or, worse, a stale legacy key that may have been overwritten
        # by a parallel run from a different image).
        raise RuntimeError(
            f"[model_sync] {pos}: no manifest at "
            f"s3://{bucket}/{manifest_key(prefix, pos)}. "
            "Producer (src/batch/train.py::upload_artifacts) writes the manifest "
            "as its atomic promotion step; absence means the producer crashed "
            "before that step or the object was hand-deleted. Refusing to fall "
            "back to legacy mirror — that's the race source layer C eliminated."
        )

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

    Per-position failures are isolated: if at least one position syncs
    successfully, the function logs the failures, records them in the
    returned summary under ``failed_positions``, and returns normally. The
    affected positions' lazy-load paths in ``src.serving.app`` will surface
    a 500 only for requests touching those positions while the healthy ones
    keep serving. If *every* position fails (S3 unreachable, bucket
    misconfigured, etc.) the first per-position exception is re-raised so
    a totally-useless container still fails loud at boot — preserving the
    existing contract that callers have come to rely on.

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
    failed: list[dict] = []
    first_exc: BaseException | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(POSITIONS)) as pool:
        fut_to_pos = {
            pool.submit(_sync_one, s3, bucket, prefix, pos, root): pos for pos in POSITIONS
        }
        for f in concurrent.futures.as_completed(fut_to_pos):
            pos = fut_to_pos[f]
            try:
                results.append(f.result())
            except Exception as e:
                # Catch broadly: ``_sync_one`` raises ``RuntimeError`` when
                # every manifest entry fails and ``ClientError`` when the
                # legacy path 404s; anything else (network blip, boto bug)
                # should still let the other 5 positions complete rather
                # than collapse the whole boot. ``first_exc`` is preserved
                # so an all-fail run still re-raises with a meaningful
                # error class/message rather than a synthetic aggregate.
                if first_exc is None:
                    first_exc = e
                print(f"[model_sync] FAILED for {pos}: {e!r}", flush=True)
                failed.append({"pos": pos, "error": repr(e)})
    total = round(time.time() - t0, 2)

    if not results:
        # All positions failed — re-raise the first per-position exception
        # so we preserve the existing "useless container should not start"
        # contract (and the original exception class/message for callers).
        assert first_exc is not None  # `failed` is non-empty here by construction.
        raise first_exc

    if failed:
        failed_positions = [f["pos"] for f in failed]
        print(
            f"[model_sync] PARTIAL: {len(results)}/{len(POSITIONS)} positions synced; "
            f"failed: {failed_positions}",
            flush=True,
        )

    print(f"[model_sync] done in {total}s: {results}")
    return {
        "total_secs": total,
        "positions": results,
        "failed_positions": failed,
    }


def _download_file(s3_client, bucket: str, key: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    dest.write_bytes(data)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def sync_benchmark_history_from_s3() -> dict | None:
    """Download every JSON under ``s3://{bucket}/{prefix}/benchmark_history/``
    into the local ``benchmark_history/`` directory.

    Gated on ``FF_MODEL_S3_BUCKET`` like the other syncs — unset/empty makes
    this a no-op so dev and CI tests don't try to hit S3. Fail-soft on an
    empty/missing prefix (a fresh bucket with no benchmark uploads yet
    shouldn't block boot); per-file failures DO raise so a partial sync
    is visible. The Docker image bundles the git-tracked floor of
    ``benchmark_history/`` (see Dockerfile + .dockerignore), so this sync is
    layering on any newer runs uploaded since the image was built — if it
    no-ops, the History tab still renders the committed history.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(
            f"[benchmark_sync] {_ENV_BUCKET} unset — skipping S3 sync, using on-disk benchmark_history."
        )
        return None

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    s3_prefix = f"{prefix}/benchmark_history/"
    root = _repo_root()
    dest_dir = root / "benchmark_history"
    dest_dir.mkdir(parents=True, exist_ok=True)
    import boto3

    s3 = boto3.client("s3")

    jobs: list[tuple[str, Path]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            jobs.append((key, dest_dir / Path(key).name))

    if not jobs:
        print(f"[benchmark_sync] no objects under s3://{bucket}/{s3_prefix}")
        return {"total_secs": 0.0, "total_bytes": 0, "files": 0}

    print(
        f"[benchmark_sync] syncing {len(jobs)} files from s3://{bucket}/{s3_prefix} -> {dest_dir}"
    )
    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futs = [pool.submit(_download_file, s3, bucket, key, dest) for key, dest in jobs]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    print(
        f"[benchmark_sync] done in {total}s, {total_bytes / 1e3:.1f} KB across {len(results)} files"
    )
    return {"total_secs": total, "total_bytes": total_bytes, "files": len(results)}


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


# ---------------------------------------------------------------------------
# Predictions cache (serving pre-warm)
# ---------------------------------------------------------------------------
#
# The Flask app caches the assembled predictions DataFrame + metrics on disk
# after the first ``_ensure_metrics()`` run. Uploading that cache to S3 lets
# the next ECS task hydrate without recomputing — fingerprint mismatch on
# the consumer side guards against serving stale predictions against
# refreshed models. See app.py::_try_hydrate_from_disk and
# _persist_cache_to_disk for the producer/consumer side.

_PREDICTIONS_CACHE_DIR_REL = "data/serving_cache"
_PREDICTIONS_CACHE_FILES = ("predictions.parquet", "metrics.json", "fingerprint.json")


def sync_predictions_cache_from_s3() -> dict | None:
    """Download the three serving-cache files from S3 into data/serving_cache/.

    Missing keys are not errors: a freshly-seeded bucket has no cache until
    the first container computes + uploads. Other S3 errors are logged and
    swallowed — the worst case is the pre-warm thread recomputes and
    re-uploads.

    Gated on FF_MODEL_S3_BUCKET like every other sync. Unset/empty -> no-op.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(
            f"[predcache_sync] {_ENV_BUCKET} unset — skipping S3 sync, "
            f"using on-disk cache (if present)."
        )
        return None

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    s3_prefix = f"{prefix}/predictions_cache"
    root = _repo_root()
    dest_dir = root / _PREDICTIONS_CACHE_DIR_REL
    dest_dir.mkdir(parents=True, exist_ok=True)

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")

    print(f"[predcache_sync] syncing s3://{bucket}/{s3_prefix}/ -> {dest_dir}")
    t0 = time.time()
    results: list[dict] = []
    missing: list[str] = []
    for name in _PREDICTIONS_CACHE_FILES:
        key = f"{s3_prefix}/{name}"
        dest = dest_dir / name
        try:
            r = _download_file(s3, bucket, key, dest)
            results.append(r)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                missing.append(name)
                continue
            print(f"[predcache_sync] {name} FAILED: {e!r} — skipping (will recompute)")
        except Exception as e:  # noqa: BLE001 — best-effort sync
            print(f"[predcache_sync] {name} FAILED: {e!r} — skipping (will recompute)")

    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    if missing:
        # If any file is missing, the fingerprint check on the consumer side
        # will fail anyway — clean up partial downloads so a stale parquet
        # can't be paired with a missing fingerprint and accidentally hydrate.
        for partial in results:
            with contextlib.suppress(OSError):
                (dest_dir / Path(partial["key"]).name).unlink(missing_ok=True)
        print(
            f"[predcache_sync] no cache available (missing: {missing}) "
            f"— first request will compute + upload."
        )
        return {"total_secs": total, "total_bytes": 0, "files": 0, "missing": missing}
    print(
        f"[predcache_sync] done in {total}s, {total_bytes / 1e6:.1f} MB across {len(results)} files"
    )
    return {"total_secs": total, "total_bytes": total_bytes, "files": len(results)}


def upload_predictions_cache_to_s3() -> dict | None:
    """Upload the three serving-cache files from data/serving_cache/ to S3.

    Best-effort: missing local files (cache wasn't written), unset env var,
    or any S3 error all log and return rather than raising — the user-facing
    request that triggered the compute has already succeeded.
    """
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        return None

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    s3_prefix = f"{prefix}/predictions_cache"
    root = _repo_root()
    src_dir = root / _PREDICTIONS_CACHE_DIR_REL

    missing = [n for n in _PREDICTIONS_CACHE_FILES if not (src_dir / n).is_file()]
    if missing:
        print(f"[predcache_upload] local files missing: {missing} — skipping upload")
        return None

    import boto3

    s3 = boto3.client("s3")

    t0 = time.time()
    total_bytes = 0
    try:
        for name in _PREDICTIONS_CACHE_FILES:
            src = src_dir / name
            data = src.read_bytes()
            content_type = (
                "application/json" if name.endswith(".json") else "application/octet-stream"
            )
            s3.put_object(
                Bucket=bucket,
                Key=f"{s3_prefix}/{name}",
                Body=data,
                ContentType=content_type,
            )
            total_bytes += len(data)
    except Exception as e:  # noqa: BLE001 — best-effort upload
        print(f"[predcache_upload] FAILED: {e!r}")
        return None
    total = round(time.time() - t0, 2)
    print(
        f"[predcache_upload] done in {total}s, "
        f"{total_bytes / 1e6:.1f} MB across {len(_PREDICTIONS_CACHE_FILES)} files"
    )
    return {"total_secs": total, "total_bytes": total_bytes, "files": len(_PREDICTIONS_CACHE_FILES)}
