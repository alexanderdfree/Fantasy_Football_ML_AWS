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
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_ENV_BUCKET = "FF_MODEL_S3_BUCKET"
_ENV_PREFIX = "FF_MODEL_S3_PREFIX"

_SPLIT_KEYS = ("train.parquet", "val.parquet", "test.parquet")
_RAW_PREFIX = "data/raw/"
_RAW_EXCLUDE_SUFFIX = "_2023_2023.parquet"

# V3 retains stable/current/previous/history plus publication_source (the main
# ancestry high-water mark) and retired (keys removed by a successful CAS).
# Its releases/ namespace isolates it from queued legacy writers and sweep GC.
# Consumers read legacy v1/v2 only while a position has no v3 manifest.
MANIFEST_SCHEMA_VERSION = 3
HISTORY_KEEP_N = 5


def _repo_root() -> Path:
    # __file__ is at <repo>/src/shared/model_sync.py. Walk THREE ``.parent``
    # hops (model_sync.py → shared/ → src/ → <repo>/) to reach the repo root,
    # which is where the Flask app expects data/ and benchmark_history/ to
    # live (CWD = /app in the container). An earlier version walked only two
    # hops and silently wrote splits/raw under src/data/ instead of
    # <repo>/data/, breaking inference at boot.
    return Path(__file__).resolve().parent.parent.parent


def manifest_key(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/releases/manifest.json"


def history_prefix(prefix: str, pos: str) -> str:
    return f"{prefix}/{pos}/releases/history/"


def new_history_key(prefix: str, pos: str, ts: str, sha7: str) -> str:
    return f"{history_prefix(prefix, pos)}{ts}-{uuid.uuid4().hex}-{sha7}/model.tar.gz"


def load_manifest(s3_client, bucket: str, prefix: str, pos: str) -> dict | None:
    """Prefer v3; read a legacy manifest only before per-position migration."""
    return _load_manifest_with_etag(s3_client, bucket, prefix, pos)[0]


def _load_manifest_with_etag(s3_client, bucket, prefix, pos):
    """Return the manifest and ETag from one GET, or (None, None) if absent.

    Re-raises any other S3 error so the caller can't silently confuse a
    missing manifest with a permissions / transient failure.
    """
    from botocore.exceptions import ClientError

    for key in (manifest_key(prefix, pos), f"{prefix}/{pos}/manifest.json"):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in ("NoSuchKey", "404"):
                continue
            raise
        return json.loads(obj["Body"].read()), obj.get("ETag")
    return None, None


def write_manifest(
    s3_client, bucket: str, prefix: str, pos: str, manifest: dict, *, etag: str | None = None
) -> None:
    """Conditionally publish v3: create if absent, otherwise match the read ETag.

    Callers must build from that same snapshot and reevaluate source order after
    a conflict. Use artifact_publication.publish_artifact for normal publication.
    """
    body = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key(prefix, pos),
        Body=body,
        ContentType="application/json",
        **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
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
    # Extraction can fail after writing some members (for example, when the
    # data filter rejects a later symlink). Stage each candidate separately so
    # the fallback cannot inherit weights or sidecars from a rejected archive.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dest.name}-", dir=dest.parent) as tmp:
        staged = Path(tmp) / "models"
        staged.mkdir()
        staged_resolved = staged.resolve()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                target = (staged / member.name).resolve()
                if staged_resolved not in target.parents and target != staged_resolved:
                    raise RuntimeError(f"Tarball escape attempt: {member.name}")
            tar.extractall(staged, filter="data")
        # Boot-time downloads may replace an existing tree; in-flight refresh
        # passes models.new and performs its live swap only after this returns.
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(staged, dest)


def _try_key(s3_client, bucket: str, key: str, dest: Path) -> dict:
    """Download + extract one tarball key. Raises on any S3 / tar / extract
    error — the caller decides whether to fall back."""
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    _extract_tarball(data, dest)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def _resolve_manifest_extract(s3_client, bucket: str, prefix: str, pos: str, dest: Path) -> dict:
    """Walk manifest.stable → current → previous, extracting the first key that
    successfully downloads + untars into ``dest``. Shared by both the boot-time
    sync (``_sync_one``, extracts into the live ``models/`` dir) and the
    in-flight refresh path (``refresh_position``, extracts into a temporary
    ``models.new/`` so the swap can be atomic).

    Order rationale: the frontend prefers ``stable`` because that's the last
    artifact a smoke test confirmed actually loads + predicts; serving
    ``current`` means stable was missing or unreadable, which is page-worthy.
    A v1-shaped manifest (no ``stable`` slot) reads as ``stable=None`` and
    falls through to ``current`` automatically — that's the migration window.

    Each fall-through is logged with the grep-able tag
    ``source=stable|current|previous`` so on-call can tell from CloudWatch
    which artifact tier we ended up on.

    Raises ``RuntimeError`` if the manifest is absent or every entry fails.
    """
    from botocore.exceptions import ClientError

    manifest, consumed_etag = _load_manifest_with_etag(s3_client, bucket, prefix, pos)

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
        return {"pos": pos, "source": label, "manifest_etag": consumed_etag, **r}

    raise RuntimeError(f"[model_sync] {pos}: all manifest entries failed: {tried!r}")


def _sync_one(s3_client, bucket: str, prefix: str, pos: str, root: Path) -> dict:
    """Sync one position with manifest-driven fallback (boot-time path).

    Extracts directly into ``src/{pos.lower()}/outputs/models``. The boot path
    can tolerate a non-atomic extract — a crash mid-extract aborts the boot
    and ECS won't mark the task healthy, so the broken intermediate state
    never serves traffic. The in-flight refresh path (``refresh_position``)
    uses ``models.new`` + rename for atomicity because live containers can't
    tolerate that intermediate state.

    S3 keys are uppercase POS (set by the producer in src/batch/train.py);
    the local layout is lowercase to match the registry's model_dir entries
    (``src/qb/outputs/models``). On case-sensitive Linux these diverge, so
    the destination must be normalized here even though macOS APFS papered
    over it during the rename refactor.
    """
    dest = root / "src" / pos.lower() / "outputs" / "models"
    result = _resolve_manifest_extract(s3_client, bucket, prefix, pos, dest)
    if isinstance(result.get("manifest_etag"), str):
        (dest.parent / ".manifest-etag").write_text(result["manifest_etag"])
    return result


# ---------------------------------------------------------------------------
# In-flight model refresh — per-position decoupling from ECS deploy
# ---------------------------------------------------------------------------
#
# At boot, ``sync_models_from_s3`` pulls every position. After boot, the
# poller below watches each ``models/{POS}/manifest.json`` and re-syncs only
# the position whose etag changed — letting a Batch fan-out's first-finished
# position (e.g. WR at t≈2min) reach prod without waiting for the slowest one.
#
# Atomicity: extract → rename-swap pattern (see ``refresh_position``). The
# brief window where ``models/`` is missing is invisible to readers — workers
# only consult the on-disk models when ``_ensure_position_loaded`` fires under
# ``_cache_lock``, and the sentinel touch happens *after* the rename, so
# either a worker sees the old mtime (and the early-return on positions_loaded
# trips) or it sees the new mtime (and acquires the lock to re-load from the
# already-renamed dir). The poller and serving runtime never share a lock.
#
# Sentinel: ``src/{pos.lower()}/outputs/.refreshed_at`` (one dir above
# ``models/`` so the swap cannot delete it). The serving code stats this on
# every ``_ensure_position_loaded`` call and re-loads on mtime advance.


# Literal per-position path mappings — the only entry points into the
# filesystem from a (potentially user-tainted) ``pos`` argument. Constructing
# paths from this dict instead of f"src/{pos.lower()}/..." (a) keeps the
# allowlist enforcement structural rather than a value-check sanitizer
# CodeQL may not recognize, and (b) eliminates the path-traversal vector
# entirely — an out-of-allowlist ``pos`` returns ``None`` here and the
# caller short-circuits before any path I/O.
_REFRESH_PATHS_BY_POS: dict[str, dict[str, str]] = {
    pos: {
        "models": f"src/{pos.lower()}/outputs/models",
        "models_new": f"src/{pos.lower()}/outputs/models.new",
        "models_bak": f"src/{pos.lower()}/outputs/models.bak",
        "sentinel": f"src/{pos.lower()}/outputs/.refreshed_at",
    }
    for pos in POSITIONS
}


def _refresh_paths(pos: str) -> dict[str, Path] | None:
    """Resolve the per-position refresh-related paths under the current repo
    root, or ``None`` if ``pos`` is not in the allowlist. The relative
    components are looked up from ``_REFRESH_PATHS_BY_POS`` (literal strings,
    not constructed from ``pos``), so out-of-allowlist values cannot reach
    ``os.path``/``Path`` operations."""
    if not isinstance(pos, str):
        return None
    rels = _REFRESH_PATHS_BY_POS.get(pos.upper())
    if rels is None:
        return None
    root = _repo_root()
    return {name: root / rel for name, rel in rels.items()}


def refresh_sentinel_mtime(pos: str) -> float:
    """Return the mtime of the in-flight refresh sentinel for ``pos``, or 0.0
    if absent. Public read-only counterpart to the poller's ``sentinel.touch()``
    — consumed by ``src.serving.app._ensure_position_loaded`` to detect that
    the on-disk model for ``pos`` has been swapped since the last in-memory
    load, triggering a re-load on the next request.

    ``pos`` is validated against the ``POSITIONS`` allowlist so this function
    is safe to call with any value — anything outside the allowlist returns
    0.0 (the "no refresh pending" sentinel) without touching the filesystem.
    """
    paths = _refresh_paths(pos)
    if paths is None:
        return 0.0
    try:
        return os.path.getmtime(paths["sentinel"])
    except OSError:
        return 0.0


def refresh_position(
    pos: str,
    last_etag: str | None,
    s3_client=None,
) -> tuple[str | None, bool]:
    """Head the manifest for ``pos``; if its etag differs from ``last_etag``,
    re-download via the same stable→current→previous fallback chain as
    ``_sync_one``, extract to ``models.new/``, atomically swap into
    ``models/``, and touch the refresh sentinel.

    Returns ``(new_etag, did_refresh)``.

    First-observation contract: when ``last_etag`` is ``None``, the function
    treats the head as a bootstrap — it records the etag and returns
    ``did_refresh=False`` without re-downloading. This keeps the poller from
    redundantly re-syncing every position immediately after the boot-time
    ``sync_models_from_s3`` already populated them.

    No-op (returns ``(None, False)``) when ``FF_MODEL_S3_BUCKET`` is unset.

    Fail-soft: any error (S3 head, manifest fetch, tarball get, extract,
    rename) is logged and ``did_refresh=False`` is returned with the OLD etag
    so the next poll retries. The live ``models/`` directory is left intact;
    the worst case is the new model stays in S3 unused for one more poll
    interval.

    ``pos`` is validated against the ``POSITIONS`` allowlist so this function
    is safe to call with any value — out-of-allowlist values return
    ``(last_etag, False)`` without touching S3 or the filesystem.
    """
    paths = _refresh_paths(pos)
    if paths is None:
        return last_etag, False
    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        return None, False
    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")

    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    from botocore.exceptions import ClientError

    try:
        try:
            head = s3_client.head_object(Bucket=bucket, Key=manifest_key(prefix, pos))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                raise
            head = s3_client.head_object(Bucket=bucket, Key=f"{prefix}/{pos}/manifest.json")
    except (ClientError, OSError) as e:
        print(f"[refresh] {pos} head_object failed: {e!r}", flush=True)
        return last_etag, False

    new_etag = head.get("ETag")
    etag_path = paths["sentinel"].with_name(".manifest-etag")
    if last_etag is None:
        # Compare with the manifest GET actually consumed at boot. A publication
        # between boot and this first HEAD must trigger a refresh, not bootstrap
        # the new ETag while keeping the old model indefinitely.
        with contextlib.suppress(OSError):
            last_etag = etag_path.read_text().strip() or None
    if new_etag == last_etag:
        return new_etag, False

    if last_etag is None:
        # Bootstrap: record the etag now so the first real change triggers a
        # refresh, not the redundant initial observation.
        print(f"[refresh] {pos} bootstrap etag={new_etag}", flush=True)
        return new_etag, False

    dest = paths["models"]
    dest_new = paths["models_new"]
    dest_bak = paths["models_bak"]
    sentinel = paths["sentinel"]

    # Clean any leftover staging dirs from a prior crashed refresh.
    for stale in (dest_new, dest_bak):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    try:
        result = _resolve_manifest_extract(s3_client, bucket, prefix, pos, dest_new)
    except Exception as e:  # noqa: BLE001 — fail-soft: keep serving old model
        print(f"[refresh] {pos} resolve/extract failed: {e!r} — keeping old model", flush=True)
        if dest_new.exists():
            shutil.rmtree(dest_new, ignore_errors=True)
        return last_etag, False

    # Atomic-ish swap. Rename pairs are O(1) on the same filesystem; the window
    # where ``dest`` is missing is bounded by two renames. Workers serialize
    # on the serving cache lock and only consult disk via the sentinel mtime,
    # so the request fast path cannot observe the gap.
    #
    # KNOWN GAP (low-probability, sticky on hit): a worker that is mid-way
    # through ``_apply_position_models``'s directory listing or first-open
    # phase during the ~microsecond window between these two renames could
    # see ``FileNotFoundError`` and end up cached in ``positions_failed``
    # until the next manifest etag change (could be hours+ away). Open file
    # handles already inside joblib.load / torch.load survive the rename via
    # Linux inode semantics, so the risk is bounded to the listing/first-open
    # phase, not mid-load. If this turns out to bite in production, the fix
    # is symlink-swap (one atomic rename instead of two) or coupling the
    # poller to ``_cache_lock``.
    try:
        if dest.exists():
            os.rename(dest, dest_bak)
        os.rename(dest_new, dest)
    except OSError as e:
        # Rollback: restore the old dir if we moved it out.
        if dest_bak.exists() and not dest.exists():
            with contextlib.suppress(OSError):
                os.rename(dest_bak, dest)
        print(f"[refresh] {pos} swap failed: {e!r} — old model restored", flush=True)
        if dest_new.exists():
            shutil.rmtree(dest_new, ignore_errors=True)
        return last_etag, False

    if dest_bak.exists():
        shutil.rmtree(dest_bak, ignore_errors=True)

    # The pointer can change again between HEAD and GET. Track the version GET
    # actually consumed, so a subsequent poll neither skips nor mislabels it.
    new_etag = result.get("manifest_etag") or new_etag
    with contextlib.suppress(OSError):
        etag_path.write_text(new_etag)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    print(
        f"[refresh] {pos} swapped to new manifest etag={new_etag} source={result['source']}",
        flush=True,
    )
    return new_etag, True


def start_refresh_poller(
    interval_s: int, stop_event: threading.Event | None = None
) -> threading.Thread:
    """Spawn a daemon thread that polls every position's manifest every
    ``interval_s`` seconds and calls ``refresh_position`` when an etag
    changes. Returns the started thread.

    Single thread looping over all 6 positions serially. Per-iteration S3
    load is 6 head_object calls; cost is negligible (~$0.01/month per
    container at 30s interval).

    No-op when ``FF_MODEL_S3_BUCKET`` is unset — ``refresh_position`` returns
    immediately in that case, but the thread still spins. Callers (e.g.
    ``gunicorn.conf.py::on_starting``) gate on the env var to avoid spawning
    a useless thread in dev/CI.

    ``stop_event`` is an optional graceful-shutdown hook: when provided, the
    loop exits once it is set and sleeps on it interruptibly. Production
    callers omit it and rely on ``daemon=True`` for process-exit teardown;
    tests pass one so the thread is joined and never leaks across the suite —
    a leaked spinning poller calls the real ``boto3.client`` and pollutes
    other tests' global boto3 mocks (see tests/shared/test_local_benchmark_sync.py).
    """

    def _loop() -> None:
        etags: dict[str, str | None] = {p: None for p in POSITIONS}
        while stop_event is None or not stop_event.is_set():
            for pos in POSITIONS:
                try:
                    etags[pos], _ = refresh_position(pos, etags[pos])
                except Exception as e:  # noqa: BLE001 — daemon must never die
                    print(f"[refresh] {pos} unexpected: {e!r}", flush=True)
            if stop_event is None:
                time.sleep(interval_s)
            elif stop_event.wait(interval_s):
                break

    t = threading.Thread(target=_loop, daemon=True, name="model-refresh-poll")
    t.start()
    return t


def sync_models_from_s3() -> dict | None:
    """Download+extract all six position tarballs in parallel, preferring
    the manifest-pointed ``stable`` artifact, falling back to ``current``
    then ``previous``.

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


def _download_file(s3_client, bucket: str, key: str, dest: Path, *, atomic: bool = False) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    if atomic:
        # Write a sibling temp then os.replace (atomic on POSIX, same fs) so a
        # concurrent reader never observes a half-written file. Used by the
        # benchmark-history poller, which downloads while the Flask app is
        # serving reads of the same directory — it honors the atomic-rename
        # INVARIANT the row cache relies on (see
        # src.serving.benchmark_history._load_benchmark_history_rows). The temp
        # name is per-(pid, thread) unique so the 8-way download pool can't
        # collide on it.
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except Exception:
            # A mid-rename failure (ENOSPC, EACCES) or a partial temp write must
            # not leave the sibling .tmp behind — the poller runs thousands of
            # times over a container's life, so a persistent failure would
            # otherwise accumulate turds. The download failure itself is recorded
            # in the caller's ``failed`` list once this re-raises.
            tmp.unlink(missing_ok=True)
            raise
    else:
        dest.write_bytes(data)
    return {"key": key, "bytes": len(data), "secs": round(time.time() - t0, 2)}


def sync_benchmark_history_from_s3() -> dict | None:
    """Download every JSON under ``s3://{bucket}/{prefix}/benchmark_history/``
    into the local ``benchmark_history/`` directory.

    Gated on ``FF_MODEL_S3_BUCKET`` like the other syncs — unset/empty makes
    this a no-op so dev and CI tests don't try to hit S3. Fail-soft on an
    empty/missing prefix (a fresh bucket with no benchmark uploads yet
    shouldn't block boot).

    Per-file failures are isolated (M17): a single broken JSON no longer
    kills the whole sync. The container still boots and serves; failed
    files are listed in the returned summary under ``failed`` so an
    operator can grep container logs for them. The Docker image bundles
    the git-tracked floor of ``benchmark_history/`` (see Dockerfile +
    .dockerignore), so this sync is layering on any newer runs uploaded
    since the image was built — if every download fails, the History tab
    still renders the committed history.

    New-file guard: benchmark_history JSONs are append-only and immutable
    (uniquely-named ``{timestamp}_{sha}.json``, atomic-rename writes, never
    edited in place — the same INVARIANT the serving row cache relies on), so
    a filename already on disk is byte-identical to S3 and is skipped, not
    re-fetched. That keeps each call cheap (one ListBucket + a GET only for
    genuinely-new run_ids) — cheap enough to drive ``start_benchmark_history_poller``
    on a short interval so newly-uploaded runs surface on the History tab
    without a container restart. The downloads are atomic so a concurrent
    request mid-poll never reads a half-written file. ``skipped`` in the
    summary counts the already-present files.
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
    skipped = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            dest = dest_dir / Path(key).name
            # New-file guard: immutable run_ids mean an on-disk filename is
            # already byte-identical to S3 — skip the redundant GET (see the
            # docstring). This is what makes the poller's steady state one
            # ListBucket and zero GETs.
            if dest.exists():
                skipped += 1
                continue
            jobs.append((key, dest))

    if not jobs:
        if skipped:
            print(
                f"[benchmark_sync] up to date — {skipped} files already present, "
                f"no new runs under s3://{bucket}/{s3_prefix}"
            )
        else:
            print(f"[benchmark_sync] no objects under s3://{bucket}/{s3_prefix}")
        return {"total_secs": 0.0, "total_bytes": 0, "files": 0, "skipped": skipped, "failed": []}

    print(
        f"[benchmark_sync] syncing {len(jobs)} new files ({skipped} already present) "
        f"from s3://{bucket}/{s3_prefix} -> {dest_dir}"
    )
    t0 = time.time()
    results: list[dict] = []
    failed: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        fut_to_key = {
            pool.submit(_download_file, s3, bucket, key, dest, atomic=True): key
            for key, dest in jobs
        }
        for f in concurrent.futures.as_completed(fut_to_key):
            key = fut_to_key[f]
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 — per-file isolation
                print(f"[benchmark_sync] FAILED for {key}: {e!r}", flush=True)
                failed.append({"key": key, "error": repr(e)})
    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    if failed:
        print(
            f"[benchmark_sync] PARTIAL: {len(results)}/{len(jobs)} files synced; "
            f"failed: {[f['key'] for f in failed]}",
            flush=True,
        )
    print(
        f"[benchmark_sync] done in {total}s, {total_bytes / 1e3:.1f} KB across "
        f"{len(results)} new files ({skipped} already present)"
    )
    return {
        "total_secs": total,
        "total_bytes": total_bytes,
        "files": len(results),
        "skipped": skipped,
        "failed": failed,
    }


def start_benchmark_history_poller(
    interval_s: int, stop_event: threading.Event | None = None
) -> threading.Thread:
    """Spawn a daemon thread that re-syncs ``benchmark_history/`` from S3 every
    ``interval_s`` seconds so a run uploaded after boot appears on the History
    tab without a container restart. Returns the started thread.

    The boot sync (``gunicorn.conf.py::on_starting``) is one-shot; this is its
    in-flight counterpart, mirroring ``start_refresh_poller`` (models) and the
    upcoming-week artifact poller. Cost is dominated by one ListBucket per
    interval — ``sync_benchmark_history_from_s3``'s new-file guard means the
    steady state downloads nothing, and a poll that *does* pull a new run
    writes it atomically into the dir, bumping the directory mtime that the
    serving row cache keys on (``_load_benchmark_history_rows``) so the tab
    refreshes on the next request. A no-new-files poll leaves the mtime
    untouched and the cache stays warm.

    No-op-friendly: ``sync_benchmark_history_from_s3`` returns immediately when
    ``FF_MODEL_S3_BUCKET`` is unset, so even a spinning thread is harmless.
    ``on_starting`` gates on the interval (``FF_BENCHMARK_SYNC_INTERVAL_S=0``
    disables) and only runs under gunicorn in production, where the bucket is
    set. ``stop_event`` mirrors ``start_refresh_poller`` for test teardown — a
    leaked spinning poller calls real boto3 and pollutes other tests' global
    mocks.
    """

    def _loop() -> None:
        while stop_event is None or not stop_event.is_set():
            try:
                sync_benchmark_history_from_s3()
            except Exception as e:  # noqa: BLE001 — daemon must never die
                print(f"[benchmark_sync] poll unexpected: {e!r}", flush=True)
            if stop_event is None:
                time.sleep(interval_s)
            elif stop_event.wait(interval_s):
                break

    t = threading.Thread(target=_loop, daemon=True, name="benchmark-history-poll")
    t.start()
    return t


def sync_data_from_s3() -> dict | None:
    """Download inference data parquets from S3 in parallel.

    Pulls s3://{bucket}/data/{train,val,test}.parquet into data/splits/ and
    every data/raw/*.parquet except the 2023-only duplicates already covered
    by the 2012-2025 range files.

    Per-file failures are isolated (M17): a single broken parquet no longer
    kills the whole sync — the container still boots, the failed file is
    listed in the returned summary under ``failed``, and any feature build
    that touches the missing file surfaces the error per-position via
    ``_apply_position_models``'s outer try/except. A position whose raw
    dependency went missing degrades to per-model NaNs while the rest of
    the site keeps serving (preserving the divergence with
    ``sync_models_from_s3``'s pattern; see PR #236).

    Returns a summary dict (now including ``failed``), or None if
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
    failed: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        fut_to_key = {pool.submit(_download_file, s3, bucket, key, dest): key for key, dest in jobs}
        for f in concurrent.futures.as_completed(fut_to_key):
            key = fut_to_key[f]
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001 — per-file isolation
                print(f"[data_sync] FAILED for {key}: {e!r}", flush=True)
                failed.append({"key": key, "error": repr(e)})
    total = round(time.time() - t0, 2)
    total_bytes = sum(r["bytes"] for r in results)
    if failed:
        print(
            f"[data_sync] PARTIAL: {len(results)}/{len(jobs)} files synced; "
            f"failed: {[f['key'] for f in failed]}",
            flush=True,
        )
    print(f"[data_sync] done in {total}s, {total_bytes / 1e6:.1f} MB across {len(results)} files")
    return {
        "total_secs": total,
        "total_bytes": total_bytes,
        "files": len(results),
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Predictions cache (serving pre-warm)
# ---------------------------------------------------------------------------
#
# The Flask app caches the assembled predictions DataFrame + metrics on disk
# after the first ``_ensure_metrics()`` run. Uploading that cache to S3 lets
# the next ECS task hydrate without recomputing — fingerprint mismatch on
# the consumer side guards against serving stale predictions against
# refreshed models. See src/serving/core.py::_try_hydrate_from_disk and
# _persist_cache_to_disk for the producer/consumer side.

_PREDICTIONS_CACHE_DIR_REL = "data/serving_cache"
_PREDICTIONS_CACHE_FILES = ("predictions.parquet", "metrics.json", "fingerprint.json")
_PREDICTIONS_CACHE_OPTIONAL = ("snapshot.json",)


def sync_predictions_cache_from_s3() -> dict | None:
    """Install one verified cache bundle; failed sync preserves the prior generation.

    The new key isolates this protocol from queued jobs/old ECS images that still
    upload loose files. Those files cannot prove a common generation and are not
    used as a fallback. A missing bundle follows the existing cache-miss path.
    """
    from src.shared import prediction_cache

    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        print(f"[predcache_sync] {_ENV_BUCKET} unset — skipping S3 sync")
        return None
    import boto3
    from botocore.exceptions import ClientError

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    key = f"{prefix}/predictions_cache/{prediction_cache.BUNDLE_NAME}"
    started = time.time()
    try:
        data = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        generation = prediction_cache.install_bundle(
            _repo_root() / _PREDICTIONS_CACHE_DIR_REL, data
        )
    except Exception as exc:  # noqa: BLE001 — boot can compute on a cache miss
        missing = isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") in (
            "NoSuchKey",
            "404",
        )
        print(f"[predcache_sync] bundle unavailable: {exc!r} — prior generation retained")
        return {"files": 0, "missing": [key] if missing else [], "failed": [] if missing else [key]}
    return {
        "total_secs": round(time.time() - started, 2),
        "total_bytes": len(data),
        "files": len(_PREDICTIONS_CACHE_FILES),
        "generation": generation.name,
    }


def upload_predictions_cache_to_s3() -> dict | None:
    """Publish one complete generation with one atomic S3 object replacement."""
    from src.shared import prediction_cache

    bucket = os.environ.get(_ENV_BUCKET, "").strip()
    if not bucket:
        return None
    import boto3

    prefix = os.environ.get(_ENV_PREFIX, "models").strip("/")
    key = f"{prefix}/predictions_cache/{prediction_cache.BUNDLE_NAME}"
    started = time.time()
    try:
        generation, data = prediction_cache.bundle_generation(
            _repo_root() / _PREDICTIONS_CACHE_DIR_REL
        )
        boto3.client("s3").put_object(
            Bucket=bucket, Key=key, Body=data, ContentType="application/gzip"
        )
    except Exception as exc:  # noqa: BLE001 — cache publication must not break serving
        print(f"[predcache_upload] FAILED: {exc!r}")
        return None
    print(f"[predcache_upload] published generation {generation[:12]}")
    return {
        "total_secs": round(time.time() - started, 2),
        "total_bytes": len(data),
        "files": len(_PREDICTIONS_CACHE_FILES),
        "generation": generation,
    }
