"""Tests for src.shared.model_sync — S3 tarball sync at container boot."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shared import model_sync
from tests.shared._helpers import FakeBody as _FakeBody
from tests.shared._helpers import make_tarball as _make_tarball


def _manifest_bytes(
    current_key: str,
    previous_key: str | None = None,
    stable_key: str | None = None,
    sha7: str = "abc1234",
    bytes_: int = 4096,
    schema_version: int = 2,
) -> bytes:
    """Build a well-formed manifest.json body pointing at the given keys.

    Kept in this test module (not src.shared.model_sync) so that tests exercise
    the exact JSON shape a real producer would write — drift between
    build_manifest and the consumer's schema expectations would show up here.

    ``schema_version=1`` omits the ``stable`` field entirely so the
    backwards-compat path (consumer reading a pre-migration manifest) can be
    exercised. ``schema_version=2`` includes ``stable`` (None unless
    ``stable_key`` is set).
    """
    current = {
        "key": current_key,
        "sha7": sha7,
        "bytes": bytes_,
        "uploaded_at": "2026-04-23T00-00-00Z",
    }
    previous = None
    if previous_key is not None:
        previous = {
            "key": previous_key,
            "sha7": "prev1234"[:7],
            "bytes": bytes_,
            "uploaded_at": "2026-04-22T00-00-00Z",
        }
    body: dict = {
        "schema_version": schema_version,
        "current": current,
        "previous": previous,
        "history": [current_key] + ([previous_key] if previous_key else []),
    }
    if schema_version >= 2:
        stable = None
        if stable_key is not None:
            stable = {
                "key": stable_key,
                "sha7": "stab123",
                "bytes": bytes_,
                "uploaded_at": "2026-04-21T00-00-00Z",
            }
        body["stable"] = stable
    return json.dumps(body).encode("utf-8")


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": k} for k in self._objects if k.startswith(Prefix)]
        yield {"Contents": contents}


def _nosuchkey_error(key: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": f"{key} not found"}},
        operation_name="GetObject",
    )


class _FakeS3:
    """Returns per-key object bodies; also paginates by prefix for ListBucket.

    Missing keys raise ``botocore.exceptions.ClientError`` with code
    ``NoSuchKey`` to mirror real S3 semantics — ``src.shared.model_sync``
    distinguishes that from other errors when falling back between
    ``current`` and ``previous`` manifest entries.
    """

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects
        self.calls: list[tuple[str, str]] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 (boto3 convention)
        self.calls.append((Bucket, Key))
        if Key not in self._objects:
            raise _nosuchkey_error(Key)
        return {"Body": _FakeBody(self._objects[Key])}

    def get_paginator(self, op: str):
        assert op == "list_objects_v2"
        return _FakePaginator(self._objects)


@pytest.mark.unit
def test_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    result = model_sync.sync_models_from_s3()
    assert result is None
    assert "unset" in capsys.readouterr().out


@pytest.mark.unit
def test_sync_noop_when_bucket_blank(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "   ")
    assert model_sync.sync_models_from_s3() is None


@pytest.mark.unit
def test_repo_root_resolves_to_actual_repo_root():
    """Contract: ``_repo_root()`` must return the directory the deployed Flask
    app's CWD resolves to (``/app`` in the container), not a subdirectory.

    Guards against the post-#150 regression: when ``model_sync.py`` moved
    from ``<repo>/shared/`` to ``<repo>/src/shared/`` the ``parent.parent``
    chain stopped at ``<repo>/src/``, so ``sync_data_from_s3`` silently wrote
    splits to ``<repo>/src/data/splits/`` while the app reads from
    ``<repo>/data/splits/``. The previous tests only checked the relative
    offset under a monkeypatched root, so they couldn't catch a mis-anchored
    root in production. Asserting the un-monkeypatched root contains both
    ``src/serving/app.py`` and ``requirements.txt`` pins the live invariant.
    """
    root = model_sync._repo_root()
    assert (root / "src" / "serving" / "app.py").is_file(), (
        f"_repo_root() = {root!s} but src/serving/app.py is not under it; "
        "the function is anchored above or below the actual repo root."
    )
    assert (root / "requirements.txt").is_file(), (
        f"_repo_root() = {root!s} but requirements.txt is not under it; "
        "the function is anchored above or below the actual repo root."
    )


@pytest.mark.unit
def test_sync_one_dest_string_path_matches_registry_model_dir(monkeypatch, tmp_path):
    """Contract: ``_sync_one``'s extraction directory must equal — *as a
    case-sensitive path string* — the directory ``registry.get_inference_spec
    (pos)['model_dir']`` resolves to. Files must land where the Flask app
    will look for them.

    Why string compare and not ``Path.samefile``: the bug we're guarding
    against is uppercase-vs-lowercase position divergence that survived PR
    #154's rename — model_sync kept extracting to the uppercase-POS path
    while the registry started reading from the lowercase one. macOS APFS
    folded the two paths to one inode and the existing tests passed; ECS
    Linux is case-sensitive and the deployed app would have failed to load
    any model. A string-based assertion is the only kind that catches this
    on a developer's mac before it hits prod.
    """
    captured: dict[str, Path] = {}

    def fake_extract(_data: bytes, dest: Path) -> None:
        # Record the dest each call; skip the actual untar — we don't need
        # real artifacts, just the path string the caller built.
        captured[fake_extract.current_pos] = dest
        dest.mkdir(parents=True, exist_ok=True)

    fake_extract.current_pos = ""  # populated per-iteration below

    monkeypatch.setattr(model_sync, "_extract_tarball", fake_extract)

    # ``_repo_root`` returns the actual repo root; ``_sync_one`` is responsible
    # for prepending ``src/`` so the dest matches the registry's
    # ``src/{pos}/outputs/models`` model_dir.
    fake_tar = _make_tarball({"sentinel": b"x"})
    # Use the manifest path — Layer C of the race fix removed the legacy
    # fallback, so an absent manifest raises. The dest-string computation is
    # identical regardless of which manifest tier the sync resolves through.
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        history_key = f"models/{pos}/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
        objects[history_key] = fake_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=history_key)
    fake_s3 = _FakeS3(objects)

    from src.shared.registry import get_inference_spec

    for pos in model_sync.POSITIONS:
        fake_extract.current_pos = pos
        model_sync._sync_one(fake_s3, "test-bucket", "models", pos, tmp_path)

    for pos in model_sync.POSITIONS:
        sync_dest_str = str(captured[pos])
        registry_rel = get_inference_spec(pos)["model_dir"]
        registry_dest_str = str(tmp_path / registry_rel)
        assert sync_dest_str == registry_dest_str, (
            f"{pos}: model_sync._extract_tarball(dest='{sync_dest_str}'), "
            f"but registry says serving reads from '{registry_dest_str}'. "
            "Path strings must match exactly (case-sensitive) so synced "
            "artifacts are reachable through the serving path on Linux."
        )


@pytest.mark.unit
def test_sync_honors_custom_prefix(monkeypatch, tmp_path):
    """Custom FF_MODEL_S3_PREFIX threads through both the manifest probe and
    the resolved history-key fetch — ``nightly/v2/{POS}/manifest.json`` and
    the manifest's ``nightly/v2/{POS}/history/.../model.tar.gz`` both."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "nightly/v2")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    tar = _make_tarball({"file.pkl": b"x"})
    objects: dict[str, bytes] = {}
    history_keys: dict[str, str] = {}
    for pos in model_sync.POSITIONS:
        key = f"nightly/v2/{pos}/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
        history_keys[pos] = key
        objects[key] = tar
        objects[f"nightly/v2/{pos}/manifest.json"] = _manifest_bytes(current_key=key)
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        model_sync.sync_models_from_s3()

    keys_called = {key for _, key in fake_s3.calls}
    for pos in model_sync.POSITIONS:
        assert f"nightly/v2/{pos}/manifest.json" in keys_called
        assert history_keys[pos] in keys_called


@pytest.mark.unit
def test_sync_raises_on_missing_manifest(monkeypatch, tmp_path):
    """Layer C: no manifest at all → loud RuntimeError. Previously this fell
    through to the legacy ``model.tar.gz`` key and only raised on ClientError
    if the legacy was also missing. Now manifest-absence itself is the bug."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fake_s3 = _FakeS3(objects={})
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(RuntimeError, match="no manifest"):
            model_sync.sync_models_from_s3()


# --- Manifest-aware sync: current / previous fallback + legacy migration ---


def _build_objects_for_all_positions(current_tarball: bytes) -> dict[str, bytes]:
    """Build a manifest + versioned tarball for EVERY position so the full
    ``sync_models_from_s3`` parallel fan-out doesn't fail on positions this
    test doesn't care about."""
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        key = f"models/{pos}/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
        objects[key] = current_tarball
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=key)
    return objects


@pytest.mark.unit
def test_sync_one_prefers_current_from_manifest(monkeypatch, tmp_path):
    """Happy path: manifest.current points at a history/ key, consumer pulls it
    and reports source=current."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    tar = _make_tarball({"nn_scaler.pkl": b"CURRENT"})
    objects = _build_objects_for_all_positions(tar)
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    wr = next(r for r in summary["positions"] if r["pos"] == "WR")
    assert wr["source"] == "current"
    assert wr["key"] == "models/WR/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
    assert (
        tmp_path / "src" / "wr" / "outputs" / "models" / "nn_scaler.pkl"
    ).read_bytes() == b"CURRENT"


@pytest.mark.unit
def test_sync_one_falls_back_to_previous_when_current_corrupt(monkeypatch, tmp_path, capsys):
    """Current points at a valid key in S3 but the bytes aren't a gzip tarball
    (e.g. a truncated upload slipped past validation, or S3 replication is
    mid-flight). _sync_one must catch the tarfile error, try previous, and
    log source=previous so on-call can grep for it.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    good_tar = _make_tarball({"marker.pkl": b"PREVIOUS_GOOD"})
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/2026-04-23T00-00-00Z-newnew1/model.tar.gz"
        prev_key = f"models/{pos}/history/2026-04-22T00-00-00Z-old0000/model.tar.gz"
        if pos == "QB":
            objects[cur_key] = b"NOT A GZIP TARBALL"
        else:
            objects[cur_key] = good_tar
        objects[prev_key] = good_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, previous_key=prev_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    qb = next(r for r in summary["positions"] if r["pos"] == "QB")
    assert qb["source"] == "previous"
    out = capsys.readouterr().out
    # On-call greps CloudWatch for these tags. Keep the grep-surface stable.
    assert "source=previous" in out
    assert "QB current" in out and "FAILED" in out
    # Other positions still serve current — one broken artifact doesn't poison
    # the fan-out.
    assert all(r["source"] == "current" for r in summary["positions"] if r["pos"] != "QB")


@pytest.mark.unit
def test_sync_one_falls_back_to_previous_when_current_nosuchkey(monkeypatch, tmp_path):
    """Current pointer exists in manifest but the actual key is missing from
    S3 (e.g. GC deleted it by mistake, or manifest-write succeeded but
    upload was rolled back). _sync_one must catch ClientError and retry
    with previous."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    good_tar = _make_tarball({"marker.pkl": b"FROM_PREVIOUS"})
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/missing-current/model.tar.gz"
        prev_key = f"models/{pos}/history/2026-04-22T00-00-00Z-old0000/model.tar.gz"
        # cur_key deliberately NOT added to objects.
        objects[prev_key] = good_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, previous_key=prev_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "previous" for r in summary["positions"])
    for pos in model_sync.POSITIONS:
        extracted = tmp_path / "src" / pos.lower() / "outputs" / "models" / "marker.pkl"
        assert extracted.read_bytes() == b"FROM_PREVIOUS"


@pytest.mark.unit
def test_sync_one_raises_when_both_current_and_previous_fail(monkeypatch, tmp_path):
    """Manifest points at two broken artifacts. We deliberately do NOT fall
    back to the legacy key here — if a manifest exists, that's the contract,
    and "my current+previous both broke" is a real bug that should block
    deploy, not something to paper over with an older stale copy.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/cur-broken/model.tar.gz"
        prev_key = f"models/{pos}/history/prev-broken/model.tar.gz"
        objects[cur_key] = b"not-gzip-A"
        objects[prev_key] = b"not-gzip-B"
        # Legacy key IS present — must be ignored when a manifest exists.
        objects[f"models/{pos}/model.tar.gz"] = _make_tarball({"marker.pkl": b"LEGACY"})
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, previous_key=prev_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(RuntimeError, match="all manifest entries failed"):
            model_sync.sync_models_from_s3()


@pytest.mark.unit
def test_sync_one_falls_back_on_truncated_gzip(monkeypatch, tmp_path):
    """A truncated gzip (valid header, cut-off payload) raises ``EOFError``
    from gzip.py, not ``tarfile.TarError``. The consumer must catch that
    shape too, otherwise a replication-lag partial upload takes the site
    down instead of triggering fallback."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    good_tar = _make_tarball({"marker.pkl": b"GOOD"})
    # Valid gzip header but truncated mid-stream.
    truncated = good_tar[:64]

    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/cur-truncated/model.tar.gz"
        prev_key = f"models/{pos}/history/prev/model.tar.gz"
        objects[cur_key] = truncated
        objects[prev_key] = good_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, previous_key=prev_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "previous" for r in summary["positions"])


@pytest.mark.unit
def test_sync_one_raises_when_current_fails_and_previous_is_null(monkeypatch, tmp_path):
    """First-ever post-migration run: previous is None. If current fails
    there's nowhere to fall back to, and the raise blocks the rollout —
    same blast radius as today, documented in the plan as acceptable."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/cur-broken/model.tar.gz"
        objects[cur_key] = b"not-gzip"
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=cur_key)

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(RuntimeError, match="all manifest entries failed"):
            model_sync.sync_models_from_s3()


# --- Per-position failure isolation ---


@pytest.mark.unit
def test_sync_isolates_single_position_failure(monkeypatch, tmp_path, capsys):
    """5 positions have working tarballs, 1 has manifest pointing at a broken
    artifact with no previous. The single failure must NOT propagate — the
    healthy 5 should sync successfully and the failure should be reported
    via ``failed_positions`` in the summary so a partial sync starts the
    container instead of taking the whole site down."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    healthy_tar = _make_tarball({"nn_scaler.pkl": b"HEALTHY"})
    broken_pos = "RB"
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        if pos == broken_pos:
            cur_key = f"models/{pos}/history/cur-broken/model.tar.gz"
            objects[cur_key] = b"not-gzip"
            objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=cur_key)
        else:
            cur_key = f"models/{pos}/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
            objects[cur_key] = healthy_tar
            objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=cur_key)

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert summary is not None
    synced = {r["pos"] for r in summary["positions"]}
    assert synced == set(model_sync.POSITIONS) - {broken_pos}
    assert len(summary["positions"]) == len(model_sync.POSITIONS) - 1
    assert [f["pos"] for f in summary["failed_positions"]] == [broken_pos]
    assert "all manifest entries failed" in summary["failed_positions"][0]["error"]

    captured = capsys.readouterr().out
    assert f"FAILED for {broken_pos}" in captured
    assert "PARTIAL: 5/6" in captured


@pytest.mark.unit
def test_sync_summary_includes_empty_failed_positions_on_full_success(monkeypatch, tmp_path):
    """When every position succeeds the summary still carries an empty
    ``failed_positions`` list so observability code can read the field
    unconditionally."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    tar = _make_tarball({"nn_scaler.pkl": b"OK"})
    objects = _build_objects_for_all_positions(tar)

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert summary is not None
    assert len(summary["positions"]) == len(model_sync.POSITIONS)
    assert summary["failed_positions"] == []


@pytest.mark.unit
def test_sync_reraises_first_exception_when_every_position_fails(monkeypatch, tmp_path):
    """When 0 positions sync the original exception class is re-raised
    (not a synthetic aggregate), preserving the existing useful error
    message and exception type for log inspection — the
    ``test_sync_one_raises_when_*`` tests above lock in the message text
    for the all-manifest-broken case; this one covers the mixed case where
    different positions fail with different exception classes."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    # Half the positions have no manifest + no legacy key -> ClientError.
    # Half have a manifest pointing at a broken artifact -> RuntimeError.
    # All 6 fail; the first per-position exception (whichever finishes
    # first under the thread pool) should be re-raised — its type must be
    # one of the two we know ``_sync_one`` produces.
    objects: dict[str, bytes] = {}
    half = len(model_sync.POSITIONS) // 2
    for pos in model_sync.POSITIONS[:half]:
        cur_key = f"models/{pos}/history/cur-broken/model.tar.gz"
        objects[cur_key] = b"not-gzip"
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(current_key=cur_key)
    # The other half intentionally has no manifest and no legacy key.

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises((RuntimeError, ClientError)):
            model_sync.sync_models_from_s3()


# --- Manifest v2: stable-first fallback chain ---


@pytest.mark.unit
def test_sync_one_prefers_stable_over_current(monkeypatch, tmp_path):
    """Happy path under the new contract: when the manifest names a stable
    artifact (from a passing smoke test on the writer side), ``_sync_one``
    pulls THAT artifact and reports source=stable. The current slot is left
    untouched in S3 — current is whatever the latest upload was, even if its
    smoke test failed."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    stable_tar = _make_tarball({"marker.pkl": b"FROM_STABLE"})
    current_tar = _make_tarball({"marker.pkl": b"FROM_CURRENT"})

    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/2026-04-25T00-00-00Z-newnew1/model.tar.gz"
        stable_key = f"models/{pos}/history/2026-04-23T00-00-00Z-stab123/model.tar.gz"
        objects[cur_key] = current_tar
        objects[stable_key] = stable_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, stable_key=stable_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "stable" for r in summary["positions"])
    for pos in model_sync.POSITIONS:
        extracted = tmp_path / "src" / pos.lower() / "outputs" / "models" / "marker.pkl"
        assert extracted.read_bytes() == b"FROM_STABLE"
    # Current key bytes must NOT have been pulled — the stable key wins
    # outright and we don't probe further when stable succeeds.
    keys_called = {key for _, key in fake_s3.calls}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/2026-04-25T00-00-00Z-newnew1/model.tar.gz"
        assert cur_key not in keys_called


@pytest.mark.unit
def test_sync_one_falls_through_when_stable_missing_v1_manifest(monkeypatch, tmp_path):
    """Backwards compat: a v1-shaped manifest has no ``stable`` field at all.
    Consumer must treat that as "no stable yet" and fall through to current
    without erroring. This is the migration window when the new producer
    rolls out — until the first post-deploy training run sets ``stable``,
    the frontend serves ``current``."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    tar = _make_tarball({"marker.pkl": b"V1_CURRENT"})
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/v1-cur/model.tar.gz"
        objects[cur_key] = tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, schema_version=1
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "current" for r in summary["positions"])


@pytest.mark.unit
def test_sync_one_falls_through_when_stable_corrupt(monkeypatch, tmp_path, capsys):
    """Stable points at a key whose bytes are corrupt (e.g. a delete-and-
    rewrite race or a flipped bit on disk). Consumer falls through to current
    rather than raising — but logs the failure so on-call sees the
    degradation."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    good_tar = _make_tarball({"marker.pkl": b"FROM_CURRENT"})
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/cur/model.tar.gz"
        stable_key = f"models/{pos}/history/stable-broken/model.tar.gz"
        objects[stable_key] = b"NOT A GZIP TARBALL"
        objects[cur_key] = good_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, stable_key=stable_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "current" for r in summary["positions"])
    out = capsys.readouterr().out
    # The stable failure must be loud — it's page-worthy in production.
    assert "stable" in out and "FAILED" in out


@pytest.mark.unit
def test_sync_one_full_chain_falls_to_previous_when_stable_and_current_corrupt(
    monkeypatch, tmp_path
):
    """Stable + current both broken, previous good — must fall through the
    full chain to previous. Establishes that the chain is stable→current→
    previous, not stable→previous (skipping current)."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    good_tar = _make_tarball({"marker.pkl": b"FROM_PREVIOUS"})
    objects: dict[str, bytes] = {}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/cur-broken/model.tar.gz"
        prev_key = f"models/{pos}/history/prev-good/model.tar.gz"
        stable_key = f"models/{pos}/history/stable-broken/model.tar.gz"
        objects[cur_key] = b"BROKEN_CUR"
        objects[stable_key] = b"BROKEN_STABLE"
        objects[prev_key] = good_tar
        objects[f"models/{pos}/manifest.json"] = _manifest_bytes(
            current_key=cur_key, previous_key=prev_key, stable_key=stable_key
        )

    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert all(r["source"] == "previous" for r in summary["positions"])


# --- Pure-function tests for build_manifest ---


@pytest.mark.unit
def test_build_manifest_first_write_has_null_previous():
    m = model_sync.build_manifest(
        new_key="models/QB/history/t1/model.tar.gz",
        sha7="abc1234",
        bytes_=1000,
        uploaded_at="2026-04-23T00-00-00Z",
        old_manifest=None,
    )
    assert m["schema_version"] == 2
    assert m["current"]["key"] == "models/QB/history/t1/model.tar.gz"
    assert m["previous"] is None
    # Default smoke_passed=False on first write — stable is null until a
    # smoke test passes.
    assert m["stable"] is None
    assert m["history"] == ["models/QB/history/t1/model.tar.gz"]


@pytest.mark.unit
def test_build_manifest_promotes_old_current_to_previous():
    old = {
        "current": {"key": "old-cur", "sha7": "old1234", "bytes": 1, "uploaded_at": "t0"},
        "previous": {"key": "old-prev", "sha7": "prv1234", "bytes": 1, "uploaded_at": "t-1"},
        "history": ["old-cur", "old-prev", "old-older"],
    }
    m = model_sync.build_manifest(
        new_key="new-cur",
        sha7="new1234",
        bytes_=2000,
        uploaded_at="t+1",
        old_manifest=old,
    )
    assert m["current"]["key"] == "new-cur"
    assert m["previous"] == old["current"]
    # Newest-first and capped; old.previous has already been demoted out.
    assert m["history"][0] == "new-cur"
    assert "old-cur" in m["history"]
    assert len(m["history"]) <= model_sync.HISTORY_KEEP_N


@pytest.mark.unit
def test_build_manifest_caps_history_at_keep_n():
    old_history = [f"k{i}" for i in range(model_sync.HISTORY_KEEP_N + 3)]
    old = {
        "current": {"key": old_history[0], "sha7": "x" * 7, "bytes": 1, "uploaded_at": "t"},
        "previous": None,
        "history": old_history,
    }
    m = model_sync.build_manifest(
        new_key="brand-new",
        sha7="new1234",
        bytes_=1,
        uploaded_at="t+1",
        old_manifest=old,
    )
    assert len(m["history"]) == model_sync.HISTORY_KEEP_N
    assert m["history"][0] == "brand-new"


@pytest.mark.unit
def test_build_manifest_smoke_passed_advances_stable():
    """A passing smoke test promotes the new entry into the ``stable`` slot.
    When there is no prior manifest, ``stable`` and ``current`` agree."""
    m = model_sync.build_manifest(
        new_key="models/QB/history/t1/model.tar.gz",
        sha7="abc1234",
        bytes_=1000,
        uploaded_at="2026-04-23T00-00-00Z",
        old_manifest=None,
        smoke_passed=True,
    )
    assert m["stable"]["key"] == "models/QB/history/t1/model.tar.gz"
    assert m["stable"] == m["current"]


@pytest.mark.unit
def test_build_manifest_smoke_failed_pins_old_stable():
    """A failing smoke test does NOT advance ``stable`` — the prior good
    pointer carries forward verbatim. Current and history still update so
    the artifact is auditable in S3."""
    old_stable = {
        "key": "old-stable",
        "sha7": "stb1234",
        "bytes": 1,
        "uploaded_at": "t-2",
    }
    old = {
        "current": {"key": "old-cur", "sha7": "cur1234", "bytes": 1, "uploaded_at": "t-1"},
        "stable": old_stable,
        "previous": {"key": "old-prev", "sha7": "prv1234", "bytes": 1, "uploaded_at": "t-3"},
        "history": ["old-cur", "old-prev"],
    }
    m = model_sync.build_manifest(
        new_key="new-broken",
        sha7="brk1234",
        bytes_=1,
        uploaded_at="t",
        old_manifest=old,
        smoke_passed=False,
    )
    assert m["current"]["key"] == "new-broken"
    assert m["stable"] == old_stable, "stable must not move when smoke fails"
    assert m["previous"] == old["current"]


@pytest.mark.unit
def test_build_manifest_smoke_failed_first_run_leaves_stable_null():
    """First-ever upload with a failing smoke test — there's no prior stable
    to pin to, so stable starts as None. Migration window: the consumer
    falls through to ``current`` until the next passing smoke test."""
    m = model_sync.build_manifest(
        new_key="brand-new-broken",
        sha7="brk1234",
        bytes_=1,
        uploaded_at="t",
        old_manifest=None,
        smoke_passed=False,
    )
    assert m["current"]["key"] == "brand-new-broken"
    assert m["stable"] is None


@pytest.mark.unit
def test_build_manifest_smoke_passed_after_prior_failure_advances_stable():
    """A retrain that passes smoke test after a stretch of failures advances
    stable to the new key, leapfrogging the old pinned stable."""
    old_stable = {"key": "old-stable", "sha7": "stb1234", "bytes": 1, "uploaded_at": "t-2"}
    old = {
        "current": {"key": "broken-cur", "sha7": "brk1234", "bytes": 1, "uploaded_at": "t-1"},
        "stable": old_stable,
        "previous": None,
        "history": ["broken-cur"],
    }
    m = model_sync.build_manifest(
        new_key="new-good",
        sha7="good123",
        bytes_=1,
        uploaded_at="t",
        old_manifest=old,
        smoke_passed=True,
    )
    assert m["stable"]["key"] == "new-good"
    assert m["current"]["key"] == "new-good"


@pytest.mark.unit
def test_build_manifest_idempotent_on_same_new_key():
    """If a retry uploads the same bytes to the same versioned key, the new
    key shouldn't get duplicated in history."""
    old = {
        "current": {"key": "k1", "sha7": "1234567", "bytes": 1, "uploaded_at": "t"},
        "previous": None,
        "history": ["k1", "k0"],
    }
    m = model_sync.build_manifest(
        new_key="k1",
        sha7="1234567",
        bytes_=1,
        uploaded_at="t",
        old_manifest=old,
    )
    assert m["history"].count("k1") == 1


@pytest.mark.unit
def test_extract_rejects_path_traversal(tmp_path):
    malicious = _make_tarball({"../../../etc/evil.pkl": b"pwn"})
    with pytest.raises(RuntimeError, match="escape"):
        model_sync._extract_tarball(malicious, tmp_path / "dest")


@pytest.mark.unit
def test_extract_allows_nested_subdirs(tmp_path):
    data = _make_tarball(
        {
            "nn_scaler.pkl": b"a",
            "lightgbm/receiving_yards.pkl": b"b",
        }
    )
    dest = tmp_path / "dest"
    model_sync._extract_tarball(data, dest)
    assert (dest / "nn_scaler.pkl").read_bytes() == b"a"
    assert (dest / "lightgbm" / "receiving_yards.pkl").read_bytes() == b"b"


@pytest.mark.unit
def test_data_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert model_sync.sync_data_from_s3() is None
    assert "unset" in capsys.readouterr().out


@pytest.mark.unit
def test_data_sync_downloads_splits_and_raw(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        "data/train.parquet": b"TRAIN",
        "data/val.parquet": b"VAL",
        "data/test.parquet": b"TEST",
        "data/raw/weekly_2012_2025.parquet": b"WEEKLY",
        "data/raw/schedules_2012_2025.parquet": b"SCHED",
        "data/raw/weekly_2023_2023.parquet": b"SHOULD_SKIP",
        "data/raw/notes.txt": b"SHOULD_SKIP_TOO",
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_data_from_s3()

    assert summary is not None
    assert summary["files"] == 5
    assert (tmp_path / "data" / "splits" / "train.parquet").read_bytes() == b"TRAIN"
    assert (tmp_path / "data" / "splits" / "val.parquet").read_bytes() == b"VAL"
    assert (tmp_path / "data" / "splits" / "test.parquet").read_bytes() == b"TEST"
    assert (tmp_path / "data" / "raw" / "weekly_2012_2025.parquet").read_bytes() == b"WEEKLY"
    assert (tmp_path / "data" / "raw" / "schedules_2012_2025.parquet").read_bytes() == b"SCHED"
    assert not (tmp_path / "data" / "raw" / "weekly_2023_2023.parquet").exists()
    assert not (tmp_path / "data" / "raw" / "notes.txt").exists()


@pytest.mark.unit
def test_data_sync_isolates_per_file_failures(monkeypatch, tmp_path, capsys):
    """M17: a missing split (or any individual download failure) no longer
    kills the whole sync. The container boots, the failed key is listed in
    the returned summary's ``failed`` field, and any feature build that
    needs the missing file surfaces the error per-position via
    ``_apply_position_models``'s outer try/except.

    Previously this raised ClientError(NoSuchKey) and gunicorn --preload
    aborted boot, taking down the whole site for one missing parquet.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    # Only one of the three splits is present; val and test will 404.
    fake_s3 = _FakeS3(
        objects={
            "data/train.parquet": b"TRAIN",
            "data/raw/weekly_2012_2025.parquet": b"WEEKLY",
        }
    )
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_data_from_s3()
    assert summary is not None
    # train + weekly succeeded; val + test failed.
    assert summary["files"] == 2
    failed_keys = sorted(item["key"] for item in summary["failed"])
    assert failed_keys == ["data/test.parquet", "data/val.parquet"]
    # Successful downloads landed on disk.
    assert (tmp_path / "data" / "splits" / "train.parquet").read_bytes() == b"TRAIN"
    # Failed-file paths weren't created.
    assert not (tmp_path / "data" / "splits" / "val.parquet").exists()
    assert not (tmp_path / "data" / "splits" / "test.parquet").exists()
    # Operator-visible log line.
    assert "PARTIAL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sync_benchmark_history_from_s3
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_benchmark_history_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert model_sync.sync_benchmark_history_from_s3() is None
    assert "unset" in capsys.readouterr().out


@pytest.mark.unit
def test_benchmark_history_sync_empty_prefix_is_not_an_error(monkeypatch, tmp_path):
    """A fresh bucket without any benchmark uploads shouldn't block boot."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fake_s3 = _FakeS3(objects={})
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_benchmark_history_from_s3()
    assert summary == {"total_secs": 0.0, "total_bytes": 0, "files": 0, "failed": []}


@pytest.mark.unit
def test_benchmark_history_sync_downloads_every_json(monkeypatch, tmp_path):
    """All ``*.json`` under the prefix land in ``<root>/benchmark_history/``;
    non-JSON keys are filtered out."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        "models/benchmark_history/2026-05-01T10-00-00_abc.json": b'{"a": 1}',
        "models/benchmark_history/2026-05-19T22-47-20_dff43fb.json": b'{"b": 2}',
        # README-style key adjacent to the JSONs — must be skipped, not crash.
        "models/benchmark_history/README.txt": b"ignore me",
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_benchmark_history_from_s3()

    assert summary is not None
    assert summary["files"] == 2
    dest = tmp_path / "benchmark_history"
    assert (dest / "2026-05-01T10-00-00_abc.json").read_bytes() == b'{"a": 1}'
    assert (dest / "2026-05-19T22-47-20_dff43fb.json").read_bytes() == b'{"b": 2}'
    assert not (dest / "README.txt").exists()


@pytest.mark.unit
def test_benchmark_history_sync_respects_custom_prefix(monkeypatch, tmp_path):
    """``FF_MODEL_S3_PREFIX`` is honored so dev/staging buckets can carve
    out their own namespace."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "nightly/v2")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        # Under the custom prefix — should be pulled.
        "nightly/v2/benchmark_history/x.json": b'{"x": true}',
        # Under the default prefix — should be ignored when prefix is overridden.
        "models/benchmark_history/y.json": b'{"y": true}',
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_benchmark_history_from_s3()
    assert summary["files"] == 1
    assert (tmp_path / "benchmark_history" / "x.json").exists()


@pytest.mark.unit
def test_benchmark_history_sync_isolates_per_file_failures(monkeypatch, tmp_path, capsys):
    """M17: a single broken JSON download no longer kills the whole sync.
    The container still boots and the History tab renders the files that
    did make it through (plus the git-tracked floor bundled in the image)."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    # Two listed objects; the second throws on GET.
    angry_key = "models/benchmark_history/broken.json"
    good_key = "models/benchmark_history/good.json"
    good_bytes = b'{"ok": true}'

    class _SometimesAngryS3:
        def __init__(self):
            self._objects = {good_key: good_bytes, angry_key: b"unreachable"}

        def get_paginator(self, _name):
            class _Paginator:
                def paginate(self, **_):
                    yield {
                        "Contents": [
                            {"Key": good_key},
                            {"Key": angry_key},
                        ]
                    }

            return _Paginator()

        def get_object(self, Bucket, Key):  # noqa: N803
            if Key == angry_key:
                raise ClientError(
                    error_response={"Error": {"Code": "InternalError", "Message": "boom"}},
                    operation_name="GetObject",
                )
            return {"Body": io.BytesIO(self._objects[Key])}

    with mock.patch("boto3.client", return_value=_SometimesAngryS3()):
        summary = model_sync.sync_benchmark_history_from_s3()
    assert summary is not None
    assert summary["files"] == 1
    assert [item["key"] for item in summary["failed"]] == [angry_key]
    # Good file landed; broken file didn't.
    assert (tmp_path / "benchmark_history" / "good.json").read_bytes() == good_bytes
    assert not (tmp_path / "benchmark_history" / "broken.json").exists()
    assert "PARTIAL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sync_predictions_cache_from_s3 / upload_predictions_cache_to_s3
# ---------------------------------------------------------------------------


class _FakeS3WithPut(_FakeS3):
    """Extends _FakeS3 with put_object so the upload tests can capture writes."""

    def __init__(self, objects: dict[str, bytes]):
        super().__init__(objects)
        self.puts: dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str):  # noqa: N803
        self.puts[Key] = Body
        # Make subsequent get_object succeed on the same key — the round-trip
        # tests rely on this.
        self._objects[Key] = Body


@pytest.mark.unit
def test_predcache_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert model_sync.sync_predictions_cache_from_s3() is None
    assert "unset" in capsys.readouterr().out


@pytest.mark.unit
def test_predcache_sync_downloads_all_three_files(monkeypatch, tmp_path):
    """When all three cache files exist in S3, they land under
    data/serving_cache/ ready for _try_hydrate_from_disk."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        "models/predictions_cache/predictions.parquet": b"\x50\x41\x52\x31fakeparquet",
        "models/predictions_cache/metrics.json": b'{"ppr": {}}',
        "models/predictions_cache/fingerprint.json": b'{"sha256": "abc"}',
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_predictions_cache_from_s3()

    assert summary is not None
    assert summary["files"] == 3
    dest = tmp_path / "data" / "serving_cache"
    assert (dest / "predictions.parquet").read_bytes() == objects[
        "models/predictions_cache/predictions.parquet"
    ]
    assert (dest / "metrics.json").read_bytes() == objects["models/predictions_cache/metrics.json"]
    assert (dest / "fingerprint.json").read_bytes() == objects[
        "models/predictions_cache/fingerprint.json"
    ]


@pytest.mark.unit
def test_predcache_sync_cleans_up_partial_when_any_file_missing(monkeypatch, tmp_path):
    """If even one of the three cache files is missing from S3, the consumer
    fingerprint check would fail. Clean up the partial downloads so a stale
    parquet can't be paired with a missing fingerprint to bypass invalidation.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        "models/predictions_cache/predictions.parquet": b"PAR1content",
        "models/predictions_cache/metrics.json": b'{"ppr": {}}',
        # fingerprint.json deliberately absent — simulates a freshly-seeded
        # bucket or an interrupted prior upload.
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_predictions_cache_from_s3()

    assert summary is not None
    assert summary["files"] == 0
    assert summary["missing"] == ["fingerprint.json"]
    dest = tmp_path / "data" / "serving_cache"
    # Partial downloads cleaned up — the directory might exist but should
    # contain none of the cache files.
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        assert not (dest / name).exists(), f"{name} should have been cleaned up"


@pytest.mark.unit
def test_predcache_sync_swallows_unexpected_s3_error(monkeypatch, tmp_path, capsys):
    """Best-effort: a transient S3 failure (not NoSuchKey) must not crash
    boot — the pre-warm thread will just recompute and re-upload.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    class _AngryS3:
        def get_object(self, Bucket, Key):  # noqa: N803
            raise ClientError(
                error_response={"Error": {"Code": "InternalError", "Message": "boom"}},
                operation_name="GetObject",
            )

    with mock.patch("boto3.client", return_value=_AngryS3()):
        # Must return without raising.
        summary = model_sync.sync_predictions_cache_from_s3()
    assert summary is not None
    assert summary["files"] == 0
    # Every file failed with a non-404 error; the new ``failed`` key surfaces
    # this so an operator can distinguish a cold bucket (missing) from an
    # S3 permissions/network issue (failed).
    assert sorted(summary.get("failed", [])) == sorted(model_sync._PREDICTIONS_CACHE_FILES)
    assert "FAILED" in capsys.readouterr().out


@pytest.mark.unit
def test_predcache_sync_cleans_up_partial_on_mixed_success_and_error(monkeypatch, tmp_path):
    """If two files download fine but one hits a non-404 ClientError, the
    consumer-side fingerprint check would still fail — clean up the partial
    downloads so a stale parquet can't be paired with a fresh fingerprint
    from a later boot. This complements the all-404 cleanup test above.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fingerprint_key = "models/predictions_cache/fingerprint.json"

    class _PartiallyAngryS3:
        def __init__(self):
            self._objects = {
                "models/predictions_cache/predictions.parquet": b"PAR1content",
                "models/predictions_cache/metrics.json": b'{"ppr": {}}',
                # fingerprint.json present in S3 but throws on GET.
            }

        def get_object(self, Bucket, Key):  # noqa: N803
            if Key == fingerprint_key:
                raise ClientError(
                    error_response={"Error": {"Code": "InternalError", "Message": "boom"}},
                    operation_name="GetObject",
                )
            return {"Body": io.BytesIO(self._objects[Key])}

    with mock.patch("boto3.client", return_value=_PartiallyAngryS3()):
        summary = model_sync.sync_predictions_cache_from_s3()

    assert summary is not None
    assert summary["files"] == 0
    assert summary.get("failed") == ["fingerprint.json"]
    # Cleanup must fire even when only non-404 errors were the trigger.
    dest = tmp_path / "data" / "serving_cache"
    for name in model_sync._PREDICTIONS_CACHE_FILES:
        assert not (dest / name).exists(), f"{name} should have been cleaned up"


@pytest.mark.unit
def test_predcache_upload_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert model_sync.upload_predictions_cache_to_s3() is None


@pytest.mark.unit
def test_predcache_upload_noop_when_any_local_file_missing(monkeypatch, tmp_path, capsys):
    """If even one of the three files isn't on disk yet (e.g., _persist_cache_to_disk
    bailed mid-write), don't upload a partial set — the next prewarm will
    recompute + re-upload cleanly.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data" / "serving_cache"
    cache.mkdir(parents=True)
    # Only two of three present.
    (cache / "predictions.parquet").write_bytes(b"PAR1x")
    (cache / "metrics.json").write_bytes(b"{}")

    with mock.patch("boto3.client") as boto_mock:
        assert model_sync.upload_predictions_cache_to_s3() is None
        boto_mock.assert_not_called()
    assert "missing" in capsys.readouterr().out


@pytest.mark.unit
def test_predcache_upload_then_sync_round_trips(monkeypatch, tmp_path):
    """End-to-end on the S3 side: write three files locally, upload, then
    sync to a separate dest dir and check bytes match. Uses the upload's
    own writes to populate the FakeS3 store.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data" / "serving_cache"
    cache.mkdir(parents=True)
    payloads = {
        "predictions.parquet": b"PAR1xyz",
        "metrics.json": b'{"ppr": {"Ridge": {}}}',
        "fingerprint.json": b'{"sha256": "deadbeef"}',
    }
    for name, body in payloads.items():
        (cache / name).write_bytes(body)

    fake_s3 = _FakeS3WithPut({})
    with mock.patch("boto3.client", return_value=fake_s3):
        upload_summary = model_sync.upload_predictions_cache_to_s3()
        assert upload_summary["files"] == 3
        # Clear local cache to prove the sync re-downloads.
        for name in payloads:
            (cache / name).unlink()
        sync_summary = model_sync.sync_predictions_cache_from_s3()

    assert sync_summary is not None
    assert sync_summary["files"] == 3
    for name, body in payloads.items():
        assert (cache / name).read_bytes() == body
        assert fake_s3.puts[f"models/predictions_cache/{name}"] == body


@pytest.mark.unit
def test_predcache_upload_swallows_s3_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data" / "serving_cache"
    cache.mkdir(parents=True)
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        (cache / name).write_bytes(b"x")

    class _AngryS3:
        def put_object(self, **_):
            raise ClientError(
                error_response={"Error": {"Code": "AccessDenied", "Message": "no"}},
                operation_name="PutObject",
            )

    with mock.patch("boto3.client", return_value=_AngryS3()):
        # Must return without raising; persist call site treats result as
        # advisory only.
        assert model_sync.upload_predictions_cache_to_s3() is None
    assert "FAILED" in capsys.readouterr().out


@pytest.mark.unit
def test_predcache_respects_custom_prefix(monkeypatch, tmp_path):
    """FF_MODEL_S3_PREFIX is honored end-to-end on both sync and upload paths."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "staging/v3")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data" / "serving_cache"
    cache.mkdir(parents=True)
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        (cache / name).write_bytes(b"x")

    fake_s3 = _FakeS3WithPut({})
    with mock.patch("boto3.client", return_value=fake_s3):
        model_sync.upload_predictions_cache_to_s3()
    assert set(fake_s3.puts.keys()) == {
        "staging/v3/predictions_cache/predictions.parquet",
        "staging/v3/predictions_cache/metrics.json",
        "staging/v3/predictions_cache/fingerprint.json",
    }
    # Every uploaded key sits under the configured staging/v3 prefix — no key
    # accidentally falls back to the default ``models/`` prefix or escapes to
    # an unrelated path. This is the actual contract under test.
    assert all(key.startswith("staging/v3/predictions_cache/") for key in fake_s3.puts)
    assert not any(key.startswith("models/") for key in fake_s3.puts)


@pytest.mark.unit
def test_predcache_sync_pulls_optional_snapshot_when_present(monkeypatch, tmp_path):
    """The browser snapshot (snapshot.json) is synced best-effort alongside the
    required triple. It is auxiliary, so it does not count toward summary['files'].
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    objects = {
        "models/predictions_cache/predictions.parquet": b"PAR1x",
        "models/predictions_cache/metrics.json": b'{"ppr": {}}',
        "models/predictions_cache/fingerprint.json": b'{"sha256": "abc"}',
        "models/predictions_cache/snapshot.json": b'{"scoring": {}}',
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_predictions_cache_from_s3()

    assert summary["files"] == 3
    dest = tmp_path / "data" / "serving_cache"
    assert (dest / "snapshot.json").read_bytes() == objects[
        "models/predictions_cache/snapshot.json"
    ]


@pytest.mark.unit
def test_predcache_sync_missing_snapshot_does_not_break_triple(monkeypatch, tmp_path):
    """Regression guard: a bucket with the required triple but NO snapshot.json
    must hydrate the triple normally. The optional file's absence must not
    trigger the partial-cleanup-and-recompute path that a missing *required*
    file does — otherwise every fresh container in the transition window would
    eat a 30-60s recompute.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    objects = {
        "models/predictions_cache/predictions.parquet": b"PAR1x",
        "models/predictions_cache/metrics.json": b'{"ppr": {}}',
        "models/predictions_cache/fingerprint.json": b'{"sha256": "abc"}',
        # snapshot.json deliberately absent
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_predictions_cache_from_s3()

    assert summary["files"] == 3
    dest = tmp_path / "data" / "serving_cache"
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        assert (dest / name).is_file(), f"{name} must survive a missing optional snapshot"
    assert not (dest / "snapshot.json").exists()


@pytest.mark.unit
def test_predcache_upload_includes_optional_snapshot_when_present(monkeypatch, tmp_path):
    """snapshot.json on disk is uploaded alongside the triple; the returned
    ``files`` count still reflects only the required triple.
    """
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data" / "serving_cache"
    cache.mkdir(parents=True)
    for name in ("predictions.parquet", "metrics.json", "fingerprint.json"):
        (cache / name).write_bytes(b"x")
    (cache / "snapshot.json").write_bytes(b'{"scoring": {}}')

    fake_s3 = _FakeS3WithPut({})
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.upload_predictions_cache_to_s3()

    assert summary["files"] == 3
    assert fake_s3.puts["models/predictions_cache/snapshot.json"] == b'{"scoring": {}}'
