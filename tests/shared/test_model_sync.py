"""Tests for src.shared.model_sync — S3 tarball sync at container boot."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
import time
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
    stable_key: str | None = "__current__",
    sha7: str = "abc1234",
    bytes_: int = 4096,
    schema_version: int = 3,
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
    if stable_key == "__current__":
        stable_key = current_key
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
    if schema_version >= 3:
        body["previous_stable"] = previous
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
        return {
            "Body": _FakeBody(self._objects[Key]),
            "ETag": hashlib.sha256(self._objects[Key]).hexdigest(),
        }

    def get_paginator(self, op: str):
        assert op == "list_objects_v2"
        return _FakePaginator(self._objects)


@pytest.mark.unit
def test_protected_writer_keys_are_disjoint_from_legacy_storage():
    assert model_sync.manifest_key("models", "QB") == "models/releases/v3/QB/manifest.json"
    assert model_sync.history_prefix("models", "QB") == "models/releases/v3/QB/history/"
    assert model_sync.legacy_manifest_key("models", "QB") == "models/QB/manifest.json"
    assert (
        model_sync.manifest_key("sandbox/alex", "K") == "sandbox/alex/releases/v3/K/manifest.json"
    )


@pytest.mark.unit
def test_boot_reader_prefers_protected_manifest_over_valid_legacy(tmp_path):
    protected = "models/releases/v3/QB/history/current/model.tar.gz"
    legacy = "models/QB/history/old/model.tar.gz"
    legacy_manifest = model_sync.legacy_manifest_key("models", "QB")
    fake = _FakeS3(
        {
            model_sync.manifest_key("models", "QB"): _manifest_bytes(protected),
            legacy_manifest: _manifest_bytes(legacy),
            protected: _make_tarball({"marker": b"protected"}),
            legacy: _make_tarball({"marker": b"legacy"}),
        }
    )
    result = model_sync._sync_one(fake, "bucket", "models", "QB", tmp_path)
    assert result["key"] == protected
    assert (tmp_path / "src/qb/outputs/models/marker").read_bytes() == b"protected"
    assert ("bucket", legacy_manifest) not in fake.calls
    assert ("bucket", legacy) not in fake.calls


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["invalid_json", "access_denied", "missing_artifact"])
def test_protected_failure_does_not_downgrade_to_working_legacy(tmp_path, failure):
    protected_manifest = model_sync.manifest_key("models", "QB")
    legacy_manifest = model_sync.legacy_manifest_key("models", "QB")
    legacy = "models/QB/history/old/model.tar.gz"

    class ProtectedFailure(_FakeS3):
        def get_object(self, Bucket, Key):
            if Key == protected_manifest and failure == "access_denied":
                self.calls.append((Bucket, Key))
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
            return super().get_object(Bucket=Bucket, Key=Key)

    fake = ProtectedFailure(
        {
            protected_manifest: b"{"
            if failure == "invalid_json"
            else _manifest_bytes("missing-protected-artifact"),
            legacy_manifest: _manifest_bytes(legacy),
            legacy: _make_tarball({"marker": b"legacy"}),
        }
    )
    with pytest.raises((ValueError, ClientError, RuntimeError)):
        model_sync._sync_one(fake, "bucket", "models", "QB", tmp_path)
    assert ("bucket", legacy_manifest) not in fake.calls
    assert ("bucket", legacy) not in fake.calls


@pytest.mark.unit
def test_legacy_fallback_is_read_only_and_never_supplies_a_writer_etag():
    legacy_key = model_sync.legacy_manifest_key("models", "QB")
    legacy_body = _manifest_bytes("models/QB/history/old/model.tar.gz")
    fake = _FakeS3({legacy_key: legacy_body})
    assert model_sync.load_manifest(fake, "bucket", "models", "QB") == json.loads(legacy_body)
    manifest, etag = model_sync.load_manifest_snapshot(fake, "bucket", "models", "QB")
    assert manifest is None and etag is None
    writer = mock.Mock()
    candidate = {"schema_version": 3, "stable": None}
    model_sync.write_manifest(writer, "bucket", "models", "QB", candidate, expected_etag=etag)
    request = writer.put_object.call_args.kwargs
    assert request["Key"] == "models/releases/v3/QB/manifest.json"
    assert request["IfNoneMatch"] == "*" and "IfMatch" not in request
    assert fake._objects[legacy_key] == legacy_body


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
    from src.shared.registry import get_inference_spec

    captured: dict[str, Path] = {}

    def resolve(_s3, _bucket, _prefix, pos, dest):
        captured[pos] = dest
        return {"pos": pos}

    monkeypatch.setattr(model_sync, "_resolve_manifest_extract", resolve)
    for pos in model_sync.POSITIONS:
        model_sync._sync_one(None, "test-bucket", "models", pos, tmp_path)
        assert str(captured[pos]) == str(tmp_path / get_inference_spec(pos)["model_dir"])


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
        key = model_sync.history_prefix("models", pos) + "2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
        objects[key] = current_tarball
        objects[model_sync.manifest_key("models", pos)] = _manifest_bytes(current_key=key)
    return objects


@pytest.mark.unit
def test_sync_one_reads_approved_stable_from_manifest(monkeypatch, tmp_path):
    """A new manifest points stable at its approved upload."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    tar = _make_tarball({"nn_scaler.pkl": b"CURRENT"})
    objects = _build_objects_for_all_positions(tar)
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    wr = next(r for r in summary["positions"] if r["pos"] == "WR")
    assert wr["source"] == "stable"
    assert wr["key"] == "models/releases/v3/WR/history/2026-04-23T00-00-00Z-aaa1234/model.tar.gz"
    assert (
        tmp_path / "src" / "wr" / "outputs" / "models" / "nn_scaler.pkl"
    ).read_bytes() == b"CURRENT"


@pytest.mark.unit
def test_sync_one_falls_back_to_previous_stable_when_stable_corrupt(monkeypatch, tmp_path, capsys):
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
    assert qb["source"] == "previous_stable"
    out = capsys.readouterr().out
    # On-call greps CloudWatch for these tags. Keep the grep-surface stable.
    assert "source=previous_stable" in out
    assert "QB stable" in out and "FAILED" in out
    # Other positions still serve current — one broken artifact doesn't poison
    # the fan-out.
    assert all(r["source"] == "stable" for r in summary["positions"] if r["pos"] != "QB")


@pytest.mark.unit
def test_sync_one_falls_back_to_previous_stable_when_stable_missing(monkeypatch, tmp_path):
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

    assert all(r["source"] == "previous_stable" for r in summary["positions"])
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

    assert all(r["source"] == "previous_stable" for r in summary["positions"])


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
def test_sync_one_rejects_candidate_only_legacy_manifest(tmp_path):
    key = "models/QB/history/unapproved/model.tar.gz"
    fake = _FakeS3(
        {
            "models/QB/manifest.json": _manifest_bytes(key, schema_version=1),
            key: _make_tarball({"marker": b"not-approved"}),
        }
    )
    with pytest.raises(RuntimeError, match="all manifest entries failed"):
        model_sync._sync_one(fake, "bucket", "models", "QB", tmp_path)
    assert ("bucket", key) not in fake.calls


@pytest.mark.unit
@pytest.mark.parametrize("schema_version", [2, 3])
@pytest.mark.parametrize("stable_exists", [False, True])
def test_sync_one_never_falls_back_to_unapproved_candidate(tmp_path, schema_version, stable_exists):
    stable = "models/QB/history/approved/model.tar.gz"
    candidate = "models/QB/history/failed-smoke/model.tar.gz"
    manifest = json.loads(
        _manifest_bytes(candidate, stable_key=stable, schema_version=schema_version)
    )
    manifest["current"]["smoke_passed"] = False
    fake = _FakeS3(
        {
            "models/QB/manifest.json": json.dumps(manifest).encode(),
            candidate: _make_tarball({"marker": b"not-approved"}),
        }
    )
    if stable_exists:
        fake._objects[stable] = b"corrupt"
    with pytest.raises(RuntimeError, match="all manifest entries failed"):
        model_sync._sync_one(fake, "bucket", "models", "QB", tmp_path)
    assert ("bucket", candidate) not in fake.calls


@pytest.mark.unit
def test_sync_one_skips_candidates_and_uses_previous_stable(monkeypatch, tmp_path):
    """Stable and current are corrupt; only approved previous_stable is eligible."""
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

    assert all(r["source"] == "previous_stable" for r in summary["positions"])


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
    assert m["schema_version"] == 3
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
    to pin to, so stable starts as None and serving fails closed."""
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
    assert m["previous_stable"] == old_stable


@pytest.mark.unit
def test_failed_candidates_cannot_displace_approved_fallback():
    first = model_sync.build_manifest("first", "a", 1, "t0", smoke_passed=True)
    latest = model_sync.build_manifest("latest", "b", 1, "t1", first, smoke_passed=True)
    for index in range(model_sync.HISTORY_KEEP_N + 2):
        latest = model_sync.build_manifest(f"failed-{index}", "c", 1, "t2", latest)
    assert latest["stable"]["key"] == "latest"
    assert latest["previous_stable"]["key"] == "first"
    assert "first" not in latest["history"]


@pytest.mark.unit
def test_corrupt_approved_extract_cannot_contaminate_fallback(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        regular = tarfile.TarInfo("failed-only")
        regular.size = 1
        archive.addfile(regular, io.BytesIO(b"x"))
        bad_link = tarfile.TarInfo("bad-link")
        bad_link.type = tarfile.SYMTYPE
        bad_link.linkname = "/outside"
        archive.addfile(bad_link)
    fake = _FakeS3(
        {
            "models/QB/manifest.json": _manifest_bytes("broken", previous_key="approved"),
            "broken": data.getvalue(),
            "approved": _make_tarball({"good-only": b"good"}),
        }
    )
    result = model_sync._sync_one(fake, "bucket", "models", "QB", tmp_path)
    assert result["source"] == "previous_stable"
    dest = tmp_path / "src/qb/outputs/models"
    assert (dest / "good-only").read_bytes() == b"good"
    assert not (dest / "failed-only").exists()


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
    assert summary == {"total_secs": 0.0, "total_bytes": 0, "files": 0, "skipped": 0, "failed": []}


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


@pytest.mark.unit
def test_benchmark_history_sync_skips_files_already_on_disk(monkeypatch, tmp_path):
    """New-file guard: benchmark_history JSONs are immutable, so a filename
    already on disk (the Docker-COPY'd floor or a prior poll) is never
    re-fetched — only genuinely-new run_ids are downloaded. This is what keeps
    the poller's steady state to a single ListBucket and zero GETs."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    dest_dir = tmp_path / "benchmark_history"
    dest_dir.mkdir()
    present_key = "models/benchmark_history/2026-05-01T10-00-00_abc.json"
    present = dest_dir / "2026-05-01T10-00-00_abc.json"
    present.write_bytes(b'{"floor": true}')  # already on disk (e.g. image floor)

    new_key = "models/benchmark_history/2026-05-19T22-47-20_new.json"
    objects = {
        # Same run_id as the on-disk file — must be skipped, not re-fetched.
        present_key: b'{"floor": "S3-VERSION-SHOULD-NOT-LAND"}',
        new_key: b'{"new": 1}',
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_benchmark_history_from_s3()

    assert summary["files"] == 1
    assert summary["skipped"] == 1
    # Only the NEW key was GET'd; the already-present one was not.
    fetched = [key for (_bucket, key) in fake_s3.calls]
    assert fetched == [new_key]
    # The on-disk file is left byte-untouched (immutable; not overwritten).
    assert present.read_bytes() == b'{"floor": true}'
    assert (dest_dir / "2026-05-19T22-47-20_new.json").read_bytes() == b'{"new": 1}'
    # Atomic download leaves no temp turds behind.
    assert not list(dest_dir.glob("*.tmp"))


@pytest.mark.unit
def test_download_file_atomic_cleans_up_tmp_on_replace_failure(monkeypatch, tmp_path):
    """The atomic path must unlink its sibling .tmp if os.replace fails, then
    re-raise — the poller runs thousands of times over a container's life, so a
    persistent rename failure would otherwise accumulate temp turds."""
    key = "models/benchmark_history/x.json"
    fake_s3 = _FakeS3({key: b'{"x": 1}'})
    dest = tmp_path / "x.json"

    def boom(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        model_sync._download_file(fake_s3, "test-bucket", key, dest, atomic=True)

    # Failure propagated (so the caller records it in `failed`), but no .tmp turd
    # is left behind and the dest was never half-written.
    assert not list(tmp_path.glob("*.tmp"))
    assert not dest.exists()


@pytest.mark.unit
def test_start_benchmark_history_poller_calls_sync_each_cycle(monkeypatch):
    """The poller thread must re-call sync_benchmark_history_from_s3 each cycle
    so a run uploaded after boot surfaces without a restart."""
    calls: list[int] = []
    barrier = threading.Event()

    def fake_sync():
        calls.append(1)
        if len(calls) >= 2:
            barrier.set()
        return None

    monkeypatch.setattr(model_sync, "sync_benchmark_history_from_s3", fake_sync)

    stop = threading.Event()
    thread = model_sync.start_benchmark_history_poller(interval_s=0, stop_event=stop)
    try:
        assert thread.daemon is True
        assert barrier.wait(timeout=2.0), f"poller did not cycle twice; calls={len(calls)}"
    finally:
        # Stop + join BEFORE monkeypatch restores the real sync, so the daemon
        # never calls real boto3 and leaks into other tests' mocks.
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "benchmark history poller thread did not stop"


@pytest.mark.unit
def test_start_benchmark_history_poller_survives_sync_exception(monkeypatch):
    """If the sync raises (unexpected bug), the daemon must not die — it logs
    and continues on the next cycle."""
    calls: list[int] = []
    raised_once = [False]

    def fake_sync():
        calls.append(1)
        if not raised_once[0]:
            raised_once[0] = True
            raise RuntimeError("simulated boom")
        return None

    monkeypatch.setattr(model_sync, "sync_benchmark_history_from_s3", fake_sync)

    stop = threading.Event()
    thread = model_sync.start_benchmark_history_poller(interval_s=0, stop_event=stop)
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(calls) >= 3:
                break
            time.sleep(0.05)
        assert len(calls) >= 3, f"poller died after exception; calls={len(calls)}"
    finally:
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "benchmark history poller thread did not stop"


# ---------------------------------------------------------------------------
# sync_predictions_cache_from_s3 / upload_predictions_cache_to_s3
# ---------------------------------------------------------------------------


class _FakeS3WithPut(_FakeS3):
    """Extends _FakeS3 with put_object so the upload tests can capture writes."""

    def __init__(self, objects: dict[str, bytes]):
        super().__init__(objects)
        self.puts: dict[str, bytes] = {}

    def put_object(
        self, Bucket: str, Key: str, Body: bytes, ContentType=None, IfMatch=None, IfNoneMatch=None
    ):  # noqa: N803
        current = self._objects.get(Key)
        etag = hashlib.sha256(current).hexdigest() if current is not None else None
        if (IfMatch is not None and IfMatch != etag) or (
            IfNoneMatch == "*" and current is not None
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts[Key] = Body
        # Make subsequent get_object succeed on the same key — the round-trip
        # tests rely on this.
        self._objects[Key] = Body


def _generation_cache_fixture(prefix="models", label="generation"):
    from src.artifacts.serving_snapshot import CACHE_SCHEMA_VERSION

    payloads = {
        "predictions.parquet": f"{label} predictions".encode(),
        "metrics.json": json.dumps({"source": label}).encode(),
        "fingerprint.json": json.dumps({"schema_version": CACHE_SCHEMA_VERSION}).encode(),
        "snapshot.json": json.dumps({"source": label}).encode(),
    }
    manifest = {
        "schema_version": 1,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "files": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
        "models": {position: f"models/{position}/approved" for position in model_sync.POSITIONS},
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    generation = hashlib.sha256(body).hexdigest()
    base = f"{prefix}/predictions_cache/generations/{generation}"
    objects = {f"{base}/{name}": value for name, value in payloads.items()}
    objects[f"{base}/manifest.json"] = body
    objects[f"{prefix}/predictions_cache/current.json"] = json.dumps(
        {
            "schema_version": 1,
            "generation": generation,
            "manifest": f"{base}/manifest.json",
        }
    ).encode()
    # Legacy objects deliberately exist so a fallback would be observable.
    objects.update({f"{prefix}/predictions_cache/{name}": b"legacy" for name in payloads})
    return generation, objects, payloads


@pytest.mark.unit
def test_predcache_sync_prefers_complete_generation_over_legacy(monkeypatch, tmp_path):
    from src.artifacts.serving_snapshot import active_directory

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "custom/models")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    generation, objects, payloads = _generation_cache_fixture("custom/models")
    s3 = _FakeS3(objects)
    legacy_download = mock.Mock()
    monkeypatch.setattr(model_sync, "_download_file", legacy_download)
    with mock.patch("boto3.client", return_value=s3):
        summary = model_sync.sync_predictions_cache_from_s3()
    assert summary == {"generation": generation, "files": 4}
    root = tmp_path / "data/serving_cache"
    current = active_directory(root)
    assert current == root / "generations" / generation
    assert all((current / name).read_bytes() == payload for name, payload in payloads.items())
    legacy_download.assert_not_called()
    assert ("test-bucket", "custom/models/predictions_cache/current.json") in s3.calls


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure", ["denied", "corrupt_manifest", "corrupt_payload", "null_pointer"]
)
def test_predcache_generation_failure_retains_old_pointer_without_legacy_fallback(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    generation, objects, _ = _generation_cache_fixture()
    pointer = "models/predictions_cache/current.json"
    base = f"models/predictions_cache/generations/{generation}"
    if failure == "corrupt_manifest":
        objects[f"{base}/manifest.json"] = b'{"schema_version":1}'
    elif failure == "corrupt_payload":
        objects[f"{base}/predictions.parquet"] = b"corrupt"
    elif failure == "null_pointer":
        objects[pointer] = b"null"

    class FailingS3(_FakeS3):
        def get_object(self, Bucket, Key):  # noqa: N803
            if failure == "denied" and Key == pointer:
                self.calls.append((Bucket, Key))
                raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
            return super().get_object(Bucket, Key)

    s3 = FailingS3(objects)
    root = tmp_path / "data/serving_cache"
    old_generation, old_objects, old_payloads = _generation_cache_fixture(label="old")
    old_directory = root / "generations" / old_generation
    old_directory.mkdir(parents=True)
    for name, payload in old_payloads.items():
        (old_directory / name).write_bytes(payload)
    old_pointer = old_objects[pointer]
    (old_directory / "manifest.json").write_bytes(
        old_objects[f"models/predictions_cache/generations/{old_generation}/manifest.json"]
    )
    (root / "current.json").write_bytes(old_pointer)
    legacy_download = mock.Mock()
    monkeypatch.setattr(model_sync, "_download_file", legacy_download)
    with mock.patch("boto3.client", return_value=s3):
        summary = model_sync.sync_predictions_cache_from_s3()
    assert "generation_error" in summary
    assert (root / "current.json").read_bytes() == old_pointer
    from src.artifacts.serving_snapshot import active_directory

    assert active_directory(root) == old_directory
    assert all(
        (old_directory / name).read_bytes() == payload for name, payload in old_payloads.items()
    )
    legacy_download.assert_not_called()
    assert not any(key == "models/predictions_cache/predictions.parquet" for _, key in s3.calls)


@pytest.mark.unit
def test_predcache_sync_noop_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    with mock.patch("boto3.client") as client:
        assert model_sync.sync_predictions_cache_from_s3() is None
        client.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy_members",
    [
        ["predictions.parquet", "metrics.json"],
        ["predictions.parquet", "metrics.json", "fingerprint.json"],
        ["predictions.parquet", "metrics.json", "fingerprint.json", "snapshot.json"],
        ["cache.tar.gz"],
    ],
)
def test_predcache_sync_requires_new_generation_before_legacy_cutover(
    monkeypatch, tmp_path, legacy_members
):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    fake = _FakeS3({f"models/predictions_cache/{name}": b"legacy" for name in legacy_members})
    with mock.patch("boto3.client", return_value=fake):
        assert model_sync.sync_predictions_cache_from_s3() == {
            "files": 0,
            "missing_generation": True,
        }
    assert fake.calls == [("test-bucket", "models/predictions_cache/current.json")]
    assert not (tmp_path / "data/serving_cache/current.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "failed_member", ["predictions.parquet", "metrics.json", "fingerprint.json", "snapshot.json"]
)
@pytest.mark.parametrize("error", ["NoSuchKey", "InternalError"])
def test_predcache_sync_rejects_incomplete_generation_without_replacing_current(
    monkeypatch, tmp_path, failed_member, error
):
    from src.artifacts.serving_snapshot import read_generation

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    old, old_objects, old_payload = _generation_cache_fixture(label="old")
    with mock.patch("boto3.client", return_value=_FakeS3(old_objects)):
        assert model_sync.sync_predictions_cache_from_s3()["generation"] == old
    candidate, objects, _ = _generation_cache_fixture(label="candidate")
    broken_key = f"models/predictions_cache/generations/{candidate}/{failed_member}"

    class PartialS3(_FakeS3):
        def get_object(self, Bucket, Key):
            if Key == broken_key:
                raise ClientError({"Error": {"Code": error}}, "GetObject")
            return super().get_object(Bucket, Key)

    with mock.patch("boto3.client", return_value=PartialS3(objects)):
        assert "generation_error" in model_sync.sync_predictions_cache_from_s3()
    current, captured = read_generation(tmp_path / "data/serving_cache")
    assert current.name == old
    assert captured == old_payload


@pytest.mark.unit
def test_predcache_sync_swallows_unexpected_s3_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    fake = mock.Mock()
    fake.get_object.side_effect = ClientError({"Error": {"Code": "InternalError"}}, "GetObject")
    with mock.patch("boto3.client", return_value=fake):
        assert "generation_error" in model_sync.sync_predictions_cache_from_s3()
    assert "retaining local snapshot" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["models", "staging/v3"])
def test_offline_publisher_then_worker_sync_round_trips(monkeypatch, tmp_path, prefix):
    from src.artifacts import serving_snapshot

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", prefix)
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path / "worker")
    fake = _FakeS3WithPut(
        {
            model_sync.manifest_key(prefix, pos): _manifest_bytes(
                f"{prefix}/releases/v3/{pos}/history/model.tar.gz"
            )
            for pos in model_sync.POSITIONS
        }
    )
    _, _, payload = _generation_cache_fixture(prefix)
    producer = tmp_path / "producer"
    serving_snapshot.publish_local(producer, payload)
    token = serving_snapshot.begin_build(fake, "test-bucket", prefix)
    published = serving_snapshot.publish(fake, "test-bucket", producer, token, prefix)
    with mock.patch("boto3.client", return_value=fake):
        assert model_sync.sync_predictions_cache_from_s3() == {
            "generation": published["generation"],
            "files": 4,
        }
    assert serving_snapshot.read_generation(tmp_path / "worker/data/serving_cache")[1] == payload
    assert all(key.startswith(f"{prefix}/predictions_cache/") for key in fake.puts)
    assert not any(key.endswith("cache.tar.gz") for key in fake.puts)


@pytest.mark.unit
@pytest.mark.parametrize("bucket", [None, "test-bucket"])
@pytest.mark.parametrize(
    "members",
    [
        [],
        ["predictions.parquet"],
        ["predictions.parquet", "metrics.json", "fingerprint.json", "snapshot.json"],
    ],
)
def test_runtime_cache_upload_never_bypasses_offline_release_publisher(
    monkeypatch, tmp_path, bucket, members
):
    if bucket is None:
        monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    else:
        monkeypatch.setenv("FF_MODEL_S3_BUCKET", bucket)
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    cache = tmp_path / "data/serving_cache"
    cache.mkdir(parents=True)
    for name in members:
        (cache / name).write_bytes(b"legacy runtime output")
    with mock.patch("boto3.client") as client:
        assert model_sync.upload_predictions_cache_to_s3() is None
        client.assert_not_called()
