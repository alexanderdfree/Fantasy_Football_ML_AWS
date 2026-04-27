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


def _make_tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


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


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


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


def test_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    result = model_sync.sync_models_from_s3()
    assert result is None
    assert "unset" in capsys.readouterr().out


def test_sync_noop_when_bucket_blank(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "   ")
    assert model_sync.sync_models_from_s3() is None


def test_sync_extracts_all_positions_via_legacy_fallback(monkeypatch, tmp_path):
    """Pre-migration bucket: no manifest.json, just legacy model.tar.gz per
    position. _sync_one must fall through to the legacy key and report
    source=legacy."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        f"models/{pos}/model.tar.gz": _make_tarball(
            {f"{pos.lower()}_marker.pkl": b"payload-" + pos.encode()}
        )
        for pos in model_sync.POSITIONS
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        summary = model_sync.sync_models_from_s3()

    assert summary is not None
    assert {r["pos"] for r in summary["positions"]} == set(model_sync.POSITIONS)
    assert {r["source"] for r in summary["positions"]} == {"legacy"}
    for pos in model_sync.POSITIONS:
        extracted = tmp_path / pos.lower() / "outputs" / "models" / f"{pos.lower()}_marker.pkl"
        assert extracted.is_file()
        assert extracted.read_bytes() == b"payload-" + pos.encode()


def test_sync_honors_custom_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "nightly/v2")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    objects = {
        f"nightly/v2/{pos}/model.tar.gz": _make_tarball({"file.pkl": b"x"})
        for pos in model_sync.POSITIONS
    }
    fake_s3 = _FakeS3(objects)
    with mock.patch("boto3.client", return_value=fake_s3):
        model_sync.sync_models_from_s3()

    # Every POS must have probed its manifest before falling back to legacy,
    # and the legacy fetch must also use the custom prefix.
    keys_called = {key for _, key in fake_s3.calls}
    for pos in model_sync.POSITIONS:
        assert f"nightly/v2/{pos}/manifest.json" in keys_called
        assert f"nightly/v2/{pos}/model.tar.gz" in keys_called


def test_sync_raises_on_missing_key(monkeypatch, tmp_path):
    """No manifest AND no legacy key → ClientError escapes the legacy path."""
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fake_s3 = _FakeS3(objects={})
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(ClientError, match="NoSuchKey"):
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
    assert (tmp_path / "wr" / "outputs" / "models" / "nn_scaler.pkl").read_bytes() == b"CURRENT"


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
        extracted = tmp_path / pos.lower() / "outputs" / "models" / "marker.pkl"
        assert extracted.read_bytes() == b"FROM_PREVIOUS"


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


# --- Manifest v2: stable-first fallback chain ---


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
        extracted = tmp_path / pos.lower() / "outputs" / "models" / "marker.pkl"
        assert extracted.read_bytes() == b"FROM_STABLE"
    # Current key bytes must NOT have been pulled — the stable key wins
    # outright and we don't probe further when stable succeeds.
    keys_called = {key for _, key in fake_s3.calls}
    for pos in model_sync.POSITIONS:
        cur_key = f"models/{pos}/history/2026-04-25T00-00-00Z-newnew1/model.tar.gz"
        assert cur_key not in keys_called


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


def test_extract_rejects_path_traversal(tmp_path):
    malicious = _make_tarball({"../../../etc/evil.pkl": b"pwn"})
    with pytest.raises(RuntimeError, match="escape"):
        model_sync._extract_tarball(malicious, tmp_path / "dest")


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


def test_data_sync_noop_when_bucket_unset(monkeypatch, capsys):
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert model_sync.sync_data_from_s3() is None
    assert "unset" in capsys.readouterr().out


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


def test_data_sync_raises_on_missing_split(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)

    fake_s3 = _FakeS3(objects={"data/raw/weekly_2012_2025.parquet": b"x"})
    with mock.patch("boto3.client", return_value=fake_s3):
        with pytest.raises(ClientError, match="NoSuchKey"):
            model_sync.sync_data_from_s3()
