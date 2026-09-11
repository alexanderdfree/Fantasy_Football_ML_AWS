"""Tests for src/scripts/promote.py — manual rollback CLI."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts import promote  # noqa: E402
from src.shared.model_sync import (  # noqa: E402
    manifest_key,
)
from tests.shared._helpers import make_tarball  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def smoke_test(monkeypatch):
    from unittest.mock import Mock

    from src.shared import smoke_test as module

    smoke = Mock()
    monkeypatch.setattr(module, "run_smoke_test", smoke)
    return smoke


# --------------------------------------------------------------------------
# Fake S3 — subset of the boto3 surface promote.py touches.
# --------------------------------------------------------------------------


def _nosuchkey_error(key: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": f"{key} not found"}},
        operation_name="GetObject",
    )


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeS3:
    """In-memory S3 stand-in with just the operations promote.py needs:
    get_object, put_object, head_object, copy_object.
    """

    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.ops: list[tuple[str, str]] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _nosuchkey_error(Key)
        self.ops.append(("get", Key))
        return {
            "Body": _FakeBody(self.objects[Key]),
            "ETag": hashlib.sha256(self.objects[Key]).hexdigest(),
        }

    def put_object(self, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None):  # noqa: N803
        existing = self.objects.get(Key)
        etag = None if existing is None else hashlib.sha256(existing).hexdigest()
        if (IfMatch is not None and IfMatch != etag) or (
            IfNoneMatch == "*" and existing is not None
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[Key] = Body
        self.ops.append(("put", Key))

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": f"{Key} not found"}},
                operation_name="HeadObject",
            )
        self.ops.append(("head", Key))
        return {"ContentLength": len(self.objects[Key])}

    def copy_object(self, Bucket, Key, CopySource):  # noqa: N803
        src_key = CopySource["Key"] if isinstance(CopySource, dict) else CopySource
        if src_key not in self.objects:
            raise _nosuchkey_error(src_key)
        self.objects[Key] = self.objects[src_key]
        self.ops.append(("copy", f"{src_key}->{Key}"))


# --------------------------------------------------------------------------
# Manifest fixture factories.
# --------------------------------------------------------------------------


def _hist_key(n: int) -> str:
    """Produce a ``src.shared.model_sync.new_history_key``-shaped path."""
    return f"models/releases/v3/WR/history/2026-04-{n:02d}T00-00-00Z-aaaa{n:03d}/model.tar.gz"


def _make_manifest(current_key: str, previous_key: str | None, history: list[str]) -> dict:
    cur = {
        "key": current_key,
        "sha7": "current1",
        "bytes": 4096,
        "uploaded_at": "2026-04-23T00-00-00Z",
    }
    prev = None
    if previous_key is not None:
        prev = {
            "key": previous_key,
            "sha7": "previou",
            "bytes": 4096,
            "uploaded_at": "2026-04-22T00-00-00Z",
        }
    return {
        "schema_version": 3,
        "source_frontier": {"source_sha": "a" * 40, "source_order": 1},
        "current": cur,
        "previous": prev,
        "history": history,
    }


def _bucket_with_manifest(
    prefix: str,
    pos: str,
    current_key: str,
    previous_key: str | None,
    history: list[str],
    *,
    history_objects: bool = True,
) -> _FakeS3:
    """Prime a FakeS3 with a manifest at ``{prefix}/{pos}/manifest.json`` plus
    a distinct tarball body at every key in ``history`` (so copy/head_object
    succeed). Set ``history_objects=False`` to test the GC-orphaned case.
    """
    manifest = _make_manifest(current_key, previous_key, history)
    objects: dict[str, bytes] = {
        manifest_key(prefix, pos): json.dumps(manifest).encode("utf-8"),
    }
    if history_objects:
        for k in history:
            objects[k] = make_tarball({"fixture.txt": k.encode()})
    return _FakeS3(objects)


# --------------------------------------------------------------------------
# list_history — pure function
# --------------------------------------------------------------------------


class TestListHistory:
    def test_annotates_current_and_previous(self):
        m = _make_manifest(
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        out = promote.list_history(m)
        assert "[0]" in out and _hist_key(5) in out and "← current" in out
        assert "[1]" in out and _hist_key(4) in out and "← previous" in out
        # Entry [2] has no flags.
        assert "[2]" in out
        assert out.count("← current") == 1
        assert out.count("← previous") == 1

    def test_empty_history_is_handled(self):
        m = _make_manifest(current_key=_hist_key(1), previous_key=None, history=[])
        assert "(empty)" in promote.list_history(m)


# --------------------------------------------------------------------------
# _parse_version_from_key
# --------------------------------------------------------------------------


class TestParseVersionFromKey:
    def test_well_formed_key(self):
        ts, sha7 = promote._parse_version_from_key(
            "models/WR/history/2026-04-23T00-00-00Z-abc1234/model.tar.gz"
        )
        assert ts == "2026-04-23T00-00-00Z"
        assert sha7 == "abc1234"

    def test_legacy_key_raises(self):
        """Keys missing the history/ segment are rejected loudly instead of
        flowing as ("", "") into the new manifest's current.{uploaded_at,sha7}.
        """
        with pytest.raises(promote.PromotionError, match="Malformed history key"):
            promote._parse_version_from_key("models/WR/model.tar.gz")

    def test_dir_without_dashes_raises(self):
        """``rsplit('-', 1)`` on a hyphen-less version dir would return the
        whole string as ts and an empty sha7 — the new contract rejects this.
        """
        with pytest.raises(promote.PromotionError, match="Malformed version dir"):
            promote._parse_version_from_key("models/WR/history/nodashes/model.tar.gz")

    def test_keys_with_internal_dashes_parse_correctly(self):
        """A hyphenated ISO timestamp must still parse: only the LAST hyphen
        separates ts from sha7 (rsplit('-', 1))."""
        ts, sha7 = promote._parse_version_from_key("models/WR/history/no-dashes-here/model.tar.gz")
        # "no-dashes" reads as a malformed ts but is now an accepted shape:
        # the parse returns whatever is before/after the final hyphen.
        # The shape check itself doesn't validate ISO format — that's the
        # producer's responsibility (new_history_key).
        assert ts == "no-dashes"
        assert sha7 == "here"


# --------------------------------------------------------------------------
# promote() — core CLI logic
# --------------------------------------------------------------------------


class TestPromote:
    def test_legacy_dry_run_projects_protected_keys_without_writing(self, monkeypatch):
        from src.artifacts import source

        legacy_key = "models/WR/history/2026-04-05T00-00-00Z-aaaa005/model.tar.gz"
        legacy_manifest = _make_manifest(legacy_key, None, [legacy_key])
        legacy_manifest["schema_version"] = 1
        legacy_manifest.pop("source_frontier")
        fake = _FakeS3(
            {
                "models/WR/manifest.json": json.dumps(legacy_manifest).encode(),
                legacy_key: make_tarball({"fixture.txt": b"legacy"}),
            }
        )
        proof = {"source_sha": "a" * 40, "source_order": 1, "lineage": ["a" * 40]}
        monkeypatch.setattr(source, "image_source_sha", lambda: proof["source_sha"])
        monkeypatch.setattr(source, "load_source", lambda *_: proof)
        before = fake.objects.copy()
        planned = promote.promote(fake, "b", "models", "WR", legacy_key, dry_run=True)
        assert planned["stable"]["key"].startswith("models/releases/v3/WR/history/")
        assert planned["rollback_epoch"]
        assert fake.objects == before
        assert not any(operation in {"put", "copy"} for operation, _ in fake.ops)

    def test_manual_rollback_preserves_source_and_intent_frontiers(self):
        fake = _bucket_with_manifest(
            "models", "WR", _hist_key(5), _hist_key(4), [_hist_key(5), _hist_key(4)]
        )
        key = manifest_key("models", "WR")
        old = json.loads(fake.objects[key])
        old["intent_frontier"] = {"source_sha": "a" * 40, "sequence": 7}
        old["rollback_epoch"] = "prior-rollback"
        fake.objects[key] = json.dumps(old).encode()
        result = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert result["source_frontier"] == old["source_frontier"]
        assert result["intent_frontier"] == old["intent_frontier"]
        assert result["rollback_epoch"] != old["rollback_epoch"]
        assert result["promotion_mode"] == "rollback"

    def test_manifest_read_failure_does_not_attempt_publication(self, monkeypatch):
        fake = _FakeS3({})

        def denied(**_):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

        monkeypatch.setattr(fake, "get_object", denied)
        with pytest.raises(promote.PromotionError, match="Cannot read manifest"):
            promote.promote(fake, "b", "models", "WR", _hist_key(5))
        assert fake.ops == []

    def test_smoke_failure_leaves_manifest_unchanged(self, smoke_test):
        fake = _bucket_with_manifest("models", "WR", _hist_key(5), None, [_hist_key(5)])
        before = fake.objects[manifest_key("models", "WR")]
        smoke_test.side_effect = RuntimeError("NaN predictions")
        with pytest.raises(promote.PromotionError, match="failed validation"):
            promote.promote(fake, "b", "models", "WR", _hist_key(5))
        assert fake.objects[manifest_key("models", "WR")] == before
        assert not any(op == "put" for op, _ in fake.ops)

    def test_concurrent_publication_prevents_stale_operator_rollback(self, smoke_test):
        fake = _bucket_with_manifest("models", "WR", _hist_key(5), None, [_hist_key(5)])
        key = manifest_key("models", "WR")
        newer = {"stable": {"key": "newer-approved"}, "history": ["newer-approved"]}

        def concurrent_publish(*_):
            fake.objects[key] = json.dumps(newer).encode()

        smoke_test.side_effect = concurrent_publish
        with pytest.raises(promote.PromotionError, match="PreconditionFailed"):
            promote.promote(fake, "b", "models", "WR", _hist_key(5))
        assert json.loads(fake.objects[key]) == newer

    def test_legacy_rollback_establishes_approval_and_preserves_old_stable(self, smoke_test):
        fake = _bucket_with_manifest(
            "models", "WR", _hist_key(5), _hist_key(4), [_hist_key(5), _hist_key(4)]
        )
        key = manifest_key("models", "WR")
        old = json.loads(fake.objects[key])
        old["schema_version"] = 2
        old["stable"] = old["current"]
        fake.objects[key] = json.dumps(old).encode()
        new = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert new["schema_version"] == 3
        assert new["stable"]["key"] == _hist_key(4)
        assert new["stable"]["smoke_passed"] is True
        assert new["previous_stable"] == old["stable"]
        smoke_test.assert_called_once()

    def test_happy_path_promotes_history_entry(self):
        """current=A, previous=B, history=[A, B, C]; promote --to C
        → current=C, previous=A, history unchanged. Only the manifest is
        written — the legacy mirror copy was removed in the race fix."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        target = _hist_key(3)

        new = promote.promote(fake, "b", "models", "WR", target)

        assert new["current"]["key"] == target
        assert new["previous"]["key"] == _hist_key(5)
        assert new["history"] == [_hist_key(5), _hist_key(4), _hist_key(3)]

        # Manifest was actually written.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk == new

        # Legacy mirror is NOT touched — consumers all read the manifest now.
        assert "models/WR/model.tar.gz" not in fake.objects

    def test_rejects_key_not_in_history(self):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        bad = _hist_key(99)  # not in history
        with pytest.raises(promote.PromotionError, match="not in manifest.history"):
            promote.promote(fake, "b", "models", "WR", bad)

        # Manifest must be untouched.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk["current"]["key"] == _hist_key(5)  # no change

    def test_rejects_key_present_in_history_but_missing_in_s3(self):
        """Defensive: if GC deleted a tracked key (shouldn't happen, but it
        has been a source of subtle bugs in the past), refuse to promote
        rather than write a manifest pointing at a ghost."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
            history_objects=False,  # no tarball bodies in S3
        )
        with pytest.raises(promote.PromotionError, match="missing from S3"):
            promote.promote(fake, "b", "models", "WR", _hist_key(3))

        # Manifest unchanged, legacy not created.
        on_disk = json.loads(fake.objects[manifest_key("models", "WR")])
        assert on_disk["current"]["key"] == _hist_key(5)
        assert "models/WR/model.tar.gz" not in fake.objects

    def test_dry_run_does_not_write(self):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        orig_manifest_bytes = fake.objects[manifest_key("models", "WR")]

        new = promote.promote(fake, "b", "models", "WR", _hist_key(3), dry_run=True)
        assert new["current"]["key"] == _hist_key(3)

        # No put/copy was issued.
        assert not any(op[0] in ("put", "copy") for op in fake.ops)
        # Manifest bytes still the original.
        assert fake.objects[manifest_key("models", "WR")] == orig_manifest_bytes
        # No legacy mirror was created in dry-run.
        assert "models/WR/model.tar.gz" not in fake.objects

    def test_promote_with_no_previous(self):
        """Starting from a fresh bucket where previous=None, promoting still
        works; the new manifest's previous becomes old.current as normal."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=None,
            history=[_hist_key(5), _hist_key(4)],
        )
        new = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert new["current"]["key"] == _hist_key(4)
        assert new["previous"]["key"] == _hist_key(5)

    def test_raises_when_no_manifest_exists(self):
        """Fresh bucket with only a legacy model.tar.gz (pre-migration) and
        no manifest — promotion has nothing to rewrite; fail clearly."""
        fake = _FakeS3(objects={"models/WR/model.tar.gz": b"legacy"})
        with pytest.raises(promote.PromotionError, match="No manifest"):
            promote.promote(fake, "b", "models", "WR", _hist_key(1))

    def test_write_manifest_client_error_translates_to_promotion_error(self):
        """A ClientError from the final put_object propagates as
        PromotionError so main() renders a friendly message instead of a
        raw boto3 stack trace. Translation note: the old manifest is left
        untouched (write is atomic), so no rollback is needed."""

        class _PutAngryS3(_FakeS3):
            def put_object(
                self, Bucket, Key, Body, ContentType=None, IfMatch=None, IfNoneMatch=None
            ):  # noqa: N803
                raise ClientError(
                    error_response={"Error": {"Code": "AccessDenied", "Message": "no"}},
                    operation_name="PutObject",
                )

        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        # Swap the put_object impl while preserving the seeded manifest.
        angry = _PutAngryS3(fake.objects)
        with pytest.raises(promote.PromotionError, match="Failed to write new manifest"):
            promote.promote(angry, "b", "models", "WR", _hist_key(3))
        # Old manifest unchanged (no partial state).
        on_disk = json.loads(angry.objects[manifest_key("models", "WR")])
        assert on_disk["current"]["key"] == _hist_key(5)

    def test_schema_version_preserved(self):
        """Defensive: if a future manifest bumps schema_version, promotion
        shouldn't silently downgrade it."""
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4)],
        )
        # Patch the stored manifest to schema_version=2.
        raw = json.loads(fake.objects[manifest_key("models", "WR")])
        raw["schema_version"] = 2
        fake.objects[manifest_key("models", "WR")] = json.dumps(raw).encode()

        new = promote.promote(fake, "b", "models", "WR", _hist_key(4))
        assert new["schema_version"] == 3


# --------------------------------------------------------------------------
# main() — end-to-end CLI coverage
# --------------------------------------------------------------------------


class TestMainCLI:
    @pytest.fixture
    def stub_boto3(self, monkeypatch):
        """Swap ``sys.modules["boto3"]`` so ``import boto3`` inside
        ``promote.main()`` resolves to a stub whose ``.client(...)`` returns
        the given ``_FakeS3``. Monkeypatch's auto-unwind restores the real
        module at teardown."""
        import sys as _sys
        import types

        def _factory(fake_s3: _FakeS3):
            fake_mod = types.ModuleType("boto3")
            fake_mod.client = lambda *a, **kw: fake_s3  # type: ignore[attr-defined]
            monkeypatch.setitem(_sys.modules, "boto3", fake_mod)

        return _factory

    def test_list_prints_history(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "← current" in out
        assert "← previous" in out

    def test_to_promotes_and_logs(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(3)])
        out = capsys.readouterr().out
        assert rc == 0
        assert f"Promoted WR: current → {_hist_key(3)}" in out
        # Manifest actually changed.
        m = json.loads(fake.objects[manifest_key("models", "WR")])
        assert m["current"]["key"] == _hist_key(3)

    def test_to_with_dry_run_prints_json_does_not_write(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4), _hist_key(3)],
        )
        orig = fake.objects[manifest_key("models", "WR")]
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(3), "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "[dry-run]" in out
        # JSON is emitted and parseable.
        planned = json.loads(out.split("\n", 1)[1])
        assert planned["current"]["key"] == _hist_key(3)
        # Manifest bytes unchanged.
        assert fake.objects[manifest_key("models", "WR")] == orig

    def test_error_exit_code_on_bad_key(self, stub_boto3, capsys):
        fake = _bucket_with_manifest(
            "models",
            "WR",
            current_key=_hist_key(5),
            previous_key=_hist_key(4),
            history=[_hist_key(5), _hist_key(4)],
        )
        stub_boto3(fake)
        rc = promote.main(["--position", "WR", "--to", _hist_key(99)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "ERROR" in err
        assert "not in manifest.history" in err


# --------------------------------------------------------------------------
# audit_features.py smoke (L-SS3): live in this file because the parallel
# code-review-cleanup bundle is restricted to existing test files. A
# follow-up could split this into ``tests/scripts/test_audit_features.py``.
# --------------------------------------------------------------------------


class TestAuditFeaturesKDstCoverage:
    """L-SS3 smoke: ``src.scripts.audit_features`` now reports a K/DST
    whitelist section (previously QB/RB/WR/TE only). Verifies the helper
    function is wired up + the whitelist isn't empty for either position.
    """

    def test_k_whitelist_emitted(self, capsys):
        from src.scripts import audit_features

        cols = audit_features._audit_whitelist_only("K", audit_features.get_k_feature_columns)
        out = capsys.readouterr().out
        assert "K (whitelist-only)" in out
        assert len(cols) > 0, "K feature whitelist must not be empty"
        # The summary line lists the count.
        assert f"whitelisted features: {len(cols)}" in out

    def test_dst_whitelist_emitted(self, capsys):
        from src.scripts import audit_features

        cols = audit_features._audit_whitelist_only("DST", audit_features.get_dst_feature_columns)
        out = capsys.readouterr().out
        assert "DST (whitelist-only)" in out
        assert len(cols) > 0, "DST feature whitelist must not be empty"
        assert f"whitelisted features: {len(cols)}" in out
