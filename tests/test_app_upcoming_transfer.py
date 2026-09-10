"""Failures at the real S3 request/stream and local publication boundaries."""

import io
import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, IncompleteReadError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from src.serving import upcoming_artifact as transfer
from src.serving import upcoming_status, upcoming_week

pytestmark = pytest.mark.unit
BUCKET = "test-upcoming"
KEY = "models/predictions_cache/upcoming_week.json"
GENERATED = "2026-09-09T12:00:00+00:00"


@pytest.fixture
def good():
    rows = [
        {"player_id": pos, "name": pos, "position": pos, "team": "KC", "ridge_pred": 0.0}
        for pos in ("QB", "RB", "WR", "TE", "K", "DST")
    ]
    return {
        "available": True,
        "generated_at": GENERATED,
        "season": 2026,
        "week": 1,
        "positions": [row["position"] for row in rows],
        "scoring": {fmt: deepcopy(rows) for fmt in ("ppr", "half_ppr", "standard")},
    }


def encoded(payload):
    return json.dumps(payload).encode()


def object_response(body, etag='"new"'):
    return {"Body": StreamingBody(io.BytesIO(body), len(body)), "ETag": etag}


@pytest.fixture
def s3(monkeypatch):
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    monkeypatch.setattr(transfer, "_client", lambda: client)
    monkeypatch.setattr(transfer, "_etags", {})
    monkeypatch.setattr(transfer.time, "sleep", Mock())
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()


def test_conditional_poll_retains_bytes_age_and_last_good_on_corruption(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    body = encoded(good)
    request = {"Bucket": BUCKET, "Key": KEY}
    s3.add_response("get_object", object_response(body), request)
    assert transfer.sync(str(path), BUCKET, KEY)
    stamp = path.stat().st_mtime_ns
    request["IfNoneMatch"] = '"new"'
    s3.add_client_error(
        "get_object", service_error_code="304", http_status_code=304, expected_params=request
    )
    assert not transfer.sync(str(path), BUCKET, KEY)
    assert path.stat().st_mtime_ns == stamp
    s3.add_response("get_object", object_response(b'{"available": true}', '"broken"'), request)
    assert not transfer.sync(str(path), BUCKET, KEY)
    assert path.read_bytes() == body
    assert path.stat().st_mtime_ns == stamp
    status = upcoming_status.freshness(
        json.loads(path.read_bytes()), now=datetime(2026, 9, 10, tzinfo=UTC)
    )
    assert status["status"] == "stale" and status["age_seconds"] == 12 * 3600
    # Bad bytes never become the conditional ETag; a later healthy update wins.
    good["week"] = 2
    s3.add_response("get_object", object_response(encoded(good), '"healthy"'), request)
    assert transfer.sync(str(path), BUCKET, KEY)
    assert json.loads(path.read_bytes())["week"] == 2


def test_etag_is_not_reused_after_local_file_loss_or_corruption(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    request = {"Bucket": BUCKET, "Key": KEY}
    for damage in (lambda: path.unlink(), lambda: path.write_bytes(b"broken")):
        s3.add_response("get_object", object_response(encoded(good)), request)
        assert transfer.sync(str(path), BUCKET, KEY)
        damage()
        s3.add_response("get_object", object_response(encoded(good)), request)
        assert transfer.sync(str(path), BUCKET, KEY)
        # Reset to model the next independent cold start.
        path.unlink()


def test_two_workers_reuse_etags_when_the_other_installs_identical_bytes(
    s3, monkeypatch, tmp_path, good
):
    path = tmp_path / "upcoming_week.json"
    request = {"Bucket": BUCKET, "Key": KEY}
    # Each gunicorn worker owns an ETag map but shares the atomic cache file.
    workers = [{}, {}]
    for state in workers:
        monkeypatch.setattr(transfer, "_etags", state)
        s3.add_response("get_object", object_response(encoded(good)), request)
        assert transfer.sync(str(path), BUCKET, KEY)
    stamp = path.stat().st_mtime_ns
    for state in workers:
        monkeypatch.setattr(transfer, "_etags", state)
        s3.add_client_error(
            "get_object",
            "304",
            http_status_code=304,
            expected_params={**request, "IfNoneMatch": '"new"'},
        )
        assert not transfer.sync(str(path), BUCKET, KEY)
        s3.assert_no_pending_responses()
        assert path.stat().st_mtime_ns == stamp


def test_cold_start_skips_bad_versions_and_recovers_without_relabeling(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    request = {"Bucket": BUCKET, "Key": KEY}
    s3.add_response("get_object", object_response(b"not json"), request)
    s3.add_response(
        "list_object_versions",
        {
            "Versions": [
                {"Key": KEY, "VersionId": "latest", "IsLatest": True},
                {"Key": KEY, "VersionId": "bad", "IsLatest": False},
                {"Key": KEY, "VersionId": "good", "IsLatest": False},
            ]
        },
        {"Bucket": BUCKET, "Prefix": KEY, "MaxKeys": 6},
    )
    s3.add_response("get_object", object_response(b"{}"), {**request, "VersionId": "bad"})
    s3.add_response(
        "get_object", object_response(encoded(good), '"old"'), {**request, "VersionId": "good"}
    )
    assert transfer.sync(str(path), BUCKET, KEY)
    assert path.read_bytes() == encoded(good)
    assert json.loads(path.read_bytes())["generated_at"] == GENERATED
    # Recovery never pins serving to the old version.
    good["week"] = 2
    s3.add_response(
        "get_object", object_response(encoded(good)), {**request, "IfNoneMatch": '"old"'}
    )
    assert transfer.sync(str(path), BUCKET, KEY)
    assert json.loads(path.read_bytes())["week"] == 2


def test_deleted_latest_can_recover_and_missing_version_permission_stays_warming(
    s3, tmp_path, good
):
    path = tmp_path / "upcoming_week.json"
    request = {"Bucket": BUCKET, "Key": KEY}
    s3.add_client_error("get_object", "NoSuchKey", http_status_code=404, expected_params=request)
    s3.add_client_error(
        "list_object_versions",
        "AccessDenied",
        http_status_code=403,
        expected_params={"Bucket": BUCKET, "Prefix": KEY, "MaxKeys": 6},
    )
    assert not transfer.sync(str(path), BUCKET, KEY)
    assert not path.exists()
    s3.add_client_error("get_object", "NoSuchKey", http_status_code=404, expected_params=request)
    s3.add_response(
        "list_object_versions",
        {
            "Versions": [
                {"Key": KEY, "VersionId": "old", "IsLatest": False},
            ]
        },
        {"Bucket": BUCKET, "Prefix": KEY, "MaxKeys": 6},
    )
    s3.add_response("get_object", object_response(encoded(good)), {**request, "VersionId": "old"})
    assert transfer.sync(str(path), BUCKET, KEY)


def test_access_denied_preserves_local_without_retry_or_version_scan(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    path.write_bytes(encoded(good))
    s3.add_client_error(
        "get_object",
        "AccessDenied",
        http_status_code=403,
        expected_params={"Bucket": BUCKET, "Key": KEY},
    )
    assert not transfer.sync(str(path), BUCKET, KEY)
    transfer.time.sleep.assert_not_called()
    assert path.read_bytes() == encoded(good)


def test_transient_request_and_interrupted_body_retry_within_one_budget(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    request = {"Bucket": BUCKET, "Key": KEY}
    s3.add_client_error("get_object", "SlowDown", http_status_code=503, expected_params=request)
    broken_stream = Mock()
    broken_stream.read.side_effect = IncompleteReadError(actual_bytes=2, expected_bytes=100)
    s3.add_response("get_object", {"Body": broken_stream}, request)
    s3.add_response("get_object", object_response(encoded(good)), request)
    assert transfer.sync(str(path), BUCKET, KEY)
    assert [call.args[0] for call in transfer.time.sleep.call_args_list] == [2, 4]
    broken_stream.close.assert_called_once()


def test_exhausted_transient_download_preserves_last_good(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    path.write_bytes(encoded(good))
    for _ in range(3):
        s3.add_client_error(
            "get_object",
            "ServiceUnavailable",
            http_status_code=503,
            expected_params={"Bucket": BUCKET, "Key": KEY},
        )
    assert not transfer.sync(str(path), BUCKET, KEY)
    assert path.read_bytes() == encoded(good)
    assert transfer.time.sleep.call_count == 2


@pytest.mark.parametrize("succeed", [True, False])
def test_publication_retries_same_built_bytes_and_only_commits_after_success(
    monkeypatch, tmp_path, good, succeed
):
    monkeypatch.setattr(upcoming_week.core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", BUCKET)
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setattr(upcoming_week, "_last_signature", "old")
    monkeypatch.setattr(transfer.time, "sleep", Mock())
    client = Mock()
    monkeypatch.setattr(transfer, "_client", lambda: client)
    path = tmp_path / "upcoming_week.json"
    attempts = []

    def put(**request):
        attempts.append(request["Body"])
        # Even if the local path changes, retries must upload the built snapshot.
        path.write_bytes(b"changed after upload started")
        if len(attempts) < 3 or not succeed:
            raise ClientError(
                {"Error": {"Code": "SlowDown"}, "ResponseMetadata": {"HTTPStatusCode": 503}},
                "PutObject",
            )

    client.put_object.side_effect = put
    if succeed:
        upcoming_week._publish_artifact(good, "new")
        assert upcoming_week._last_signature == "new"
    else:
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            upcoming_week._publish_artifact(good, "new")
        assert upcoming_week._last_signature == "old"
    assert len(attempts) == 3 and all(body == encoded(good) for body in attempts)
    assert transfer.time.sleep.call_count == 2


def test_failed_atomic_replace_keeps_previous_file_and_cleans_temp(s3, monkeypatch, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    path.write_bytes(encoded(good))
    new = {**good, "week": 2}
    s3.add_response("get_object", object_response(encoded(new)), {"Bucket": BUCKET, "Key": KEY})
    monkeypatch.setattr(transfer.os, "replace", Mock(side_effect=OSError("disk error")))
    assert not transfer.sync(str(path), BUCKET, KEY)
    assert path.read_bytes() == encoded(good)
    assert set(p.name for p in tmp_path.iterdir()) == {
        "upcoming_week.json",
        "upcoming_week.sync.lock",
    }


def test_overlapping_poll_is_skipped_and_lock_is_released(s3, tmp_path, good):
    path = tmp_path / "upcoming_week.json"
    with transfer._exclusive_sync(path) as acquired:
        assert acquired
        assert not transfer.sync(str(path), BUCKET, KEY)
    s3.add_response("get_object", object_response(encoded(good)), {"Bucket": BUCKET, "Key": KEY})
    assert transfer.sync(str(path), BUCKET, KEY)


@pytest.mark.parametrize(
    "bad",
    [
        b"not json",
        b"[]",
        b"{}",
        b'{"available":false,"reason":"no_roster"}',
        b'{"available":false,"reason":"offseason","generated_at":"2026-09-10"}',
    ],
)
def test_malformed_artifacts_are_rejected(bad):
    with pytest.raises(ValueError):
        transfer.decode_artifact(bad)


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: p.update(positions=["QB"]),
        lambda p: p.update(week=True),
        lambda p: p.update(sources={"injuries": []}),
        lambda p: p.update(sources={"practice": ["bad"]}),
        lambda p: p.update(data_quality={"issues": "bad"}),
        lambda p: p.update(source_status={"weather": {}}),
        lambda p: p["scoring"].update(ppr=[]),
        lambda p: p["scoring"]["ppr"][0].update(name=""),
        lambda p: p["scoring"]["ppr"][0].update(ridge_pred=None),
        lambda p: p["scoring"]["ppr"][0].update(ridge_pred=True),
        lambda p: p["scoring"]["ppr"][0].update(ridge_pred=float("nan")),
        lambda p: p["scoring"]["ppr"].pop(),
        lambda p: p["scoring"]["ppr"].append(p["scoring"]["ppr"][0]),
    ],
)
def test_invalid_projection_contract_rejected(good, damage):
    damage(good)
    with pytest.raises(ValueError):
        transfer.decode_artifact(encoded(good))


def test_optional_models_and_legacy_metadata_are_not_required(good):
    good.pop("positions")
    assert transfer.decode_artifact(encoded(good)) == good
    offseason = {"available": False, "reason": "offseason", "generated_at": GENERATED}
    assert transfer.decode_artifact(encoded(offseason)) == offseason
    with pytest.raises(ValueError, match="Non-finite"):
        transfer.decode_artifact(encoded(good).replace(b"0.0", b"1e999"))
