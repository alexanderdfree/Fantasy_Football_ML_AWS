"""Failure, duplicate-delivery, and publication tests without live writes."""

import copy
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pytest
from botocore.exceptions import ClientError

from src.artifacts import model_sync
from src.maintenance import control, coordination, sources, storage, worker

pytestmark = pytest.mark.unit


class S3:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": '"' + hashlib.sha256(body).hexdigest() + '"'}

    def put_object(self, *, Bucket, Key, Body, IfMatch=None, IfNoneMatch=None, **kwargs):
        existing = self.objects.get(Key)
        if (IfNoneMatch and existing is not None) or (
            IfMatch
            and (existing is None or self.get_object(Bucket=Bucket, Key=Key)["ETag"] != IfMatch)
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = bytes(Body)
        self.writes.append(Key)
        return {"ETag": self.get_object(Bucket=Bucket, Key=Key)["ETag"]}

    def upload_file(self, filename, bucket, key):
        from pathlib import Path

        self.put_object(Bucket=bucket, Key=key, Body=Path(filename).read_bytes())

    def delete_object(self, *, Bucket, Key, IfMatch=None):
        if IfMatch != self.get_object(Bucket=Bucket, Key=Key)["ETag"]:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "DeleteObject")
        del self.objects[Key]


class Dynamo:
    def __init__(self):
        self.items = {}

    def put_item(self, *, Item, ExpressionAttributeValues, **kwargs):
        key = Item["pk"]["S"]
        old = self.items.get(key)
        if old and int(old["expires"]["N"]) >= int(ExpressionAttributeValues[":now"]["N"]):
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.items[key] = Item

    def get_item(self, *, Key, **kwargs):
        return {"Item": self.items.get(Key["pk"]["S"], {})}

    def delete_item(self, *, Key, ExpressionAttributeValues, **kwargs):
        key = Key["pk"]["S"]
        if self.items.get(key, {}).get("owner") != ExpressionAttributeValues[":owner"]:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "DeleteItem")
        del self.items[key]


def models(s3):
    for pos in storage.POSITIONS:
        key = model_sync.new_history_key("models", pos, "2026-09-11T00-00-00Z", "a" * 64)
        manifest = model_sync.build_manifest(
            key, "a" * 7, 3, "2026-09-11T00:00:00Z", smoke_passed=True
        )
        model_sync.write_manifest(s3, "bucket", "models", pos, manifest, expected_etag=None)
    return storage.model_pins(s3, "bucket")


@pytest.mark.parametrize(
    "key",
    [
        "models/QB/releases/history/one/model.tar.gz",
        model_sync.new_history_key("models", "RB", "one", "a" * 64),
    ],
)
def test_model_pins_reject_predecessor_and_wrong_position_keys(key):
    s3 = S3()
    models(s3)
    storage.put_json(
        s3,
        "bucket",
        model_sync.manifest_key("models", "QB"),
        {"stable": {"key": key, "bytes": 3}},
    )
    with pytest.raises(ValueError, match="No verified stable manifest for QB"):
        storage.model_pins(s3, "bucket")


def forecast():
    now = storage.now_iso()
    rows = [
        dict(player_id=p, name=p, position=p, team="NYJ", **dict.fromkeys(storage.MODELS, 1.0))
        for p in storage.POSITIONS
    ]
    return {
        "available": True,
        "season": 2026,
        "week": 1,
        "generated_at": now,
        "inputs_fetched_at": now,
        "degraded_positions": [],
        "scoring": {f: copy.deepcopy(rows) for f in storage.FORMATS},
        "sources": {"roster": {"covered_teams": ["NYJ"], "failed_teams": []}},
    }


def test_expired_owner_cannot_unlock_or_publish_after_successor():
    client = Dynamo()
    coordination.acquire(client, "table", "scope", "old", seconds=10, clock=lambda: 0)
    with pytest.raises(coordination.LeaseBusy):
        coordination.acquire(client, "table", "scope", "new", clock=lambda: 5)
    coordination.acquire(client, "table", "scope", "new", clock=lambda: 11)
    coordination.release(client, "table", "scope", "old")
    coordination.assert_owned(client, "table", "scope", "new", clock=lambda: 12)
    with pytest.raises(coordination.LeaseBusy):
        coordination.assert_owned(client, "table", "scope", "old", clock=lambda: 12)


def test_model_change_is_not_hidden_by_an_identical_artifact_size():
    s3 = S3()
    pins = models(s3)
    storage.assert_models_current(s3, "bucket", pins)
    key = pins["QB"]["manifest_key"]
    body, _ = storage.get_json(s3, "bucket", key)
    body["stable"]["key"] = "models/QB/releases/history/two/model.tar.gz"
    storage.put_json(s3, "bucket", key, body)
    with pytest.raises(RuntimeError, match="QB changed"):
        storage.assert_models_current(s3, "bucket", pins)


@pytest.mark.parametrize(
    "mutation", ["nan", "format", "position", "team", "stale", "missing-source", "unavailable"]
)
def test_forecast_validation_rejects_semantically_bad_output(mutation):
    payload = forecast()
    worker.validate_forecast(payload)
    if mutation == "nan":
        payload["scoring"]["ppr"][0]["nn_pred"] = float("nan")
    elif mutation == "format":
        payload["scoring"]["standard"][0]["player_id"] = "wrong"
    elif mutation == "position":
        payload["scoring"]["ppr"].pop()
    elif mutation == "team":
        payload["sources"]["roster"]["covered_teams"].append("BUF")
    elif mutation == "stale":
        payload["inputs_fetched_at"] = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    elif mutation == "missing-source":
        payload["sources"]["roster"]["failed_teams"] = ["BUF"]
    else:
        payload.update(available=False, reason="no_slate")
    with pytest.raises(ValueError):
        worker.validate_forecast(payload)


def test_optional_unavailable_source_and_verified_offseason_are_explicit():
    payload = forecast()
    payload["sources"]["history"] = {
        "coverage": {"qbr": {"status": "unavailable", "observed": 0, "expected": 3}}
    }
    worker.validate_forecast(payload)
    worker.validate_forecast(
        {"available": False, "reason": "offseason", "generated_at": storage.now_iso()}
    )


def test_sources_detect_corrections_without_a_new_season_or_week():
    current = ["2026-09-10T00:00:00Z"]

    def fetch(url):
        if urlsplit(url).hostname != "api.github.com":
            return b"season,week,home_team,away_team,gsis_id,pfr_id\n2025,1,NYJ,BUF,a,a", {}
        return json.dumps(
            {
                "assets": [
                    {
                        "name": "stats_player_week_2025.parquet",
                        "size": 100,
                        "updated_at": current[0],
                    }
                ]
            }
        ).encode(), {}

    first = sources.check_sources(fetcher=fetch)
    assert first["status"] == "complete"
    second = sources.check_sources(first, fetcher=fetch)
    assert {v["status"] for v in second["sources"].values()} == {"unchanged"}
    current[0] = "2026-09-11T00:00:00Z"
    third = sources.check_sources(second, fetcher=fetch)
    assert third["sources"]["player_stats"]["status"] == "changed"


def test_failed_source_check_does_not_become_unchanged():
    def fail(url):
        raise OSError("source unreachable")

    result = sources.check_sources(fetcher=fail)
    assert result["status"] == "partial"
    assert all(s["status"] == "fetch_failed" for s in result["sources"].values())


def test_forecast_publication_retains_a_newer_result():
    s3 = S3()
    old = forecast()
    old["inputs_fetched_at"] = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    key = "models/predictions_cache/upcoming_week.json"
    storage.put_json(s3, "bucket", key, old)
    result = {"run_id": "a" * 64, "request": {"data_release": "release"}}
    result["forecast"] = worker._record(
        s3,
        "bucket",
        result["run_id"],
        "upcoming_week.json",
        storage.json_bytes(forecast()),
        "application/json",
    )
    before = s3.objects[key]
    with pytest.raises(RuntimeError, match="newer forecast"):
        control.publish_forecast(s3, "bucket", result)
    assert s3.objects[key] == before


def test_corrupted_staging_object_cannot_be_published():
    s3 = S3()
    descriptor = worker._record(
        s3, "bucket", "a" * 64, "upcoming_week.json", b"valid", "application/json"
    )
    s3.objects[descriptor["key"]] = b"wrong"
    with pytest.raises(ValueError, match="checksum"):
        control.verified_blob(s3, "bucket", descriptor)


def test_expired_staging_cannot_be_published_on_resume():
    s3 = S3()
    payload = forecast()
    payload["inputs_fetched_at"] = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    result = {"run_id": "a" * 64, "request": {"data_release": "release"}}
    result["forecast"] = worker._record(
        s3,
        "bucket",
        result["run_id"],
        "upcoming_week.json",
        storage.json_bytes(payload),
        "application/json",
    )
    with pytest.raises(ValueError, match="expired"):
        control.publish_forecast(s3, "bucket", result)
    assert "models/predictions_cache/upcoming_week.json" not in s3.objects


def test_public_verification_waits_for_the_expected_run():
    s3 = S3()
    settings = {"bucket": "bucket", "url": "https://example.test"}
    run_id = "a" * 64
    storage.put_json(
        s3,
        "bucket",
        storage.run_key(run_id, "published.json"),
        {"run_id": run_id, "kind": "inference", "mode": "active"},
    )
    payload = forecast()
    payload["freshness"] = {"status": "fresh"}
    with pytest.raises(control.NotVisibleYet):
        control.verify(s3, None, settings, run_id, fetcher=lambda _: payload)
    payload["maintenance"] = {"run_id": run_id}
    assert "verified_at" in control.verify(s3, None, settings, run_id, fetcher=lambda _: payload)
    payload["freshness"]["status"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        control.verify(s3, None, settings, run_id, fetcher=lambda _: payload)


def test_watchdog_uses_input_age_even_when_the_http_response_claims_fresh():
    s3 = S3()
    now = datetime.now(UTC)
    storage.put_json(
        s3,
        "bucket",
        "maintenance/status/source-check.json",
        {"checked_at": now.isoformat(), "status": "complete"},
    )
    storage.put_json(
        s3, "bucket", "maintenance/status/weekly.json", {"verified_at": now.isoformat()}
    )
    payload = forecast()
    payload["freshness"] = {"status": "fresh"}
    settings = {"bucket": "bucket", "url": "https://example.test"}
    assert control.watchdog(s3, settings, fetcher=lambda _: payload, now=now)["ok"]
    payload["inputs_fetched_at"] = (now - timedelta(hours=5)).isoformat()
    assert not control.watchdog(s3, settings, fetcher=lambda _: payload, now=now)["ok"]
