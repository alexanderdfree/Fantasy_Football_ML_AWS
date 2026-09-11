"""Shared producer ordering and replay of a completed worker attempt."""

import json

import pandas as pd
import pytest

from src.data import maintenance_build
from src.maintenance import storage, worker
from tests.maintenance.test_contracts import S3, Dynamo, models

pytestmark = pytest.mark.unit


def test_ci_and_aws_producer_keeps_the_full_production_build_order(monkeypatch):
    from src.data import cache_io, loader, nfl_source, preprocessing, release, split
    from src.features import engineer

    events = []
    injury = pd.DataFrame({"player_id": ["a"]})
    roster = pd.DataFrame({"season": [2025] * 5, "week": [1, 2, 3, 4, 5]})
    monkeypatch.setattr(nfl_source, "injuries", lambda seasons: events.append("injuries") or injury)
    monkeypatch.setattr(
        nfl_source, "rosters_weekly", lambda seasons: events.append("rosters") or roster
    )
    monkeypatch.setattr(
        cache_io, "atomic_write_parquet", lambda *a, **k: events.append("cache-source")
    )
    monkeypatch.setattr(loader, "load_raw_data", lambda: events.append("raw") or "raw")
    monkeypatch.setattr(
        preprocessing, "preprocess", lambda df: events.append("preprocess") or "preprocessed"
    )

    def features(df, *, injuries_df, rosters_df):
        assert df == "preprocessed" and injuries_df is injury and rosters_df is roster
        events.append("features")
        return "features"

    monkeypatch.setattr(engineer, "build_features", features)
    monkeypatch.setattr(
        split,
        "temporal_split",
        lambda df: events.append("split") if df == "features" else pytest.fail("features bypassed"),
    )
    monkeypatch.setattr(release, "prewarm_training_dependencies", lambda: events.append("prewarm"))
    monkeypatch.setattr(release, "seal_inputs", lambda: events.append("seal") or {"sealed": True})
    assert maintenance_build.build() == {"sealed": True}
    assert events == [
        "injuries",
        "rosters",
        "cache-source",
        "cache-source",
        "raw",
        "preprocess",
        "features",
        "split",
        "prewarm",
        "seal",
    ]


def test_completed_worker_retry_does_not_recompute_or_republish(monkeypatch, tmp_path):
    s3 = S3()
    calls = []
    request = {
        "run_id": "a" * 64,
        "kind": "inference",
        "source_sha": "b" * 40,
        "producer_sha": "c" * 64,
        "data_release": "d" * 64,
        "models": models(s3),
    }
    (tmp_path / "maintenance-image.json").write_text(
        json.dumps(
            {"source_sha": request["source_sha"], "data_producer_sha256": request["producer_sha"]}
        )
    )
    monkeypatch.setattr(
        worker,
        "produce_forecast",
        lambda *a, **k: calls.append("infer") or {"forecast": {"key": "staged"}},
    )
    first = worker.execute(
        s3, Dynamo(), bucket="bucket", table="table", request=request, root=tmp_path
    )
    second = worker.execute(
        s3, Dynamo(), bucket="bucket", table="table", request=request, root=tmp_path
    )
    assert first == second
    assert calls == ["infer"]
    assert all(not key.startswith("models/predictions_cache/") for key in s3.writes)


def test_worker_rejects_a_differently_baked_image(tmp_path):
    (tmp_path / "maintenance-image.json").write_text(
        json.dumps({"source_sha": "wrong", "data_producer_sha256": "recipe"})
    )
    with pytest.raises(ValueError, match="provenance"):
        worker.execute(
            S3(),
            Dynamo(),
            bucket="bucket",
            table="table",
            request={"run_id": "a" * 64, "source_sha": "expected", "producer_sha": "recipe"},
            root=tmp_path,
        )


def test_shadow_source_check_does_not_refresh_production_health(monkeypatch, tmp_path):
    from src.maintenance import readiness, sources

    s3 = S3()
    request = {
        "run_id": "a" * 64,
        "kind": "inference",
        "mode": "shadow",
        "check_sources": True,
        "source_sha": "b" * 40,
        "producer_sha": "c" * 64,
        "models": models(s3),
    }
    (tmp_path / "maintenance-image.json").write_text(
        json.dumps(
            {"source_sha": request["source_sha"], "data_producer_sha256": request["producer_sha"]}
        )
    )
    monkeypatch.setattr(
        sources,
        "check_sources",
        lambda previous: {"status": "complete", "checked_at": storage.now_iso()},
    )
    monkeypatch.setattr(worker, "produce_forecast", lambda *a, **k: {"forecast": {"key": "staged"}})
    s3.objects["staged"] = b"{}"
    monkeypatch.setattr(
        readiness, "complete_report", lambda check, *a, **k: {**check, "readiness": "ready"}
    )
    worker.execute(s3, Dynamo(), bucket="bucket", table="table", request=request, root=tmp_path)
    assert "maintenance/shadow/source-check.json" in s3.objects
    assert "maintenance/status/source-check.json" not in s3.objects


def test_worker_bootstrap_uses_the_configured_aws_region(monkeypatch):
    import sys

    import boto3

    s3 = S3()
    run_id = "a" * 64
    key = storage.run_key(run_id, "request.json")
    storage.put_json(s3, "bucket", key, {"run_id": run_id})
    calls = []

    def client(service, **options):
        calls.append((service, options))
        return s3 if service == "s3" else object()

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(
        worker,
        "execute",
        lambda *a, **k: {"run_id": run_id, "kind": "inference", "completed_at": storage.now_iso()},
    )
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "bucket")
    monkeypatch.setenv("FF_MAINTENANCE_LOCK_TABLE", "table")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setattr(sys, "argv", ["worker", "--request-key", key])
    worker.main()
    assert calls == [
        ("s3", {"region_name": "us-west-2"}),
        ("dynamodb", {"region_name": "us-west-2"}),
    ]
