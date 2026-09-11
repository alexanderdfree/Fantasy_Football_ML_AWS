import hashlib
import json

import pytest

from src.artifacts import model_sync, receipts
from src.orchestration import datasets
from tests.orchestration.test_build_plan import _plan
from tests.orchestration.test_datasets import published  # noqa: F401
from tests.shared._helpers import make_tarball

pytestmark = pytest.mark.unit


def _published_artifact(published):
    s3, plan_id, plan = _plan(published)
    metrics = {
        "position": "QB",
        "build_plan_id": plan_id,
        "dataset_id": plan["dataset_id"],
        "data_release": plan["dataset_id"],
        "data_format": plan["data_format"],
        "git_sha": plan["git_sha"],
        "publication_intent": plan["intents"]["QB"],
        "publication_revision": plan["publication_revisions"]["QB"],
    }
    data = make_tarball({"benchmark_metrics.json": json.dumps(metrics).encode()})
    entry = {
        **metrics,
        "key": model_sync.history_prefix("models", "QB") + "exact/model.tar.gz",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "smoke_passed": True,
    }
    s3.objects[entry["key"]] = data
    receipts.publish_receipt(s3, "bucket", plan_id, "QB", entry)
    return s3, plan_id, entry, metrics


def test_receipt_resolves_original_artifact_after_later_global_promotion(published, tmp_path):
    s3, plan_id, _, metrics = _published_artifact(published)
    s3.objects[model_sync.manifest_key("models", "QB")] = json.dumps(
        {"stable": {"key": "other-plan"}}
    ).encode()
    assert (
        receipts.download_receipt_artifact(s3, "bucket", plan_id, "QB", tmp_path / "model.tar.gz")
        == metrics
    )


def test_receipt_rejects_wrong_plan_content_or_failed_smoke(published):
    s3, plan_id, entry, _ = _published_artifact(published)
    entry["smoke_passed"] = False
    s3.objects[receipts.receipt_key(plan_id, "QB")] = json.dumps(
        {
            "schema_version": 1,
            "position": "QB",
            **entry,
        }
    ).encode()
    with pytest.raises(datasets.DatasetError, match="smoke_passed"):
        receipts.load_receipt(s3, "bucket", plan_id, "QB")


def test_benchmark_plan_mode_uses_receipt_and_refuses_fallback(published, monkeypatch):
    from src.batch import benchmark

    s3, plan_id, _, metrics = _published_artifact(published)
    monkeypatch.setenv("FF_BUILD_PLAN_ID", plan_id)
    monkeypatch.setattr(benchmark.boto3, "client", lambda *_, **__: s3)
    monkeypatch.setattr(benchmark, "S3_BUCKET", "bucket")
    monkeypatch.setattr(benchmark, "load_manifest", lambda *_: pytest.fail("mutable manifest used"))
    assert benchmark.download_metrics(["QB"]) == {"QB": metrics}
    del s3.objects[receipts.receipt_key(plan_id, "QB")]
    with pytest.raises(Exception, match="NoSuchKey"):
        benchmark.download_metrics(["QB"])


def test_receipt_checksum_failure_rejects_corrupt_published_object(published, tmp_path):
    s3, plan_id, entry, _ = _published_artifact(published)
    s3.objects[entry["key"]] = b"corrupt"
    with pytest.raises(datasets.DatasetError, match="checksum"):
        receipts.download_receipt_artifact(s3, "bucket", plan_id, "QB", tmp_path / "model.tar.gz")
