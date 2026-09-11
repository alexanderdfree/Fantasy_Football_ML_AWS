"""Canonical output collection cannot substitute another selected data release."""

import hashlib
import json

import pytest

from src.artifacts import intent, model_sync, receipts
from src.orchestration.datasets import DatasetError
from tests.orchestration.test_datasets import MemoryS3
from tests.shared._helpers import make_tarball

pytestmark = pytest.mark.unit


def test_planless_receipt_authenticates_release_and_checks_it_before_download(tmp_path):
    s3 = MemoryS3()
    source, dataset = "a" * 40, "b" * 64
    reservation = intent.reserve_intent(s3, "bucket", "models", "QB", source, dataset, "run")
    metrics = {
        "position": "QB",
        "git_sha": source,
        "dataset_id": dataset,
        "data_release": dataset,
        "data_format": "data-release-v1",
        "publication_intent": reservation,
        "publication_revision": None,
    }
    payload = make_tarball({"benchmark_metrics.json": json.dumps(metrics).encode()})
    digest = hashlib.sha256(payload).hexdigest()
    key = model_sync.new_history_key("models", "QB", "test", digest)
    s3.objects[key] = payload
    entry = {**metrics, "key": key, "sha256": digest, "bytes": len(payload), "smoke_passed": True}
    receipts.claim_successful_output(s3, "bucket", "models", "QB", entry)
    target = tmp_path / "model.tar.gz"
    with pytest.raises(DatasetError, match="selected data release"):
        receipts.download_run_artifact(
            s3, "bucket", "models", source, "QB", "run", target, expected_dataset_id="c" * 64
        )
    assert not target.exists()
    assert (
        receipts.download_run_artifact(
            s3, "bucket", "models", source, "QB", "run", target, expected_dataset_id=dataset
        )
        == metrics
    )


def test_mismatched_release_alias_cannot_poison_accepted_output_slot():
    s3 = MemoryS3()
    source, dataset = "a" * 40, "b" * 64
    reservation = intent.reserve_intent(s3, "bucket", "models", "QB", source, dataset, "run")
    entry = {
        "position": "QB",
        "git_sha": source,
        "dataset_id": dataset,
        "data_release": "c" * 64,
        "data_format": "data-release-v1",
        "publication_intent": reservation,
        "publication_revision": None,
        "key": model_sync.new_history_key("models", "QB", "test", "d" * 64),
        "sha256": "d" * 64,
        "bytes": 1,
        "smoke_passed": True,
    }
    with pytest.raises(DatasetError, match="data release differs"):
        receipts.claim_successful_output(s3, "bucket", "models", "QB", entry)
    assert receipts.run_receipt_key("models", source, "QB", "run") not in s3.objects
