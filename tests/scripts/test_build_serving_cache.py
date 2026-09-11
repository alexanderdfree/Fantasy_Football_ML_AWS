"""Import + guard smoke tests for the off-container serving-cache builder.

Operator CLI (src/scripts/build_serving_cache.py): an import-smoke test makes
signature/import drift fail the unit shard instead of surfacing only when the
refresh-splits workflow runs it. The no-bucket guard is exercised directly — it
returns before importing torch/serving, so it stays fast and dependency-light.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

_MODULE = "src.scripts.build_serving_cache"


def test_module_imports_and_exposes_main():
    mod = importlib.import_module(_MODULE)
    assert callable(mod.main)


def test_main_refuses_without_s3_bucket(monkeypatch):
    """No FF_MODEL_S3_BUCKET -> exit 1 before any S3 sync / heavy import."""
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    mod = importlib.import_module(_MODULE)
    assert mod.main() == 1


def test_legacy_build_selects_exact_receipts_for_its_trained_subset(monkeypatch):
    from src.artifacts import receipts
    from src.scripts.build_serving_cache import expected_model_keys

    calls = []

    def load(s3, bucket, prefix, source_sha, position, run_id):
        calls.append((s3, bucket, prefix, source_sha, position, run_id))
        return {
            "key": f"models/releases/v3/{position}/history/this-run/model.tar.gz",
            "image_id": "image@sha256:digest",
        }

    monkeypatch.setattr(receipts, "load_run_receipt", load)
    monkeypatch.setenv("FF_BUILD_POSITIONS", "QB TE")
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", "a" * 40)
    monkeypatch.setenv("FF_TRAIN_IMAGE_ID", "image@sha256:digest")
    result = expected_model_keys(None, "b", "models", legacy_run_id="ec2:42:1")
    assert result == {
        pos: f"models/releases/v3/{pos}/history/this-run/model.tar.gz" for pos in ("QB", "TE")
    }
    assert calls == [(None, "b", "models", "a" * 40, pos, "ec2:42:1") for pos in ("QB", "TE")]
    monkeypatch.setenv("FF_TRAIN_IMAGE_ID", "different-image")
    with pytest.raises(RuntimeError, match="image differs"):
        expected_model_keys(None, "b", "models", legacy_run_id="ec2:42:1")


def test_legacy_build_never_falls_back_when_its_receipt_is_missing(monkeypatch):
    from src.artifacts import receipts
    from src.scripts.build_serving_cache import expected_model_keys

    def missing(*_):
        raise RuntimeError("exact run receipt missing")

    monkeypatch.setattr(receipts, "load_run_receipt", missing)
    monkeypatch.setenv("FF_BUILD_POSITIONS", "DST")
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", "a" * 40)
    with pytest.raises(RuntimeError, match="exact run receipt missing"):
        expected_model_keys(None, "b", "models", legacy_run_id="ec2:42:1")


def test_training_cache_build_requires_unambiguous_source_identity(monkeypatch):
    from src.scripts.build_serving_cache import expected_model_keys

    with pytest.raises(RuntimeError, match="both plan and legacy"):
        expected_model_keys(None, "b", "models", plan_id="plan", legacy_run_id="run")
    monkeypatch.delenv("FF_BUILD_POSITIONS", raising=False)
    with pytest.raises(RuntimeError, match="FF_BUILD_POSITIONS"):
        expected_model_keys(None, "b", "models", legacy_run_id="run")


def test_planned_build_checks_the_selected_namespace_and_dataset(monkeypatch):
    from src.artifacts import receipts
    from src.orchestration import build_plan
    from src.scripts.build_serving_cache import expected_model_keys

    monkeypatch.setattr(
        build_plan,
        "load_plan",
        lambda *_: {"dataset_id": "data", "model_prefix": "models", "positions": ["K"]},
    )
    monkeypatch.setattr(receipts, "load_receipt", lambda *_: {"key": "exact-k"})
    assert expected_model_keys(None, "b", "models", plan_id="plan", dataset_id="data") == {
        "K": "exact-k"
    }
    for prefix, data in (("other", "data"), ("models", "other")):
        with pytest.raises(RuntimeError, match="source differs"):
            expected_model_keys(None, "b", prefix, plan_id="plan", dataset_id=data)


@pytest.mark.parametrize("receipt_dataset", ["a" * 64, "b" * 64, None])
def test_standalone_receipts_must_match_selected_canonical_data_release(
    monkeypatch, receipt_dataset
):
    from src.artifacts import receipts
    from src.scripts.build_serving_cache import expected_model_keys

    monkeypatch.setenv("FF_BUILD_POSITIONS", "QB")
    monkeypatch.setattr(
        receipts,
        "load_run_receipt",
        lambda *args: {"key": "model-key", "dataset_id": receipt_dataset},
    )
    if receipt_dataset == "a" * 64:
        assert expected_model_keys(
            None, "bucket", "models", dataset_id="a" * 64, legacy_run_id="standalone"
        ) == {"QB": "model-key"}
    else:
        with pytest.raises(RuntimeError, match="data release differs"):
            expected_model_keys(
                None, "bucket", "models", dataset_id="a" * 64, legacy_run_id="standalone"
            )
