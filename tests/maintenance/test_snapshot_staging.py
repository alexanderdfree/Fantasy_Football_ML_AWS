"""The canonical builder can stage its output without publishing production state."""

import json

import boto3
import pandas as pd
import pytest

from src.artifacts import model_sync, serving_snapshot, snapshot_state
from src.data.providers import snapshot as providers
from src.orchestration import datasets
from src.prediction import build_snapshot, comparison_snapshot, historical
from src.shared import evaluation_cohorts
from tests.maintenance.test_activation import cache_files
from tests.maintenance.test_contracts import S3, models

pytestmark = pytest.mark.unit


@pytest.fixture
def builder(monkeypatch, tmp_path):
    s3 = S3()
    pins = models(s3)
    dataset = "d" * 64
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.setenv("FF_DATA_RELEASE", dataset)
    monkeypatch.setenv("FF_DATASET_ID", dataset)
    monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
    monkeypatch.delenv("FF_LEGACY_RUN_ID", raising=False)
    monkeypatch.setattr(boto3, "client", lambda service: s3)
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(datasets, "materialize_dataset", lambda *a, **kw: calls.append("data"))
    monkeypatch.setattr(
        model_sync,
        "sync_models_from_s3",
        lambda: {"positions": [{"pos": p, "key": v["artifact"]["key"]} for p, v in pins.items()]},
    )
    owner = snapshot_state.ServingState(publish_remote=False)
    monkeypatch.setattr(snapshot_state, "ServingState", lambda **kw: owner)
    rows = pd.DataFrame(
        [
            {"position": p, **{f"{m}_pred_ppr": 1.0 for m in ("ridge", "nn", "attn_nn", "lgbm")}}
            for p in serving_snapshot.POSITIONS
        ]
    )
    monkeypatch.setattr(historical, "_ensure_metrics", lambda: owner.cache.update(results=rows))
    monkeypatch.setattr(
        providers, "assert_snapshot_sources_complete", lambda: calls.append("sources")
    )
    monkeypatch.setattr(evaluation_cohorts, "load_reference", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(comparison_snapshot, "build_comparison_snapshot", lambda *a, **kw: {})
    cache = tmp_path / "cache"
    monkeypatch.setattr(historical, "_PREDICTIONS_CACHE_DIR", cache)
    monkeypatch.setattr(
        historical,
        "_persist_cache_to_disk",
        lambda **kw: serving_snapshot.publish_local(cache, cache_files(b"verified-data")),
    )
    return s3, calls, tmp_path, dataset


def test_staging_runs_the_canonical_builder_without_remote_publication(builder):
    s3, calls, root, dataset = builder
    before = dict(s3.objects)
    target = root / "staged"
    assert build_snapshot.main(["--stage-directory", str(target)]) == 0
    assert calls == ["data", "sources"]
    assert s3.objects == before
    assert set(p.name for p in target.iterdir()) == {*serving_snapshot.FILES, "build.json"}
    build = json.loads((target / "build.json").read_text())
    assert build["dataset_id"] == dataset
    assert len(build["model_generations"]) == 6
    assert (target / "predictions.parquet").read_bytes() == b"verified-data"


def test_default_builder_still_publishes_a_verified_generation(builder):
    s3, calls, root, dataset = builder
    assert build_snapshot.main([]) == 0
    pointer, manifest = serving_snapshot.verify_remote(s3, "bucket", expected_dataset_id=dataset)
    assert len(manifest["models"]) == 6
    assert pointer["generation"]


def test_incomplete_sources_cannot_create_a_staged_or_published_candidate(builder, monkeypatch):
    s3, calls, root, dataset = builder
    before = dict(s3.objects)

    def unavailable():
        raise RuntimeError("provider snapshot incomplete")

    monkeypatch.setattr(providers, "assert_snapshot_sources_complete", unavailable)
    with pytest.raises(RuntimeError, match="provider snapshot incomplete"):
        build_snapshot.main(["--stage-directory", str(root / "staged")])
    assert s3.objects == before
    assert not (root / "staged").exists()


def test_staging_rejects_modified_local_generation_bytes(tmp_path):
    root = tmp_path / "cache"
    generation = serving_snapshot.publish_local(root, cache_files(b"original"))
    (generation / "predictions.parquet").write_bytes(b"corruption")
    token = serving_snapshot.SnapshotBuild(None, (), None, "d" * 64)
    with pytest.raises(ValueError, match="checksum"):
        build_snapshot.stage_cache(root, token, tmp_path / "staged", generation=generation.name)
    assert not (tmp_path / "staged").exists()
