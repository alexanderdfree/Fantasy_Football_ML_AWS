"""Published comparison cohorts survive artifact-only serving without raw data."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.artifacts import serving_snapshot
from src.serving import comparison, core, state
from src.serving.app import create_app
from src.serving.comparison_snapshot import build_comparison_snapshot
from src.shared.evaluation_cohorts import REFERENCE_VERSION
from tests.test_comparison_paired import records

pytestmark = pytest.mark.unit


def _reference(frame):
    result = frame[["player_id", "position", "season", "week"]].copy()
    result["reference_rank"] = np.arange(1, len(result) + 1)
    result["reference_version"] = REFERENCE_VERSION
    return result


def _owner(frame, block):
    metrics = {format_: {} for format_ in ("ppr", "half_ppr", "standard")}
    return state.ServingState(
        cache={
            "results": frame,
            "metrics": {},
            "metrics_by_format": metrics,
            "comparison_snapshot": block,
        },
        publish_remote=False,
    )


def _forbidden(*args, **kwargs):
    raise AssertionError("Published comparison must not evaluate or read raw references")


def test_published_comparison_never_reads_raw_references_or_recomputes(monkeypatch):
    frame = records()
    frame.loc[0, "nflcom_comparison_pred_ppr"] = np.nan
    block = build_comparison_snapshot(frame, reference=_reference(frame))
    owner = _owner(frame, block)
    owner.publish()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    monkeypatch.setattr(pd, "read_parquet", _forbidden)
    monkeypatch.setattr(comparison, "load_reference", _forbidden)
    monkeypatch.setattr(comparison, "comparison_tables", _forbidden)
    monkeypatch.setattr(comparison, "_load_comparison_experts", _forbidden)
    response = app.test_client().get("/api/comparison")
    assert response.status_code == 200
    assert response.get_json() == block
    coverage = response.get_json()["coverage"]["weekly_reference_top24"]["WR"]
    assert coverage["n"] == 23 and coverage["cohort_n"] == 24
    assert coverage["reference_status"] == "available"
    assert response.get_json()["subsets"]["weekly_reference_top24"]["WR"]["nflcom"]["mae"] == 7
    assert response.headers["X-FFP-Snapshot-Generation"] == owner.snapshots.current().generation


def test_comparison_persists_and_hydrates_without_raw_reference_files(tmp_path, monkeypatch):
    frame = records()
    block = build_comparison_snapshot(frame, reference=_reference(frame))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(core, "_compute_models_fingerprint", lambda: ("f" * 64, []))
    owner = _owner(frame, block)
    owner.cache["prediction_inputs_fingerprint"] = "f" * 64
    with state.use_state(owner):
        core._persist_cache_to_disk()
    assert (
        json.loads((serving_snapshot.active_directory(tmp_path) / "metrics.json").read_text())[
            "comparison_snapshot"
        ]
        == block
    )
    read = pd.read_parquet

    def artifact_only(path, *args, **kwargs):
        assert "data/raw" not in str(path)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", artifact_only)
    monkeypatch.setattr(comparison, "load_reference", _forbidden)
    monkeypatch.setattr(comparison, "comparison_tables", _forbidden)
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get("/api/comparison")
    assert response.status_code == 200
    assert response.get_json() == block
    assert app.extensions["ffp_state"].cache["comparison_snapshot"] == block


def test_missing_reference_is_serialized_as_unavailable(monkeypatch):
    monkeypatch.setattr(comparison, "load_reference", lambda: None)
    block = build_comparison_snapshot(records())
    coverage = block["coverage"]["weekly_reference_top24"]["WR"]
    assert coverage["status"] == "unavailable"
    assert coverage["reason"] == "reference_artifact_missing"
    assert all(cell is None for cell in block["subsets"]["weekly_reference_top24"]["WR"].values())
    assert block["coverage"]["all"]["WR"]["n"] == 30


@pytest.mark.parametrize("reference_available", [True, False])
def test_builder_persists_comparison_after_reference_and_before_publication(
    tmp_path, monkeypatch, reference_available
):
    import boto3

    from src.artifacts import model_sync, serving_snapshot
    from src.data.providers import snapshot as providers
    from src.scripts import build_evaluation_reference, build_serving_cache
    from src.shared import evaluation_cohorts

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "fixture-bucket")
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
    monkeypatch.delenv("FF_DATASET_ID", raising=False)
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: object())
    positions = build_serving_cache._ALL_POSITIONS
    generations = tuple((position, "etag", position + ".tar.gz") for position in positions)
    monkeypatch.setattr(
        serving_snapshot,
        "begin_build",
        lambda *args, **kwargs: SimpleNamespace(model_generations=generations),
    )
    monkeypatch.setattr(model_sync, "sync_data_from_s3", lambda: None)
    monkeypatch.setattr(
        model_sync,
        "sync_models_from_s3",
        lambda: {"positions": [{"pos": pos, "key": key} for pos, _, key in generations]},
    )
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(providers, "assert_snapshot_sources_complete", lambda: None)
    frame = pd.concat(
        [records().assign(position=position) for position in positions], ignore_index=True
    )
    events = []

    def hydrate():
        state.current_state().cache.update(_owner(frame, None).cache)

    monkeypatch.setattr(core, "_ensure_metrics", hydrate)

    def reference(*args, **kwargs):
        events.append("reference")
        if not reference_available:
            return None
        return _reference(records())

    monkeypatch.setattr(build_evaluation_reference, "write_reference", _forbidden)
    monkeypatch.setattr(evaluation_cohorts, "load_reference", reference)
    monkeypatch.setattr(comparison, "load_reference", lambda: None)
    persisted = {}

    def persist(*, required):
        assert required is True
        events.append("persist")
        persisted.update(copy.deepcopy(state.current_state().cache["comparison_snapshot"]))
        return SimpleNamespace(name="completed-generation")

    monkeypatch.setattr(core, "_persist_cache_to_disk", persist)

    def publish(*args, **kwargs):
        events.append("publish")
        assert kwargs["generation"] == "completed-generation"
        assert persisted == state.current_state().cache["comparison_snapshot"]
        return {"generation": "fixture-generation"}

    monkeypatch.setattr(serving_snapshot, "publish", publish)
    assert build_serving_cache.main() == 0
    assert events == ["reference", "persist", "publish"]
    coverage = persisted["coverage"]["weekly_reference_top24"]["WR"]
    assert coverage["status"] == ("available" if reference_available else "unavailable")
    assert persisted["subsets"]["all"]["WR"]["ridge"]["n"] == 30


@pytest.mark.parametrize("failure", [None, "io", "input_change"])
def test_final_cache_commit_is_required_before_remote_publication(tmp_path, monkeypatch, failure):
    import boto3

    from src.artifacts import model_sync
    from src.data.providers import snapshot as providers
    from src.orchestration import datasets
    from src.scripts import build_serving_cache
    from src.serving import comparison_snapshot
    from src.shared import evaluation_cohorts
    from tests.artifacts.test_serving_snapshot import MemoryS3, publish
    from tests.serving.test_state import _cache

    s3 = MemoryS3()
    publish(s3, tmp_path, "previous-release")
    previous_pointer = s3.objects["models/predictions_cache/current.json"]
    for name, value in {
        "FF_MODEL_S3_BUCKET": "bucket",
        "FF_MODEL_S3_PREFIX": "models",
        "FF_DATASET_ID": "a" * 64,
        "FF_DATA_RELEASE": "a" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FF_BUILD_PLAN_ID", raising=False)
    monkeypatch.delenv("FF_LEGACY_RUN_ID", raising=False)
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: s3)
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(datasets, "materialize_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_sync,
        "sync_models_from_s3",
        lambda: {
            "positions": [
                {
                    "pos": pos,
                    "key": model_sync.history_prefix("models", pos) + "original/model.tar.gz",
                }
                for pos in serving_snapshot.POSITIONS
            ]
        },
    )
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(providers, "assert_snapshot_sources_complete", lambda: None)
    monkeypatch.setattr(evaluation_cohorts, "load_reference", lambda **kwargs: None)
    monkeypatch.setattr(
        comparison_snapshot, "build_comparison_snapshot", lambda *args, **kwargs: {"proof": "final"}
    )

    def fingerprint():
        final = state.current_state().cache.get("comparison_snapshot") is not None
        return ("e" if failure == "input_change" and final else "f") * 64, []

    monkeypatch.setattr(core, "_compute_models_fingerprint", fingerprint)
    committed = []
    actual_publish = serving_snapshot.publish_local

    def publish_local(directory, files):
        comparison = json.loads(files["metrics.json"])["comparison_snapshot"]
        if failure == "io" and comparison is not None:
            raise OSError("final comparison persistence failed")
        generation = actual_publish(directory, files)
        committed.append(generation)
        return generation

    monkeypatch.setattr(serving_snapshot, "publish_local", publish_local)

    def construct_initial_generation():
        cache = _cache()
        cache["results"] = pd.concat(
            [cache["results"].assign(position=position) for position in serving_snapshot.POSITIONS],
            ignore_index=True,
        )
        cache.update(prediction_inputs_fingerprint="f" * 64, positions_loaded=set())
        state.current_state().cache.update(cache)
        core._persist_cache_to_disk()

    monkeypatch.setattr(core, "_ensure_metrics", construct_initial_generation)
    if failure is not None:
        with pytest.raises(RuntimeError, match="Required serving cache persistence failed"):
            build_serving_cache.main()
        assert len(committed) == 1
        assert (
            json.loads((committed[0] / "metrics.json").read_text())["comparison_snapshot"] is None
        )
        assert s3.objects["models/predictions_cache/current.json"] == previous_pointer
    else:
        assert build_serving_cache.main() == 0
        assert len(committed) == 2
        pointer = json.loads(s3.objects["models/predictions_cache/current.json"])
        base = pointer["manifest"].rsplit("/", 1)[0]
        payload = json.loads(s3.objects[f"{base}/metrics.json"])
        assert payload["comparison_snapshot"] == {"proof": "final"}
