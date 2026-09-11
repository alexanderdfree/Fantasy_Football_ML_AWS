"""Application ownership, immutable request generations, and artifact-only serving."""

import io
import json
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from flask import jsonify

from src.artifacts import serving_snapshot
from src.serving import core, state
from src.serving.app import create_app

pytestmark = pytest.mark.unit


def _cache(value=10.0, *, degraded=False):
    frame = pd.DataFrame(
        {
            "player_id": ["qb", "te"],
            "player_display_name": ["Quarterback", "Tight End"],
            "position": ["QB", "TE"],
            "recent_team": "KC",
            "season": 2025,
            "week": 1,
        }
    )
    for scoring, actual in (
        ("ppr", "fantasy_points"),
        ("half_ppr", "fantasy_points_half_ppr"),
        ("standard", "fantasy_points_standard"),
    ):
        frame[actual] = value
        for family in ("ridge", "nn", "attn_nn", "lgbm"):
            frame[f"{family}_pred_{scoring}"] = [value, np.nan if degraded else value]
    metrics = {scoring: {"value": value} for scoring in ("ppr", "half_ppr", "standard")}
    return {
        "results": frame,
        "metrics": metrics["ppr"],
        "metrics_by_format": metrics,
        "positions_loaded": set(core._ALL_POSITIONS) - ({"TE"} if degraded else set()),
        "position_load_errors": {"TE_nn": "unavailable"} if degraded else {},
        "model_bundle_ids": {"QB": {"nn": "served-id"}},
        "model_metadata": {},
    }


def _forbid_compute(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Artifact-only request attempted model/data computation")

    for name in (
        "_load_base_data_locked",
        "_load_splits_locked",
        "_apply_position_models",
        "_compute_metrics_locked",
    ):
        monkeypatch.setattr(core, name, forbidden)


def _write_generation(directory, cache, *, snapshot=None):
    parquet = io.BytesIO()
    cache["results"].to_parquet(parquet)
    payload = {
        key: cache.get(key, {})
        for key in (
            "metrics_by_format",
            "position_load_errors",
            "model_bundle_ids",
            "model_metadata",
            "position_details",
        )
    }
    files = {
        "predictions.parquet": parquet.getvalue(),
        "metrics.json": json.dumps(payload).encode(),
        "fingerprint.json": json.dumps(
            {"schema_version": core._PREDICTIONS_CACHE_SCHEMA_VERSION, "sha256": "f" * 64}
        ).encode(),
        "snapshot.json": json.dumps(
            snapshot or {"scoring": {}, "weeks": [1], "degraded_positions": []}
        ).encode(),
    }
    return serving_snapshot.publish_local(directory, files).name


def test_app_factories_have_independent_state_and_contract_headers():
    owners = [state.ServingState(cache=_cache(value)) for value in (1.0, 2.0)]
    applications = [create_app(serving_state=owner) for owner in owners]
    for owner in owners:
        owner.publish()
    for application, owner, value in zip(applications, owners, (1.0, 2.0), strict=True):
        response = application.test_client().get("/api/metrics")
        assert response.get_json() == {"value": value}
        assert response.headers["X-FFP-Contract-Version"] == "1.0"
        assert response.headers["X-FFP-Snapshot-Generation"] == owner.snapshots.current().generation
    assert state.current_snapshot() is None


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("loaded", [set(), {"QB"}])
def test_health_sanitizes_diagnostics_without_mutating_owned_errors(published, loaded):
    diagnostic = "FileNotFoundError('/private/model/config.json?token=internal-only')"
    cache = _cache()
    cache.update(
        positions_loaded=loaded,
        position_load_errors={"TE_nn": diagnostic},
        base_load_error=diagnostic,
    )
    owner = state.ServingState(cache=cache)
    if published:
        owner.publish()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get("/health")
    assert response.status_code == (200 if loaded else 503)
    body = response.get_json()
    assert body["status"] == ("degraded" if loaded else "unhealthy")
    assert body["position_load_errors"] == {"TE_nn": "Position or model initialization failed"}
    assert body["base_load_error"] == "Shared data initialization failed"
    assert "internal-only" not in response.get_data(as_text=True)
    assert "/private/" not in response.get_data(as_text=True)
    assert owner.cache["position_load_errors"]["TE_nn"] == diagnostic
    assert owner.cache["base_load_error"] == diagnostic
    if published:
        assert owner.snapshots.current().cache["position_load_errors"]["TE_nn"] == diagnostic


def test_request_keeps_captured_generation_during_background_publication():
    owner = state.ServingState(cache=_cache(1.0))
    old = owner.publish()
    application = create_app(serving_state=owner)

    @application.get("/api/capture-probe")
    def capture_probe():
        before = state._cache["metrics"]["value"]

        def publish():
            owner.cache = _cache(2.0)
            owner.publish()

        thread = threading.Thread(target=publish)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        return jsonify(before=before, after=state._cache["metrics"]["value"])

    response = application.test_client().get("/api/capture-probe")
    assert response.get_json() == {"before": 1.0, "after": 1.0}
    assert response.headers["X-FFP-Snapshot-Generation"] == old.generation
    following = application.test_client().get("/api/metrics")
    assert following.get_json() == {"value": 2.0}
    assert following.headers["X-FFP-Snapshot-Generation"] != old.generation


@pytest.mark.parametrize(
    "path",
    [
        "/api/predictions?position=ALL",
        "/api/predictions?position=QB",
        "/api/player/qb",
        "/api/weeks",
        "/api/teams",
    ],
)
def test_cold_artifact_only_endpoints_never_start_computation(tmp_path, monkeypatch, path):
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get(path)
    assert response.status_code == 503
    assert response.headers["X-FFP-Contract-Version"] == "1.0"


def test_readiness_and_first_degraded_position_request_hydrate_without_models(
    tmp_path, monkeypatch
):
    cache = _cache(degraded=True)
    _write_generation(tmp_path, cache)
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)
    owner = state.ServingState()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get("/api/predictions?position=TE")
    assert response.status_code == 200
    assert response.get_json()["players"][0]["nn_pred"] is None
    assert response.get_json()["degraded_positions"] == ["TE"]
    ready = app.test_client().get("/ready")
    assert ready.status_code == 200
    assert ready.get_json()["generation"] == owner.snapshots.current().generation
    assert "splits" not in owner.cache


def test_artifact_only_background_entrypoints_hydrate_in_app_context(tmp_path, monkeypatch):
    _write_generation(tmp_path, _cache(degraded=True))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})
    with app.app_context():
        core._ensure_position_loaded("TE")
        core._ensure_all_positions_loaded()
        assert state._cache["model_bundle_ids"]["QB"]["nn"] == "served-id"


def test_cold_readiness_is_warming_and_health_is_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})
    assert app.test_client().get("/ready").status_code == 503
    assert app.test_client().get("/health").get_json() == {"status": "ok"}


@pytest.mark.parametrize("matches", [True, False])
def test_prediction_metadata_requires_exact_returned_bundle_identity(monkeypatch, matches):
    cache = _cache()
    owner = state.ServingState(cache=cache)
    frame = cache["results"].iloc[:1].copy()
    targets = core.POSITION_REGISTRY["QB"]["targets"]
    document = {
        "inputs": {"features": ["saved_input"], "targets": targets},
        "architecture": {"class": "MultiHeadNet", "kwargs": {"backbone_layers": [7]}},
        "training_options": {"nn_lr": 0.002},
        "provenance": {"data_id": "training"},
    }
    prediction = SimpleNamespace(
        frame=frame,
        raw={"nn": {target: np.array([1.0]) for target in targets}},
        totals={"nn": {scoring: np.array([10.0]) for scoring in ("ppr", "half_ppr", "standard")}},
        errors={},
        details={},
        bundle_ids={"nn": "expected"},
    )
    monkeypatch.setattr(
        "src.prediction.frames.predict_position", lambda *args, **kwargs: prediction
    )
    descriptor = SimpleNamespace(
        bundle_id="expected" if matches else "newer", to_dict=lambda: document
    )
    monkeypatch.setattr("src.prediction.bundle.read_bundle", lambda *args, **kwargs: descriptor)
    with state.use_state(owner):
        core._apply_position_models(frame, frame, frame, "QB", cache["results"])
    metadata = owner.cache["model_metadata"]["QB"]["nn"]
    assert metadata["status"] == ("available" if matches else "unavailable")
    assert metadata["bundle_id"] == "expected"
    if matches:
        assert metadata["inputs"]["features"] == ["saved_input"]
    else:
        assert "inputs" not in metadata


def _served_metadata():
    return {
        "QB": {
            "nn": {
                "bundle_id": "served-id",
                "status": "available",
                "inputs": {"features": ["saved_input"], "targets": ["passing_yards"]},
                "architecture": {
                    "class": "MultiHeadNet",
                    "kwargs": {"backbone_layers": [7], "head_hidden": 3, "dropout": 0.125},
                },
                "training_options": {"nn_lr": 0.002, "nn_epochs": 4, "scheduler_type": "plateau"},
                "provenance": {
                    "dependencies": {"torch": "recorded-test-version"},
                    "data_id": "fitted-input",
                },
            }
        }
    }


def test_architecture_uses_served_bundle_and_labels_configuration_fallback():
    cache = _cache()
    cache["model_metadata"] = _served_metadata()
    owner = state.ServingState(cache=cache)
    owner.publish()
    response = create_app(serving_state=owner).test_client().get("/api/model_architecture")
    assert response.status_code == 200
    body = response.get_json()
    qb = body["positions"]["QB"]
    assert qb["metadata_source"] == "bundle" and qb["metadata_family"] == "nn"
    assert qb["backbone_layers"] == [7]
    assert qb["feature_count"] == 1 and qb["features"] == {"other": ["saved_input"]}
    assert qb["lr"] == 0.002 and qb["epochs"] == 4
    assert body["positions"]["TE"]["metadata_source"] == "config_fallback"
    assert "recorded-test-version" in body["overview"]["framework"]


def test_model_metadata_survives_persist_and_hydration_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(core, "_compute_models_fingerprint", lambda: ("f" * 64, []))
    source = state.ServingState(cache=_cache(), publish_remote=False)
    source.cache["model_metadata"] = _served_metadata()
    source.cache["prediction_inputs_fingerprint"] = "f" * 64
    source.cache["positions_mtime"] = {
        position: core.refresh_sentinel_mtime(position) for position in core._ALL_POSITIONS
    }
    with state.use_state(source):
        core._persist_cache_to_disk()
    _forbid_compute(monkeypatch)
    target = state.ServingState()
    app = create_app(serving_state=target, config={"ALLOW_RUNTIME_INFERENCE": False})
    assert app.test_client().get("/ready").status_code == 200
    assert target.cache["model_metadata"] == source.cache["model_metadata"]
    qb = app.test_client().get("/api/model_architecture").get_json()["positions"]["QB"]
    assert qb["backbone_layers"] == [7] and qb["metadata_source"] == "bundle"


def test_healthy_generation_replaces_previous_errors_and_position_details(tmp_path, monkeypatch):
    cache = _cache(degraded=True)
    cache["position_details"] = {"TE": {"target_metrics": {"total": {"nn_mae": 99.0}}}}
    _write_generation(tmp_path, cache)
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)
    owner = state.ServingState()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    assert app.test_client().get("/ready").status_code == 200
    assert app.test_client().get("/health").get_json()["status"] == "degraded"
    healthy_generation = _write_generation(tmp_path, _cache())
    with app.app_context():
        core._ensure_metrics()
    response = app.test_client().get("/health")
    assert response.get_json() == {"status": "ok"}
    assert response.headers["X-FFP-Snapshot-Generation"] == healthy_generation
    assert owner.cache["position_details"] == {}
    assert owner.cache["position_load_errors"] == {}


@pytest.mark.parametrize("mode", ["captured", "in_memory", "cold"])
def test_snapshot_file_matches_captured_generation_while_download_pointer_advances(
    tmp_path, monkeypatch, mode
):
    old = _write_generation(tmp_path, _cache(1), snapshot={"source_generation": "a"})
    new = _write_generation(tmp_path, _cache(2), snapshot={"source_generation": "b"})
    # A loose legacy file must not override a committed or captured generation.
    (tmp_path / "snapshot.json").write_text(json.dumps({"source_generation": "legacy"}))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    owner = state.ServingState(cache=_cache())
    if mode != "cold":
        owner.cache["snapshot_generation"] = old if mode == "captured" else None
        owner.publish()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get("/api/snapshot")
    assert response.status_code == 200
    if mode == "in_memory":
        assert "source_generation" not in response.get_json()
        assert response.get_json()["scoring"]["ppr"][0]["ridge_pred"] == 10.0
    else:
        assert response.get_json()["source_generation"] == ("a" if mode == "captured" else "b")
    expected = owner.snapshots.current().generation if mode != "cold" else new
    assert response.headers["X-FFP-Snapshot-Generation"] == expected
    with app.app_context():
        assert core._cache_read_directory() == tmp_path / "generations" / new


def test_revocation_reaches_hydrated_and_replacement_workers_without_local_inputs(
    tmp_path, monkeypatch
):
    old = _write_generation(tmp_path, _cache(1))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    _forbid_compute(monkeypatch)

    def forbidden(*args):
        raise AssertionError("Artifact-only worker inspected local inputs")

    monkeypatch.setattr(core, "_compute_models_fingerprint", forbidden)
    monkeypatch.setattr(core, "refresh_sentinel_mtime", forbidden)
    workers = [create_app(config={"ALLOW_RUNTIME_INFERENCE": False}) for _ in range(2)]
    for worker in workers:
        assert worker.test_client().get("/api/metrics").get_json() == {"value": 1}
    serving_snapshot.invalidate_generation(tmp_path, old)
    workers.append(create_app(config={"ALLOW_RUNTIME_INFERENCE": False}))
    for worker in workers:
        assert worker.test_client().get("/api/metrics").status_code == 503
        assert worker.test_client().get("/api/snapshot").status_code == 404
    new = _write_generation(tmp_path, _cache(2))
    for worker in workers:
        response = worker.test_client().get("/api/metrics")
        assert response.get_json() == {"value": 2}
        assert response.headers["X-FFP-Snapshot-Generation"] == new
    assert (tmp_path / "generations" / old / "predictions.parquet").exists()


def test_hydration_deserializes_verified_bytes_even_if_files_change(tmp_path, monkeypatch):
    generation = _write_generation(tmp_path, _cache(1))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    read = pd.read_parquet
    swapped = []

    def swap_after_verification(buffer, *args, **kwargs):
        assert isinstance(buffer, io.BytesIO)
        _cache(99)["results"].to_parquet(
            tmp_path / "generations" / generation / "predictions.parquet"
        )
        swapped.append(True)
        return read(buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", swap_after_verification)
    owner = state.ServingState()
    app = create_app(serving_state=owner, config={"ALLOW_RUNTIME_INFERENCE": False})
    assert app.test_client().get("/api/metrics").get_json() == {"value": 1}
    assert swapped == [True]
    assert owner.cache["results"]["ridge_pred_ppr"].eq(1).all()
    assert app.test_client().get("/api/snapshot").status_code == 404


def test_snapshot_response_uses_verified_bytes_without_reopening_the_file(tmp_path, monkeypatch):
    generation = _write_generation(tmp_path, _cache(1), snapshot={"source_generation": "verified"})
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    original = core._verified_cache_bytes
    swapped = []

    def swap_after_verification(*args, **kwargs):
        selected = original(*args, **kwargs)
        (tmp_path / "generations" / generation / "snapshot.json").write_text(
            '{"source_generation":"tampered"}'
        )
        swapped.append(True)
        return selected

    monkeypatch.setattr(core, "_verified_cache_bytes", swap_after_verification)
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})
    response = app.test_client().get("/api/snapshot")
    assert response.get_json() == {"source_generation": "verified"}
    assert response.headers["X-FFP-Snapshot-Generation"] == generation
    assert swapped == [True]


def test_mid_request_revocation_rejects_body_and_retains_contract_headers(tmp_path, monkeypatch):
    generation = _write_generation(tmp_path, _cache(1))
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    app = create_app(config={"ALLOW_RUNTIME_INFERENCE": False})

    @app.get("/api/revocation-probe")
    def revoke():
        serving_snapshot.invalidate_generation(tmp_path, generation)
        return jsonify(value=state._cache["metrics"]["value"])

    assert app.test_client().get("/ready").status_code == 200
    response = app.test_client().get("/api/revocation-probe")
    assert response.status_code == 503
    assert response.get_json() == {"error": "Serving snapshot was revoked"}
    assert response.headers["X-FFP-Contract-Version"] == "1.0"
    assert "X-FFP-Snapshot-Generation" not in response.headers
