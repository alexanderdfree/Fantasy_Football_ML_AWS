"""Import + guard smoke tests for the off-container serving-cache builder.

Operator CLI (src/scripts/build_serving_cache.py): an import-smoke test makes
signature/import drift fail the unit shard instead of surfacing only when the
refresh-splits workflow runs it. The no-bucket guard is exercised directly — it
returns before importing torch/serving, so it stays fast and dependency-light.
"""

from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

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


@pytest.fixture
def cache_builder(monkeypatch, tmp_path):
    """Real generation IO/hydration, stubbed S3 and expensive inference only."""
    from src.scripts import build_evaluation_reference
    from src.serving import app, core
    from src.shared import evaluation_cohorts, model_sync, prediction_cache

    mod = importlib.import_module(_MODULE)
    state = SimpleNamespace(
        variant="valid", events=[], uploaded=[], rebuild_errors={}, fail_upload=False
    )
    results = pd.DataFrame(
        {
            "position": list(mod._ALL_POSITIONS),
            "player_id": list(mod._ALL_POSITIONS),
            "week": [1] * 6,
            "fantasy_points": [10.0] * 6,
            "ridge_pred_ppr": [11.0] * 6,
        }
    )
    metrics = {"ppr": {"Ridge Regression": {"overall": {"mae": 1.0}, "by_position": []}}}
    # Data hydration already supplies the archived cohort. Cache reuse/rebuild
    # must leave those historical bytes intact, including on failure paths.
    reference_dir = tmp_path / "data" / "raw"
    reference_dir.mkdir(parents=True)
    reference = results[["position", "player_id", "week"]].assign(season=2025)
    reference_path = reference_dir / evaluation_cohorts.REFERENCE_FILENAME
    reference.to_parquet(reference_path, index=False)
    reference_bytes = reference_path.read_bytes()
    reference_mtime = reference_path.stat().st_mtime_ns
    monkeypatch.setattr(evaluation_cohorts, "CACHE_DIR", str(reference_dir))
    cache_dir = tmp_path / "serving_cache"
    monkeypatch.setattr(app, "_cache", {})
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(core, "_compute_models_fingerprint", lambda: ("live-inputs", []))
    monkeypatch.setattr(core, "refresh_sentinel_mtime", lambda pos: 0.0)
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "serving-cache-test")
    monkeypatch.setattr(model_sync, "sync_data_from_s3", lambda: state.events.append("data"))
    monkeypatch.setattr(model_sync, "sync_models_from_s3", lambda: state.events.append("models"))

    def sync_cache():
        state.events.append("cache")
        if state.variant == "missing":
            return {"files": 0, "missing": ["cache.tar.gz"]}
        frame = results.iloc[:-1] if state.variant == "missing_position" else results
        parquet = io.BytesIO()
        frame.to_parquet(parquet)
        fingerprint = {
            "sha256": "stale-inputs" if state.variant == "stale" else "live-inputs",
            "schema_version": core._PREDICTIONS_CACHE_SCHEMA_VERSION,
        }
        if state.variant == "old_schema":
            fingerprint["schema_version"] -= 1
        payload = {
            "metrics_by_format": metrics,
            "position_load_errors": {"QB": "old failure"} if state.variant == "degraded" else {},
        }
        files = {
            "predictions.parquet": b"broken" if state.variant == "corrupt" else parquet.getvalue(),
            "metrics.json": json.dumps(payload).encode(),
            "fingerprint.json": json.dumps(fingerprint).encode(),
        }
        generation = prediction_cache.publish_generation(cache_dir, files)
        if state.variant == "download_failed_local_valid":
            return {"files": 0, "failed": ["cache.tar.gz"]}
        return {"files": 3, "generation": generation.name}

    def compute():
        state.events.append("compute")
        assert os.environ["FF_MODEL_S3_BUCKET"] == ""
        assert not app._cache, "rebuild must discard any state restored during failed reuse"
        # Exercise the same cache-only reference reader used by build_cohorts.
        pd.testing.assert_frame_equal(evaluation_cohorts.load_reference(), reference)
        app._cache.update(
            results=results.copy(),
            metrics_by_format=metrics,
            positions_loaded=set(mod._ALL_POSITIONS),
            positions_mtime=dict.fromkeys(mod._ALL_POSITIONS, 0.0),
            position_load_errors=state.rebuild_errors,
            prediction_inputs_fingerprint="live-inputs",
        )
        core._persist_cache_to_disk()

    def upload():
        state.events.append("upload")
        assert os.environ["FF_MODEL_S3_BUCKET"] == "serving-cache-test"
        if state.fail_upload:
            return None
        state.uploaded.append(prediction_cache.read_generation(cache_dir)[1])
        return {"files": 3}

    monkeypatch.setattr(model_sync, "sync_predictions_cache_from_s3", sync_cache)
    monkeypatch.setattr(model_sync, "upload_predictions_cache_to_s3", upload)
    monkeypatch.setattr(core, "upload_predictions_cache_to_s3", lambda: None)
    monkeypatch.setattr(core, "_ensure_metrics", compute)
    monkeypatch.setattr(
        build_evaluation_reference,
        "write_reference",
        lambda *a, **kw: pytest.fail("cache builder must not regenerate the sealed reference"),
    )
    yield mod, state
    assert reference_path.read_bytes() == reference_bytes
    assert reference_path.stat().st_mtime_ns == reference_mtime


def test_reuse_valid_skips_inference_and_publication(cache_builder):
    mod, state = cache_builder
    assert mod.main(["--reuse-valid"]) == 0
    assert state.events == ["data", "models", "cache"]
    assert state.uploaded == []


@pytest.mark.parametrize(
    "variant",
    [
        "missing",
        "stale",
        "old_schema",
        "corrupt",
        "degraded",
        "missing_position",
        "download_failed_local_valid",
    ],
)
def test_unusable_s3_generation_rebuilds_before_success(cache_builder, variant):
    mod, state = cache_builder
    state.variant = variant
    assert mod.main(["--reuse-valid"]) == 0
    assert state.events == ["data", "models", "cache", "compute", "upload"]
    assert len(state.uploaded) == 1
    assert json.loads(state.uploaded[0]["fingerprint.json"])["sha256"] == "live-inputs"
    assert not json.loads(state.uploaded[0]["metrics.json"])["position_load_errors"]


def test_pending_hydrated_position_forces_rebuild(cache_builder, monkeypatch):
    from src.serving import app, core

    mod, state = cache_builder
    hydrate = core._try_hydrate_from_disk

    def hydrate_with_pending_position():
        result = hydrate()
        app._cache["positions_loaded"].discard("QB")
        return result

    monkeypatch.setattr(core, "_try_hydrate_from_disk", hydrate_with_pending_position)
    assert mod.main(["--reuse-valid"]) == 0
    assert state.events.count("compute") == 1
    assert len(state.uploaded) == 1


def test_default_still_forces_rebuild_without_syncing_old_cache(cache_builder):
    mod, state = cache_builder
    assert mod.main([]) == 0
    assert state.events == ["data", "models", "compute", "upload"]
    assert len(state.uploaded) == 1


def test_degraded_rebuild_cannot_pass_deployment_gate(cache_builder):
    mod, state = cache_builder
    state.variant = "missing"
    state.rebuild_errors = {"QB": "model unavailable"}
    assert mod.main(["--reuse-valid"]) == 1
    assert state.events == ["data", "models", "cache", "compute"]
    assert state.uploaded == []


def test_failed_upload_cannot_pass_deployment_gate(cache_builder):
    mod, state = cache_builder
    state.variant = "missing"
    state.fail_upload = True
    assert mod.main(["--reuse-valid"]) == 1
    assert state.events[-1] == "upload"
    assert state.uploaded == []


def test_deploy_cache_gate_is_required_before_ecs_or_alb_mutations():
    root = Path(__file__).resolve().parents[2]
    deploy = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    refresh = yaml.safe_load((root / ".github/workflows/refresh-splits.yml").read_text())
    steps = deploy["jobs"]["deploy"]["steps"]
    gate_indices = [
        index
        for index, step in enumerate(steps)
        if "src.scripts.build_serving_cache" in step.get("run", "")
    ]
    assert len(gate_indices) == 1
    gate_index = gate_indices[0]
    gate = steps[gate_index]
    assert gate["run"] == "python -m src.scripts.build_serving_cache --reuse-valid"
    assert not gate.get("continue-on-error", False)
    assert "if" not in gate
    assert gate["env"]["FF_DEVICE"] == "cpu"
    assert gate["env"]["FF_MODEL_S3_BUCKET"] == "${{ env.S3_BUCKET }}"
    assert gate["env"]["FF_MODEL_S3_PREFIX"] == "models"
    assert deploy["env"]["S3_BUCKET"] == refresh["env"]["S3_BUCKET"]
    assert deploy["env"]["UV_INDEX_STRATEGY"] == "unsafe-best-match"
    earlier = steps[:gate_index]
    assert any(step.get("uses", "").startswith("actions/setup-python@") for step in earlier)
    assert any(step.get("uses", "").startswith("astral-sh/setup-uv@") for step in earlier)
    assert any("-r requirements-dev.txt" in step.get("run", "") for step in earlier)
    assert (
        next(i for i, s in enumerate(steps) if s.get("name") == "Build, tag, push image")
        < gate_index
    )
    for name in (
        "Pull current task definition",
        "Tune ALB target group for fast rolling deploys",
        "Deploy to ECS",
    ):
        assert gate_index < next(i for i, s in enumerate(steps) if s.get("name") == name)
    deploy_step = next(step for step in steps if step.get("name") == "Deploy to ECS")
    assert "if" not in deploy_step, "deployment must not override the normal success dependency"
