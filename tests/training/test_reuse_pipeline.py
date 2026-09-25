"""Real loader/pipeline round trips, using the existing tiny E2E harness."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.pipeline import run_pipeline
from src.shared.registry import get_runner
from src.training.context import RunContext
from tests._pipeline_e2e_utils import ALL_POSITIONS, build_tiny_config, load_tiny_splits
from tests._skip_helpers import require_splits

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("position", ALL_POSITIONS)
def test_real_pipeline_cache_hit_preserves_predictions_models_and_metrics(
    position, tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    require_splits(root / "data/splits")
    monkeypatch.setenv("FF_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FF_RESULT_CACHE_BUCKET", "")
    monkeypatch.delenv("FF_FRESH", raising=False)
    cfg = build_tiny_config(position)
    splits = load_tiny_splits(position)
    context = RunContext(tmp_path / "first", root / "data", reuse_results=True, report_sink=None)
    from src.training.reuse_identity import source_manifest

    before = source_manifest(position)

    def execute(run_context):
        if os.environ.get("FF_REUSE_PRODUCTION") == "1":
            return get_runner(position)(seed=42, context=run_context)
        return run_pipeline(position, cfg, *splits, context=run_context)

    first = execute(context)
    after = source_manifest(position)
    changed_code = {
        kind: [
            key
            for key in before[kind].keys() | after[kind].keys()
            if before[kind].get(key) != after[kind].get(key)
        ]
        for kind in before
    }
    second = execute(replace(context, output_root=tmp_path / "second", run_id="second"))
    assert second["reuse"]["cache_hit"], (second["reuse"], changed_code)
    assert second["reuse"]["reused_from"] == first.run_id
    assert set(second.models) == set(first.models)
    for family, targets in first.predictions.items():
        for target, predictions in targets.items():
            np.testing.assert_array_equal(second.predictions[family][target], predictions)
        assert second[f"{family}_metrics"] == first[f"{family}_metrics"]
    assert second["cohorts"] == first["cohorts"]
    original = context.output_dir(position) / "models"
    replay = tmp_path / "second" / position.lower() / "outputs/models"
    for path in original.rglob("*"):
        if path.is_file():
            assert (replay / path.relative_to(original)).read_bytes() == path.read_bytes()


@pytest.fixture
def rb_reuse_setup(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    require_splits(root / "data/splits")
    monkeypatch.setenv("FF_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FF_RESULT_CACHE_BUCKET", "")
    monkeypatch.delenv("FF_FRESH", raising=False)
    return (
        build_tiny_config("RB"),
        load_tiny_splits("RB"),
        RunContext(tmp_path / "outputs", root / "data", reuse_results=True, report_sink=None),
    )


def test_replay_preserves_elasticnet_public_columns(rb_reuse_setup):
    from src.evaluation.metrics import pred_columns_from_test_df

    cfg, splits, context = rb_reuse_setup
    cfg.update(train_elasticnet=True, enet_l1_ratios=[0.5])
    first = run_pipeline("RB", cfg, *splits, context=context)
    second = run_pipeline("RB", cfg, *splits, context=replace(context, run_id="second"))
    assert second["reuse"]["cache_hit"]
    assert "ElasticNet" in pred_columns_from_test_df(first["test_df"])
    assert "ElasticNet" in pred_columns_from_test_df(second["test_df"])
    for target in ["total", *cfg["targets"]]:
        np.testing.assert_array_equal(
            first["test_df"][f"pred_enet_{target}"], second["test_df"][f"pred_enet_{target}"]
        )


@pytest.mark.parametrize("sink", [None, lambda effect: None])
def test_disabled_artifact_sink_cannot_publish_preexisting_models(rb_reuse_setup, sink):
    cfg, splits, context = rb_reuse_setup
    run_pipeline("RB", cfg, *splits, context=replace(context, reuse_results=False, seed=42))
    disabled = replace(context, artifact_sink=sink, seed=7, run_id="seed-seven")
    first = run_pipeline("RB", cfg, *splits, context=disabled)
    second = run_pipeline("RB", cfg, *splits, context=disabled)
    assert not first["reuse"]["cache_hit"]
    assert not second["reuse"]["cache_hit"]
