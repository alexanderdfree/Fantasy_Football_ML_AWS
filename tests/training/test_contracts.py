"""Executable recipe and execution-boundary contracts."""

from __future__ import annotations

import importlib
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from src.shared.position_pipeline import PipelineConfigError
from src.shared.registry import get_config
from src.training.context import RunContext, current_context, raw_data_dir, use_context
from src.training.contracts import (
    DatasetSplits,
    PreparedDataset,
    ResolvedRecipe,
    TrainingResult,
    resolve_recipe,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_every_production_recipe_resolves_without_changing_values(position):
    cfg = get_config(position)
    recipe = resolve_recipe(position, cfg, require_runtime=False)
    assert isinstance(recipe, ResolvedRecipe)
    assert list(recipe.model.targets) == cfg["targets"]
    assert recipe["get_feature_columns_fn"]() == cfg["get_feature_columns_fn"]()
    for key in ("nn_lr", "nn_backbone_layers", "loss_weights", "huber_deltas"):
        assert recipe[key] == cfg[key]


def test_recipe_detaches_nested_containers_and_resolved_feature_order():
    cfg = dict(get_config("RB"))
    cfg["loss_weights"] = dict(cfg["loss_weights"])
    columns = ["a", "b"]
    cfg["get_feature_columns_fn"] = lambda: list(columns)
    recipe = resolve_recipe("RB", cfg)
    cfg["loss_weights"][cfg["targets"][0]] = 999
    columns.reverse()
    recipe["loss_weights"][cfg["targets"][0]] = 111
    assert recipe["loss_weights"][cfg["targets"][0]] not in {999, 111}
    assert recipe["get_feature_columns_fn"]() == ["a", "b"]
    with pytest.raises(FrozenInstanceError):
        recipe.features.columns = ("b", "a")


def test_unknown_override_rejected_and_extension_namespace_supported():
    cfg = dict(get_config("RB"))
    with pytest.raises(PipelineConfigError, match="Unknown pipeline options"):
        resolve_recipe("RB", {**cfg, "nn_lrr": 0.01})
    recipe = resolve_recipe("RB", {**cfg, "experimental_options": {"trial_tag": "control"}})
    assert recipe["trial_tag"] == "control"
    assert pickle.loads(pickle.dumps(recipe))["trial_tag"] == "control"
    with pytest.raises(PipelineConfigError, match="cannot shadow"):
        resolve_recipe("RB", {**cfg, "experimental_options": {"nn_lr": 0.01}})


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda cfg: cfg.update(targets=[]), "nonempty"),
        (lambda cfg: cfg.update(loss_weights={}), "target coverage"),
        (lambda cfg: cfg.update(attn_history_stats=["a", "a"]), "Duplicate"),
        (lambda cfg: cfg.update(attn_history_structure="wrong"), "Unknown history"),
        (lambda cfg: cfg.update(nn_non_negative_targets={"wrong"}), "unknown targets"),
    ],
)
def test_invalid_recipes_fail_before_training(mutate, match):
    cfg = dict(get_config("RB"))
    mutate(cfg)
    with pytest.raises(PipelineConfigError, match=match):
        resolve_recipe("RB", cfg)


def test_prepared_dataset_and_result_preserve_legacy_access():
    frame = pd.DataFrame({"a": [1.0]})
    frame.attrs["preprocessing_state"] = {"fill": {"a": 0.0}}
    matrix = frame.to_numpy()
    prepared = PreparedDataset(
        matrix, matrix, matrix, {}, {}, {}, frame, frame, frame, ("a",), "snapshot"
    )
    assert len(prepared) == 10
    assert prepared[-1] == ["a"]
    assert prepared.preprocessing == frame.attrs["preprocessing_state"]
    result = TrainingResult(
        {"per_target_preds": {"nn": {"a": np.ones(1)}}, "nn_metrics": {"mae": 1}},
        resolve_recipe("RB", get_config("RB")),
        prepared,
        {},
        "run",
    )
    assert dict(result)["nn_metrics"] == {"mae": 1}
    assert result.metrics == {"nn_metrics": {"mae": 1}}
    restored = pickle.loads(pickle.dumps(result))
    assert restored.prepared.data_id == "snapshot"
    np.testing.assert_array_equal(restored.predictions["nn"]["a"], np.ones(1))


def test_context_nesting_does_not_mutate_cwd_or_loader_configuration(tmp_path):
    import os

    original = os.getcwd()
    first = RunContext(tmp_path / "one", tmp_path / "inputs-one")
    second = RunContext(tmp_path / "two", tmp_path / "inputs-two")
    with use_context(first):
        assert raw_data_dir("unchanged") == str(first.raw_root)
        with use_context(second):
            assert raw_data_dir("unchanged") == str(second.raw_root)
        assert current_context() is first
    assert current_context() is None
    assert raw_data_dir("unchanged") == "unchanged"
    assert os.getcwd() == original


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
@pytest.mark.parametrize("cross_validation", [False, True])
def test_all_position_adapters_forward_explicit_context(
    tmp_path, monkeypatch, position, cross_validation
):
    module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    context = RunContext(tmp_path / "outputs", tmp_path / "inputs", seed=7)
    frame = pd.DataFrame({"a": [1.0]})
    if position in {"K", "DST"}:
        monkeypatch.setattr(
            module,
            "provide_dataset",
            lambda *args, **kwargs: DatasetSplits(frame, frame, frame, {}),
        )
    seen = {}

    def pipeline(*args, **kwargs):
        seen.update(kwargs)
        assert current_context() is context
        return "done"

    monkeypatch.setattr(module, "run_cv_pipeline" if cross_validation else "run_pipeline", pipeline)
    assert getattr(module, "run_cv" if cross_validation else "run")(context=context) == "done"
    assert seen["context"] is context


def test_preparation_retries_snapshot_changes_and_bounds_instability(monkeypatch):
    from types import SimpleNamespace

    from src.shared import pipeline

    identifiers = iter(["old", "current"])
    monkeypatch.setattr(pipeline.feature_cache, "cache_key", lambda *args: "current")
    monkeypatch.setattr(
        pipeline.feature_cache,
        "load_or_compute",
        lambda *args: SimpleNamespace(data_id=next(identifiers)),
    )
    assert pipeline._prepare_position_data("RB", {}, None, None).data_id == "current"
    calls = []

    def unstable(*args):
        calls.append(1)
        return SimpleNamespace(data_id="old")

    monkeypatch.setattr(pipeline.feature_cache, "load_or_compute", unstable)
    with pytest.raises(RuntimeError, match="changed repeatedly"):
        pipeline._prepare_position_data("RB", {}, None, None)
    assert len(calls) == 3


def test_disabled_artifact_sink_does_not_run_effect_or_claim_bundle(tmp_path):
    context = RunContext(tmp_path / "outputs", tmp_path / "data", artifact_sink=None)
    assert context.emit_artifacts(lambda: pytest.fail("artifact write")) is None
    assert not context.output_root.exists()
