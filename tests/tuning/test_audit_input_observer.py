"""No fitting: input capture, exact passthrough and independent drift controls."""

import copy
import importlib
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.tuning import audit_input_observer as observer

pytestmark = pytest.mark.unit
CHANGED = {"opportunity_index_L3", "redzone_target_share_L3"}


def _prepared():
    columns = ["untouched", *sorted(CHANGED)]
    frame = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "season": [2022, 2022],
            "week": [1, 2],
            "untouched": [1.0, 2.0],
            "opportunity_index_L3": [3.0, 4.0],
            "redzone_target_share_L3": [0.25, 0.5],
            "target": [0.0, 1.0],
        }
    )
    result = SimpleNamespace(
        feature_columns=columns,
        preprocessing={
            "fill_values": {c: np.float32(0) for c in columns},
            "clip": [-5, 5],
            "dtype": "float32",
        },
    )
    for split in ("train", "val", "test"):
        setattr(result, split, frame.copy())
        setattr(result, f"X_{split}", frame[columns].to_numpy(dtype=np.float32, copy=True))
        setattr(result, f"y_{split}", {"target": frame.target.to_numpy(dtype=np.float32)})
    return result


def test_prepared_proof_detects_intended_columns_and_keeps_order():
    prepared = _prepared()
    baseline = observer.prepared_proof(prepared)
    for split in ("train", "val", "test"):
        for index, column in enumerate(prepared.feature_columns):
            if column in CHANGED:
                getattr(prepared, split).loc[0, column] += 1
                getattr(prepared, f"X_{split}")[0, index] += 1
    changed = observer.prepared_proof(prepared)
    assert changed["feature_columns"] == prepared.feature_columns
    assert changed["target_columns"] == ["target"]
    assert changed["preprocessing"] == baseline["preprocessing"]
    for split in ("train", "val", "test"):
        a, b = baseline["prepared"][split], changed["prepared"][split]
        assert a["row_keys_hash"] == b["row_keys_hash"]
        assert a["y"] == b["y"]
        for field in ("frame_column_hashes", "X"):
            left, right = a[field], b[field]
            if field == "X":
                left, right = left["columns"], right["columns"]
            assert {c for c in left if left[c] != right[c]} == CHANGED


def test_prepared_proof_detects_order_dtype_and_excluded_prediction_columns():
    prepared = _prepared()
    baseline = observer.prepared_proof(prepared)
    prepared.train["pred_nn_total"] = [9.0, 8.0]
    assert observer.prepared_proof(prepared) == baseline
    prepared.train = prepared.train.iloc[::-1]
    assert (
        observer.prepared_proof(prepared)["prepared"]["train"]["row_keys_hash"]
        != (baseline["prepared"]["train"]["row_keys_hash"])
    )
    assert observer.array_proof(np.ones(2, dtype=np.float32)) != observer.array_proof(
        np.ones(2, dtype=np.float64)
    )
    with pytest.raises(ValueError, match="columns"):
        observer.array_proof(np.ones((2, 3)), ["a"])


@pytest.mark.parametrize("position", ["WR", "TE"])
def test_native_feature_routes_exclude_stint_windows_from_attention(position):
    from src.shared.pipeline import get_attn_static_columns

    cfg = importlib.import_module(f"src.{position.lower()}.run_pipeline").CONFIG
    columns = cfg["get_feature_columns_fn"]()
    static = get_attn_static_columns(columns, cfg["attn_static_features"])
    assert set(columns) >= CHANGED
    assert not CHANGED.intersection(static)
    assert not CHANGED.intersection(cfg["attn_history_stats"])
    assert {"game_opportunity_index", "redzone_target_share"} <= set(cfg["attn_history_stats"])


@pytest.mark.parametrize("family", ["nn", "attn_nn"])
@pytest.mark.parametrize("resident", [False, True])
def test_loader_storage_capture_preserves_rng_and_never_iterates(family, resident):
    from torch.utils.data import DataLoader

    from src.shared.training import (
        MultiTargetDataset,
        MultiTargetHistoryDataset,
        _GPUResidentBatcher,
    )

    static = np.arange(6, dtype=np.float32).reshape(3, 2)
    history = np.ones((3, 2, 1), dtype=np.float32)
    mask = np.ones((3, 2), dtype=bool)
    targets = {"target": np.arange(3, dtype=np.float32)}
    if family == "nn":
        dataset = MultiTargetDataset(static, targets)
        features = (dataset.X,)
    else:
        dataset = MultiTargetHistoryDataset(static, history, mask, targets)
        features = (dataset.X_static, dataset.X_history, dataset.history_mask)
    loader = (
        _GPUResidentBatcher(features, dataset.targets, 2, True, True)
        if resident
        else DataLoader(dataset, batch_size=2, shuffle=True)
    )
    record = {
        "family": family,
        "feature_columns": ["a", "b"],
        "history_columns": ["raw"],
        "opp_history_columns": [],
        "target_columns": ["target"],
        "actual": {},
    }
    before = torch.get_rng_state().clone()
    token = observer._ACTIVE.set(record)
    try:
        observer.observe_loaders(loader, loader)
    finally:
        observer._ACTIVE.reset(token)
    assert torch.equal(torch.get_rng_state(), before)
    assert record["actual"]["train"]["inputs"]["static"] == observer.array_proof(static, ["a", "b"])
    if family == "attn_nn":
        assert record["actual"]["val"]["inputs"]["history"] == observer.array_proof(
            history, ["raw"]
        )
        assert record["actual"]["val"]["inputs"]["mask"] == observer.array_proof(mask)


@pytest.mark.parametrize("position", ["WR", "TE"])
def test_real_pipeline_boundaries_without_fitting_preserve_outputs_rng_and_replay(
    monkeypatch, position
):
    """Execute native helper wiring, replacing fit/scale with fixed no-fit stubs."""
    from src.shared import neural_net, pipeline

    cfg = dict(importlib.import_module(f"src.{position.lower()}.run_pipeline").CONFIG)
    cfg.update(
        nn_backbone_layers=[4], nn_head_hidden=2, nn_dropout=0.0, attn_d_model=8, attn_n_heads=2
    )
    features = cfg["get_feature_columns_fn"]()
    targets = cfg["targets"]
    x = np.arange(4 * len(features), dtype=np.float32).reshape(4, -1) / 100
    y = {target: np.arange(4, dtype=np.float32) for target in targets}
    history = np.ones((4, 2, len(cfg["attn_history_stats"])), dtype=np.float32)
    mask = np.ones((4, 2), dtype=bool)

    def no_fit_scale(*arrays, cfg=None, feature_cols=None):
        size = len(feature_cols)
        scaler = SimpleNamespace(
            mean_=np.zeros(size), var_=np.ones(size), scale_=np.ones(size), n_samples_seen_=4
        )
        return scaler, [array.copy() for array in arrays]

    def no_fit_training(**kwargs):
        observer.observe_loaders(kwargs["train_loader"], kwargs["val_loader"])
        return {"no_fit": True}

    monkeypatch.setattr(pipeline, "_scale_xs", no_fit_scale)
    monkeypatch.setattr(pipeline, "_run_nn_training", no_fit_training)
    monkeypatch.setattr(pipeline, "_nn_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "_maybe_compile", lambda model: model)
    # Register restoration before the installer replaces these attributes.
    for owner, name in [
        (pipeline, "_train_nn"),
        (pipeline, "_train_attention_nn"),
        (neural_net.MultiHeadNet, "predict_numpy"),
        (neural_net.MultiHeadNetWithHistory, "predict_numpy"),
    ]:
        original = getattr(owner, name)
        monkeypatch.setattr(owner, name, getattr(original, "_audit_input_original", original))

    def calls():
        plain = pipeline._train_nn(x, x, x, y, y, y, cfg, targets, 42, features)
        attn = pipeline._train_attention_nn(
            x,
            x,
            x,
            history,
            mask,
            history,
            mask,
            history,
            mask,
            y,
            y,
            y,
            cfg,
            targets,
            42,
            features,
        )
        return plain, attn

    expected = calls()
    expected_rng = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    state = {"position": position}
    observer.install(state)
    observer.install(state)  # repeated configure must not double-wrap captures
    actual = calls()
    assert torch.equal(torch.get_rng_state(), expected_rng[0])
    np.testing.assert_array_equal(np.random.get_state()[1], expected_rng[1][1])
    assert random.getstate() == expected_rng[2]
    for baseline, observed in zip(expected, actual, strict=True):
        for target in targets:
            np.testing.assert_array_equal(baseline[2][target], observed[2][target])
    for family in ("nn", "attn_nn"):
        record = state["neural_inputs"][family]
        assert set(record["actual"]) == {"train", "val", "test"}
        assert record["target_columns"] == targets
        for split in ("train", "val", "test"):
            assert record["actual"][split]["inputs"]["static"] == record["scaled"][split]
        for field in ("mean_", "var_", "scale_"):
            assert list(record["scaler"][field]["columns"]) == record["feature_columns"]
    captured = copy.deepcopy(state["neural_inputs"])
    actual[0][0].predict_numpy(x + 10, torch.device("cpu"))
    static_columns = state["neural_inputs"]["attn_nn"]["feature_columns"]
    actual[1][0].predict_numpy(x[:, : len(static_columns)] + 10, history, mask, torch.device("cpu"))
    assert state["neural_inputs"] == captured


def test_missing_observations_fail_closed():
    with pytest.raises(ValueError, match="Both WR/TE"):
        observer.result_proof(_prepared(), {})
    assert observer._ACTIVE.get() is None
