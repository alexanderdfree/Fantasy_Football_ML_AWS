"""Nonzero native/loaded parity and adversarial artifact-consumer boundaries."""

import os
import shutil
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.prediction import frames
from src.prediction.bundle import canonical_json, digest, write_bundle
from src.prediction.predictor import PredictionInputs, Predictor
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import wrap_state_dict, write_scaler_meta
from src.shared.feature_build import scale_and_clip
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)
from src.shared.registry import ALL_POSITIONS, get_inference_spec

pytestmark = pytest.mark.unit
FEATURES = ["feature_a", "feature_b"]


def _neural_artifact(directory, position, family, *, seed=33, data_id="training-data"):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = get_inference_spec(position)
    cfg["get_feature_columns_fn"] = lambda: FEATURES
    torch.manual_seed(seed)
    if family == "nn":
        model = MultiHeadNet(2, cfg["targets"], **cfg["nn_kwargs"])
    elif position == "K":
        model = MultiHeadNetWithNestedHistory(
            2, len(cfg["attn_kick_stats"]), cfg["targets"], **cfg["attn_nn_kwargs_static"]
        )
    else:
        model = MultiHeadNetWithHistory(
            2,
            len(cfg["attn_history_stats"]),
            cfg["targets"],
            opp_game_dim=len(cfg.get("opp_attn_history_stats", [])) or None,
            **cfg["attn_nn_kwargs_static"],
        )
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    stem = "attention_nn" if family == "attn_nn" else "nn"
    weight_stem = "attention_nn" if family == "attn_nn" else "multihead_nn"
    joblib.dump(scaler, directory / f"{stem}_scaler.pkl")
    write_scaler_meta(directory / f"{stem}_scaler_meta.json", FEATURES, cfg["targets"])
    torch.save(
        wrap_state_dict(model.state_dict(), FEATURES, cfg["targets"]),
        directory / f"{position.lower()}_{weight_stem}.pt",
    )
    bundle = write_bundle(
        directory,
        position,
        family,
        cfg,
        FEATURES,
        model,
        preprocessing={"clip": [-4.0, 4.0]},
        data_id=data_id,
    )
    return model, scaler, cfg, bundle


def _nonzero_inputs(predictor):
    template = predictor.zero_inputs()
    rng = np.random.default_rng(42)
    values = np.array([[-12.0, -2.0], [0.0, 0.0], [4.0, 8.0], [30.0, 60.0]], dtype=np.float32)
    changed = {"values": values}
    for field in ("history", "game_history", "opponent_history"):
        array = getattr(template, field)
        if array is not None:
            changed[field] = rng.uniform(0, 10, (4, *array.shape[1:])).astype(np.float32)
    for field in ("history_mask", "inner_mask", "opponent_mask"):
        mask = getattr(template, field)
        if mask is not None:
            changed[field] = np.zeros((4, *mask.shape[1:]), dtype=bool)
            changed[field][1:, :3] = True
            changed[field][..., -1] = False
    return replace(template, **changed)


def _native(model, scaler, family, inputs):
    x = scale_and_clip(scaler, inputs.values)
    device = torch.device("cpu")
    if family == "nn":
        return model.predict_numpy(x, device)
    if inputs.schema.structure == "nested":
        return model.predict_numpy(
            x,
            inputs.history,
            inputs.history_mask,
            inputs.inner_mask,
            device,
            X_game_history=inputs.game_history,
        )
    kwargs = (
        {"X_opp_history": inputs.opponent_history, "opp_history_mask": inputs.opponent_mask}
        if inputs.opponent_history is not None
        else {}
    )
    return model.predict_numpy(x, inputs.history, inputs.history_mask, device, **kwargs)


@pytest.mark.parametrize("position", ALL_POSITIONS)
@pytest.mark.parametrize("family", ["nn", "attn_nn"])
def test_nonzero_masked_histories_and_all_scoring_formats_match_native(position, family, tmp_path):
    model, scaler, cfg, _ = _neural_artifact(tmp_path, position, family)
    loaded = Predictor.from_bundle(tmp_path, family, position=position)
    inputs = _nonzero_inputs(loaded)
    expected = _native(model, scaler, family, inputs)
    actual = loaded.predict_raw(inputs)
    for target in cfg["targets"]:
        assert np.isfinite(actual[target]).all()
        np.testing.assert_array_equal(actual[target], expected[target])
    for scoring in ("ppr", "half_ppr", "standard"):
        np.testing.assert_array_equal(
            loaded.score(actual, scoring),
            predictions_to_fantasy_points(position, expected, scoring),
        )
    if family == "attn_nn":
        zeros = {
            field: np.zeros_like(getattr(inputs, field))
            for field in ("history", "game_history", "opponent_history")
            if getattr(inputs, field) is not None
        }
        without_history = _native(model, scaler, family, replace(inputs, **zeros))
        assert any(
            not np.array_equal(expected[target], without_history[target])
            for target in cfg["targets"]
        ), "Positive control: nonzero history must actually affect a head"


@pytest.mark.parametrize("position", ALL_POSITIONS)
@pytest.mark.parametrize("family", ["ridge", "lgbm"])
def test_fitted_cpu_models_roundtrip_nonzero_predictions(position, family, tmp_path):
    cfg = get_inference_spec(position)
    cfg["get_feature_columns_fn"] = lambda: FEATURES
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, (40, 2)).astype(np.float32)
    y = {
        target: (x[:, 0] + index * x[:, 1]).astype(np.float32)
        for index, target in enumerate(cfg["targets"], 1)
    }
    model = (
        RidgeMultiTarget(cfg["targets"], alpha=1.0)
        if family == "ridge"
        else LightGBMMultiTarget(cfg["targets"], n_estimators=4, min_child_samples=2, n_jobs=1)
    )
    model.fit(x, y)
    model.save(str(tmp_path))
    write_bundle(tmp_path, position, family, cfg, FEATURES, model, data_id="known-training")
    loaded = Predictor.from_bundle(tmp_path, family, position=position)
    expected = model.predict(x[:5])
    actual = loaded.predict_raw(PredictionInputs(loaded.schema, x[:5]))
    for target in cfg["targets"]:
        np.testing.assert_array_equal(actual[target], expected[target])
    for scoring in ("ppr", "half_ppr", "standard"):
        np.testing.assert_array_equal(
            loaded.score(actual, scoring),
            predictions_to_fantasy_points(position, expected, scoring),
        )


def test_unsupported_scoring_version_is_rejected(tmp_path):
    _, _, _, bundle = _neural_artifact(tmp_path, "QB", "nn")
    document = bundle.to_dict()
    document["scoring_version"] = 999
    document["bundle_id"] = digest(
        {key: value for key, value in document.items() if key != "bundle_id"}
    )
    (tmp_path / "nn.bundle.json").write_text(canonical_json(document))
    with pytest.raises(ValueError, match="scoring"):
        Predictor.from_bundle(tmp_path, "nn", position="QB")


def test_refresh_after_verification_cannot_load_other_generation_under_old_identity(
    tmp_path, monkeypatch
):
    live, candidate = tmp_path / "models", tmp_path / "candidate"
    old_model, scaler, _, old_bundle = _neural_artifact(live, "QB", "nn", seed=7, data_id="old")
    _neural_artifact(candidate, "QB", "nn", seed=11, data_id="new")
    load = torch.load
    swapped = False

    def refresh_before_deserialization(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(live, tmp_path / "previous")
            shutil.copytree(candidate, live)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", refresh_before_deserialization)
    try:
        loaded = Predictor.from_bundle(live, "nn", position="QB")
    except ValueError as error:
        assert swapped, "Positive control: refresh must occur before the rejection"
        assert any(word in str(error).lower() for word in ("generation", "artifact", "bundle"))
        return  # Explicit rejection/retry is as valid as loading a pinned old generation.
    assert swapped, "Positive control: refresh must occur before deserialization"
    assert loaded.bundle.bundle_id == old_bundle.bundle_id
    inputs = _nonzero_inputs(loaded)
    expected = _native(old_model, scaler, "nn", inputs)
    actual = loaded.predict_raw(inputs)
    for target in expected:
        np.testing.assert_array_equal(
            actual[target],
            expected[target],
            err_msg="Verified identity must describe the loaded bytes",
        )


@pytest.mark.parametrize("identity_field", ["data_id", "dataset_id"])
def test_families_with_different_declared_training_data_are_not_combined(
    tmp_path, monkeypatch, identity_field
):
    _, _, cfg, nn_bundle = _neural_artifact(tmp_path, "QB", "nn", data_id="same")
    cfg.update(model_dir=str(tmp_path), train_attention_nn=False, train_lightgbm=False)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    y = {target: np.arange(3, dtype=np.float32) for target in cfg["targets"]}
    ridge = RidgeMultiTarget(cfg["targets"], alpha=1.0)
    ridge.fit(x, y)
    ridge.save(str(tmp_path))
    ridge_bundle = write_bundle(
        tmp_path,
        "QB",
        "ridge",
        cfg,
        FEATURES,
        ridge,
        preprocessing={"clip": [-4.0, 4.0]},
        data_id="same",
    )
    for family, bundle, identity in (("ridge", ridge_bundle, "a"), ("nn", nn_bundle, "b")):
        document = bundle.to_dict()
        document["provenance"][identity_field] = identity
        document["bundle_id"] = digest(
            {key: value for key, value in document.items() if key != "bundle_id"}
        )
        (tmp_path / f"{family}.bundle.json").write_text(canonical_json(document))
    frame = pd.DataFrame(x, columns=FEATURES)
    # Isolate family selection; feature replay has independent test coverage.
    monkeypatch.setattr(
        frames, "prepare_position_frame", lambda *args, **kwargs: (frame, frame, frame, FEATURES)
    )
    result = frames.predict_position("QB", frame, frame, frame, cfg)
    assert "ridge" in result.raw
    assert "nn" not in result.raw
    assert "QB_nn" in result.errors


@pytest.mark.parametrize("position", ALL_POSITIONS)
@pytest.mark.parametrize("family", ["nn", "attn_nn"])
@pytest.mark.parametrize("trained_norm", ["batch", "layer"])
def test_bundle_normalization_is_independent_of_runtime_environment(
    tmp_path, monkeypatch, position, family, trained_norm
):
    monkeypatch.setenv("FF_NN_NORM", trained_norm)
    model, scaler, cfg, bundle = _neural_artifact(tmp_path, position, family)
    assert bundle.to_dict()["architecture"]["kwargs"]["backbone_norm"] == trained_norm
    runtime_norm = "layer" if trained_norm == "batch" else "batch"
    monkeypatch.setenv("FF_NN_NORM", runtime_norm)
    loaded = Predictor.from_bundle(tmp_path, family, position=position)
    assert loaded.model.backbone_norm == trained_norm
    assert os.environ["FF_NN_NORM"] == runtime_norm
    inputs = _nonzero_inputs(loaded)
    expected = _native(model, scaler, family, inputs)
    actual = loaded.predict_raw(inputs)
    for target in cfg["targets"]:
        np.testing.assert_array_equal(actual[target], expected[target])
