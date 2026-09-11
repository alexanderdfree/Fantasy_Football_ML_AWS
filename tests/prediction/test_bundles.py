from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.prediction.bundle import InputSchema, read_bundle, write_bundle
from src.prediction.predictor import PredictionInputs, Predictor
from src.shared.artifact_integrity import wrap_state_dict, write_scaler_meta
from src.shared.feature_build import (
    build_position_features,
    fill_nans_with_train_means,
    scale_and_clip,
)
from src.shared.models import RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)
from src.shared.registry import ALL_POSITIONS, get_inference_spec

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", ALL_POSITIONS)
@pytest.mark.parametrize("family", ["nn", "attn_nn"])
def test_six_position_bundle_roundtrip_owns_constructor_and_order(position, family, tmp_path):
    reg = get_inference_spec(position)
    features = ["feature_a", "feature_b"]
    reg["get_feature_columns_fn"] = lambda: features
    targets = reg["targets"]
    torch.manual_seed(9)
    if family == "nn":
        model = MultiHeadNet(input_dim=2, target_names=targets, **reg["nn_kwargs"])
    elif position == "K":
        model = MultiHeadNetWithNestedHistory(
            static_dim=2,
            kick_dim=len(reg["attn_kick_stats"]),
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        )
    else:
        model = MultiHeadNetWithHistory(
            static_dim=2,
            game_dim=len(reg["attn_history_stats"]),
            opp_game_dim=len(reg.get("opp_attn_history_stats", [])) or None,
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        )
    stem = "attention_nn" if family == "attn_nn" else "nn"
    weight_stem = "attention_nn" if family == "attn_nn" else "multihead_nn"
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    joblib.dump(scaler, tmp_path / f"{stem}_scaler.pkl")
    write_scaler_meta(tmp_path / f"{stem}_scaler_meta.json", features, targets)
    torch.save(
        wrap_state_dict(model.state_dict(), features, targets),
        tmp_path / f"{position.lower()}_{weight_stem}.pt",
    )
    bundle = write_bundle(
        tmp_path,
        position,
        family,
        reg,
        features,
        model,
        preprocessing={"clip": [-4.0, 4.0]},
        data_id="fixed-data",
    )
    loaded = Predictor.from_bundle(tmp_path, family, position=position)
    inputs = loaded.zero_inputs()
    # Compare identical raw inputs and trained state through the old native API.
    scaled = scale_and_clip(scaler, inputs.values)
    if family == "nn":
        expected = model.predict_numpy(scaled, torch.device("cpu"))
    elif position == "K":
        expected = model.predict_numpy(
            scaled,
            inputs.history,
            inputs.history_mask,
            inputs.inner_mask,
            torch.device("cpu"),
            X_game_history=inputs.game_history,
        )
    else:
        kwargs = (
            {"X_opp_history": inputs.opponent_history, "opp_history_mask": inputs.opponent_mask}
            if inputs.schema.opponent_history
            else {}
        )
        expected = model.predict_numpy(
            scaled, inputs.history, inputs.history_mask, torch.device("cpu"), **kwargs
        )
    actual = loaded.predict_raw(inputs)
    for target in targets:
        np.testing.assert_array_equal(actual[target], expected[target])
    assert loaded.bundle.bundle_id == bundle.bundle_id
    assert read_bundle(tmp_path, family).inputs.features == tuple(features)
    if family == "attn_nn":
        fields = "kicks" if position == "K" else "history"
        order = getattr(inputs.schema, fields)
        assert len(order) >= 2
        changed = replace(inputs.schema, **{fields: (order[1], order[0], *order[2:])})
        with pytest.raises(ValueError, match="ordered input schema"):
            loaded.predict_raw(replace(inputs, schema=changed))


@pytest.mark.parametrize("with_selection", [False, True])
def test_ridge_bundle_binds_files_and_feature_schema(tmp_path, with_selection):
    reg = get_inference_spec("QB")
    features = ["a", "b"]
    reg["get_feature_columns_fn"] = lambda: features
    x = np.array([[1.0, 2.0], [3.0, 1.0], [4.0, 6.0], [2.0, 3.0]], dtype=np.float32)
    model = RidgeMultiTarget(reg["targets"], alpha=1.0)
    model.fit(x, {target: np.arange(4, dtype=np.float32) for target in reg["targets"]})
    model.save(str(tmp_path))
    selection = tmp_path / "ridge_selection.json"
    if with_selection:
        selection.write_text('{"metric": "mean_cv_fantasy_rmse_ppr", "score": 1.25}')
    bundle = write_bundle(tmp_path, "QB", "ridge", reg, features, model, data_id="fixed")
    with bundle.pinned_directory(tmp_path) as pinned:
        assert (pinned / selection.name).exists() == with_selection
        if with_selection:
            assert (pinned / selection.name).read_bytes() == selection.read_bytes()
    loaded = Predictor.from_bundle(tmp_path, "ridge", position="QB")
    actual = loaded.predict_raw(PredictionInputs(loaded.schema, x))
    for target, expected in model.predict(x).items():
        np.testing.assert_array_equal(actual[target], expected)
    if with_selection:
        selection.write_text('{"metric": "mean_cv_fantasy_rmse_ppr", "score": 99.0}')
        with pytest.raises(ValueError, match="artifact mismatch.*ridge_selection"):
            Predictor.from_bundle(tmp_path, "ridge", position="QB")
        return
    weights = tmp_path / reg["targets"][0] / "ridge_model.pkl"
    weights.write_bytes(weights.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="artifact mismatch"):
        Predictor.from_bundle(tmp_path, "ridge", position="QB")


def test_saved_imputation_state_does_not_refit_on_changed_history(monkeypatch):
    monkeypatch.setattr("src.shared.feature_build.merge_schedule_features", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.shared.feature_build.merge_team_box_score_features", lambda *a, **k: None
    )
    cfg = {
        "add_features_fn": lambda a, b, c, **kw: (a, b, c),
        "fill_nans_fn": fill_nans_with_train_means,
        "specific_features": ["a"],
    }
    features = ["a", "depth_chart_rank"]
    train = pd.DataFrame({"a": [2.0, 4.0], "depth_chart_rank": [1.0, -1.0]})
    query = pd.DataFrame({"a": [np.nan], "depth_chart_rank": [-1.0]})
    fitted, _, original = build_position_features(
        train.copy(), query.copy(), query.copy(), cfg, features
    )
    state = fitted.attrs["preprocessing_state"]
    changed_train = train.copy()
    changed_train["a"] *= 100
    changed_train["depth_chart_rank"] = 8.0
    _, _, replay = build_position_features(
        changed_train, query.copy(), query.copy(), cfg, features, fitted_state=state
    )
    pd.testing.assert_frame_equal(original, replay)
    assert replay["a"].iloc[0] == 3.0


def test_input_schema_rejects_duplicate_names():
    with pytest.raises(ValueError, match="ordered features"):
        InputSchema(("a", "a"), ("target",))
