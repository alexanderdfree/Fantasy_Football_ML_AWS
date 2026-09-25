from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analysis.prediction_reuse import predict_reusing
from src.prediction.frames import PositionPredictions
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.registry import INFERENCE_REGISTRY

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_prediction_hit_rescores_raw_values_without_inference(position, tmp_path, monkeypatch):
    monkeypatch.setenv("FF_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FF_RESULT_CACHE_BUCKET", "")
    monkeypatch.delenv("FF_FRESH", raising=False)
    monkeypatch.setattr(
        "src.analysis.prediction_reuse.execution_identity", lambda device=None: {"device": "cpu"}
    )
    monkeypatch.setattr(
        "src.analysis.prediction_reuse.data_identity", lambda context: "immutable-data"
    )
    monkeypatch.setattr("src.prediction.bundle.bundled_families", lambda directory: ("ridge",))
    # The spy mutates its call counter; numerical code identity is covered
    # separately, so this storage/re-scoring fixture has a stable implementation.
    monkeypatch.setattr(
        "src.analysis.prediction_reuse.source_identity", lambda position: "prediction-code"
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "weights"
    model_file.write_text("generation-one")
    spec = {**INFERENCE_REGISTRY[position], "model_dir": str(model_dir)}
    raw = {target: np.array([1.0, 2.0, 3.0]) for target in spec["targets"]}
    frame = pd.DataFrame(raw)
    frame["fantasy_points"] = predictions_to_fantasy_points(position, raw)
    calls = []

    def predict(*args, **kwargs):
        calls.append(1)
        return PositionPredictions(
            frame.copy(),
            {"ridge": raw},
            {"ridge": {"ppr": np.zeros(3)}},
            {},
            {"n_features": 1},
            {"ridge": "bundle"},
        )

    monkeypatch.setattr("src.prediction.frames.predict_position", predict)
    first = predict_reusing(position, frame, frame, frame, spec)
    assert not first.frame.attrs["reuse"]["cache_hit"]
    second = predict_reusing(position, frame, frame, frame, spec)
    assert calls == [1]
    assert second.frame.attrs["reuse"]["cache_hit"]
    for scoring in ("ppr", "half_ppr", "standard"):
        np.testing.assert_array_equal(
            second.totals["ridge"][scoring], predictions_to_fantasy_points(position, raw, scoring)
        )
    model_file.write_text("generation-two")
    predict_reusing(position, frame, frame, frame, spec)
    predict_reusing(position, frame, frame, frame, spec, fresh=True)
    assert calls == [1, 1, 1]


def test_partial_predictions_are_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FF_RESULT_CACHE_BUCKET", "")
    monkeypatch.setattr(
        "src.analysis.prediction_reuse.execution_identity", lambda device=None: {"device": "cpu"}
    )
    monkeypatch.setattr("src.analysis.prediction_reuse.data_identity", lambda context: "data")
    calls = []
    frame = pd.DataFrame({"x": [1.0]})

    def predict(*args, **kwargs):
        calls.append(1)
        return PositionPredictions(frame.copy(), {}, {}, {"ridge": "missing"}, {}, {})

    monkeypatch.setattr("src.prediction.frames.predict_position", predict)
    spec = {"model_dir": str(tmp_path), "targets": ["x"]}
    for _ in range(2):
        result = predict_reusing("RB", frame, frame, frame, spec)
        assert result.errors
    assert calls == [1, 1]
