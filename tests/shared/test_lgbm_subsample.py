"""The configured LightGBM row fraction must actually affect fitted trees."""

import json

import numpy as np
import pytest

from src.shared.models import LightGBMMultiTarget

pytestmark = pytest.mark.unit


def _fit(fraction, *, frequency=None):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(160, 5))
    y = {"yards": np.maximum(30 + 10 * x[:, 0] + 8 * x[:, 1] + rng.normal(size=160), 0)}
    model = LightGBMMultiTarget(
        ["yards"],
        n_estimators=12,
        num_leaves=7,
        min_child_samples=3,
        objective="regression",
        subsample=fraction,
        n_jobs=1,
        seed=31,
    )
    if frequency is not None:
        model._models["yards"].set_params(subsample_freq=frequency)
    model.fit(x, y)
    return model, x, model.predict(x)["yards"]


def test_configured_partial_fraction_matches_enabled_bagging_and_changes_predictions():
    configured, _, predictions = _fit(0.5)
    _, _, enabled = _fit(0.5, frequency=1)
    _, _, disabled = _fit(0.5, frequency=0)
    assert configured._params["subsample_freq"] == 1
    np.testing.assert_array_equal(predictions, enabled)
    assert not np.array_equal(predictions, disabled)
    assert np.max(np.abs(predictions - disabled)) > 1


def test_full_data_fraction_preserves_legacy_predictions():
    model, _, predictions = _fit(1.0)
    _, _, legacy = _fit(1.0, frequency=0)
    assert model._params["subsample_freq"] == 0
    np.testing.assert_array_equal(predictions, legacy)


@pytest.mark.parametrize("fraction,frequency", [(0.5, 1), (1.0, 0)])
def test_saved_metadata_and_loaded_model_retain_effective_fraction(tmp_path, fraction, frequency):
    model, x, expected = _fit(fraction)
    model.save(tmp_path)
    meta = json.loads((tmp_path / "lightgbm" / "meta.json").read_text())
    assert meta["params"]["subsample"] == fraction
    assert meta["params"]["subsample_freq"] == frequency
    loaded = LightGBMMultiTarget(["yards"], n_jobs=1)
    loaded.load(tmp_path)
    assert loaded._models["yards"].get_params()["subsample_freq"] == frequency
    assert loaded._models["yards"].get_params()["subsample"] == fraction
    np.testing.assert_array_equal(loaded.predict(x)["yards"], expected)
