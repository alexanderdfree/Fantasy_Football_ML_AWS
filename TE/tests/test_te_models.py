"""Tests for shared.models.RidgeMultiTarget (using TE targets)."""

import numpy as np
import pandas as pd
import pytest

from shared.models import RidgeMultiTarget
from TE.te_config import TE_TARGETS

pytestmark = pytest.mark.unit


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "receiving_tds": np.random.randint(0, 3, n).astype(np.float32),
        "receiving_yards": np.random.rand(n).astype(np.float32) * 80,
        "receptions": np.random.rand(n).astype(np.float32) * 8,
        "fumbles_lost": np.random.binomial(1, 0.03, n).astype(np.float32),
    }
    return X, y_dict


class TestRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == set(TE_TARGETS) | {"total"}
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_total_is_sum_of_components(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        expected = sum(preds[t] for t in TE_TARGETS)
        np.testing.assert_allclose(preds["total"], expected, atol=1e-6)

    def test_predict_total_matches(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        np.testing.assert_allclose(total, preds["total"], atol=1e-6)

    def test_different_alphas(self, simple_data):
        X, y_dict = simple_data
        m1 = RidgeMultiTarget(target_names=TE_TARGETS, alpha=0.01)
        m2 = RidgeMultiTarget(target_names=TE_TARGETS, alpha=100.0)
        m1.fit(X, y_dict)
        m2.fit(X, y_dict)
        assert not np.allclose(m1.predict_total(X), m2.predict_total(X))

    def test_single_sample_prediction(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X[:1])
        assert preds["total"].shape == (1,)

    def test_feature_importance_keys(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == set(TE_TARGETS)
        for _target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_save_and_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "te_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_constant_target(self):
        np.random.seed(0)
        X = np.random.randn(10, 3).astype(np.float32)
        y_dict = {
            "receiving_tds": np.full(10, 1.0),
            "receiving_yards": np.full(10, 60.0),
            "receptions": np.full(10, 5.0),
            "fumbles_lost": np.full(10, 0.0),
        }
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert np.allclose(preds["fumbles_lost"], 0.0, atol=1.0)

    def test_per_target_alpha_construction(self, simple_data):
        X, y_dict = simple_data
        alpha_dict = {
            "receiving_tds": 100.0,
            "receiving_yards": 0.01,
            "receptions": 1.0,
            "fumbles_lost": 100.0,
        }
        m_per = RidgeMultiTarget(target_names=TE_TARGETS, alpha=alpha_dict)
        m_uniform = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        m_per.fit(X, y_dict)
        m_uniform.fit(X, y_dict)
        p_per = m_per.predict(X)
        p_uniform = m_uniform.predict(X)
        any_different = any(not np.allclose(p_per[t], p_uniform[t]) for t in TE_TARGETS)
        assert any_different

    def test_per_target_alpha_save_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        alpha_dict = {
            "receiving_tds": 500.0,
            "receiving_yards": 0.1,
            "receptions": 10.0,
            "fumbles_lost": 500.0,
        }
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=alpha_dict)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "per_target_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=TE_TARGETS)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_alpha_dict_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            RidgeMultiTarget(
                target_names=TE_TARGETS,
                alpha={"receiving_tds": 1.0, "receiving_yards": 1.0},
            )
