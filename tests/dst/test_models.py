"""Tests for src.shared.models.RidgeMultiTarget (using a DST target subset).

Uses 3 of the 10 DST raw-stat targets (def_sacks, def_tds, points_allowed)
as representative test data — enough to exercise multi-target ridge
behavior without pulling the full config.
"""

import numpy as np
import pandas as pd
import pytest

from src.shared.models import RidgeMultiTarget

DST_TARGETS = ["def_sacks", "def_tds", "points_allowed"]


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "def_sacks": np.random.rand(n).astype(np.float32) * 5,
        "def_tds": np.random.rand(n).astype(np.float32) * 1,
        "points_allowed": np.random.rand(n).astype(np.float32) * 40,
    }
    return X, y_dict


@pytest.mark.unit
class TestRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == {"def_sacks", "def_tds", "points_allowed"}
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_predict_total_matches(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        expected = preds["def_sacks"] + preds["def_tds"] + preds["points_allowed"]
        np.testing.assert_allclose(total, expected, atol=1e-6)

    def test_different_alphas(self, simple_data):
        X, y_dict = simple_data
        m1 = RidgeMultiTarget(target_names=DST_TARGETS, alpha=0.01)
        m2 = RidgeMultiTarget(target_names=DST_TARGETS, alpha=100.0)
        m1.fit(X, y_dict)
        m2.fit(X, y_dict)
        assert not np.allclose(m1.predict_total(X), m2.predict_total(X))

    def test_single_sample_prediction(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X[:1])
        for t in DST_TARGETS:
            assert preds[t].shape == (1,)

    def test_feature_importance_keys(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == {"def_sacks", "def_tds", "points_allowed"}
        for _target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_save_and_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "dst_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_constant_target(self):
        np.random.seed(0)
        X = np.random.randn(10, 3).astype(np.float32)
        y_dict = {
            "def_sacks": np.full(10, 7.0),
            "def_tds": np.full(10, 0.0),
            "points_allowed": np.full(10, 1.0),
        }
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert np.allclose(preds["def_tds"], 0.0, atol=1.0)

    def test_per_target_alpha_construction(self, simple_data):
        X, y_dict = simple_data
        alpha_dict = {"def_sacks": 0.01, "def_tds": 1.0, "points_allowed": 100.0}
        m_per = RidgeMultiTarget(target_names=DST_TARGETS, alpha=alpha_dict)
        m_uniform = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        m_per.fit(X, y_dict)
        m_uniform.fit(X, y_dict)
        p_per = m_per.predict(X)
        p_uniform = m_uniform.predict(X)
        any_different = any(not np.allclose(p_per[t], p_uniform[t]) for t in DST_TARGETS)
        assert any_different

    def test_per_target_alpha_save_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        alpha_dict = {"def_sacks": 0.1, "def_tds": 10.0, "points_allowed": 500.0}
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=alpha_dict)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "per_target_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=DST_TARGETS)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_alpha_dict_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing keys"):
            RidgeMultiTarget(
                target_names=DST_TARGETS,
                alpha={"def_sacks": 1.0, "def_tds": 1.0},
            )
