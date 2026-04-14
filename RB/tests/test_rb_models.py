"""Tests for RB.rb_models — RBRidgeMultiTarget."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from RB.rb_models import RBRidgeMultiTarget


@pytest.fixture
def simple_data():
    """Small training dataset with 20 samples and 5 features."""
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "rushing_floor": np.random.rand(n).astype(np.float32) * 10,
        "receiving_floor": np.random.rand(n).astype(np.float32) * 8,
        "td_points": np.random.rand(n).astype(np.float32) * 6,
    }
    return X, y_dict


class TestRBRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_data):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_total_is_sum_of_components(self, simple_data):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        expected_total = preds["rushing_floor"] + preds["receiving_floor"] + preds["td_points"]
        np.testing.assert_allclose(preds["total"], expected_total, atol=1e-6)

    def test_predict_total_matches(self, simple_data):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        np.testing.assert_allclose(total, preds["total"], atol=1e-6)

    def test_different_alphas(self, simple_data):
        """Different regularization strengths produce different predictions."""
        X, y_dict = simple_data
        m1 = RBRidgeMultiTarget(alpha=0.01)
        m2 = RBRidgeMultiTarget(alpha=100.0)
        m1.fit(X, y_dict)
        m2.fit(X, y_dict)
        p1 = m1.predict_total(X)
        p2 = m2.predict_total(X)
        # Strong regularization should shrink coefficients toward 0
        assert not np.allclose(p1, p2)

    def test_single_sample_prediction(self, simple_data):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        single = X[:1]
        preds = model.predict(single)
        assert preds["total"].shape == (1,)

    def test_feature_importance_keys(self, simple_data):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == {"rushing_floor", "receiving_floor", "td_points"}
        for target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_save_and_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "rb_model")
        model.save(model_dir)

        model2 = RBRidgeMultiTarget(alpha=1.0)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_constant_target(self):
        """Model should handle constant target (zero variance)."""
        np.random.seed(0)
        X = np.random.randn(10, 3).astype(np.float32)
        y_dict = {
            "rushing_floor": np.full(10, 5.0),
            "receiving_floor": np.full(10, 3.0),
            "td_points": np.full(10, 0.0),
        }
        model = RBRidgeMultiTarget(alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        # Should predict close to the constant values
        assert np.allclose(preds["td_points"], 0.0, atol=1.0)
