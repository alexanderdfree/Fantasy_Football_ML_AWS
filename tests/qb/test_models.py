"""Tests for src.shared.models.RidgeMultiTarget (using QB raw-stat targets)."""

import numpy as np
import pandas as pd
import pytest

from src.qb.config import QB_TARGETS
from src.shared.models import RidgeMultiTarget


@pytest.fixture
def simple_data():
    """Small training dataset with 20 samples and 5 features."""
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    # Scale each target to plausible per-game ranges.
    scales = {
        "passing_yards": 350.0,
        "rushing_yards": 40.0,
        "passing_tds": 3.0,
        "rushing_tds": 1.0,
        "interceptions": 2.0,
        "fumbles_lost": 1.0,
    }
    y_dict = {t: np.random.rand(n).astype(np.float32) * scales[t] for t in QB_TARGETS}
    return X, y_dict


@pytest.mark.unit
class TestRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == set(QB_TARGETS)
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_predict_total_matches(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        np.testing.assert_allclose(total, sum(preds[t] for t in QB_TARGETS), atol=1e-6)

    def test_different_alphas(self, simple_data):
        """Different regularization strengths produce different predictions."""
        X, y_dict = simple_data
        m1 = RidgeMultiTarget(target_names=QB_TARGETS, alpha=0.01)
        m2 = RidgeMultiTarget(target_names=QB_TARGETS, alpha=100.0)
        m1.fit(X, y_dict)
        m2.fit(X, y_dict)
        p1 = m1.predict_total(X)
        p2 = m2.predict_total(X)
        assert not np.allclose(p1, p2)

    def test_single_sample_prediction(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        single = X[:1]
        preds = model.predict(single)
        for t in QB_TARGETS:
            assert preds[t].shape == (1,)

    def test_feature_importance_keys(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == set(QB_TARGETS)
        for _target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_save_and_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "qb_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_constant_target(self):
        """Model should handle constant target (zero variance)."""
        np.random.seed(0)
        X = np.random.randn(10, 3).astype(np.float32)
        y_dict = {
            "passing_yards": np.full(10, 250.0),
            "rushing_yards": np.full(10, 20.0),
            "passing_tds": np.full(10, 2.0),
            "rushing_tds": np.full(10, 0.0),
            "interceptions": np.full(10, 1.0),
            "fumbles_lost": np.full(10, 0.0),
        }
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert np.allclose(preds["rushing_tds"], 0.0, atol=1.0)

    def test_per_target_alpha_construction(self, simple_data):
        """Per-target alphas produce different results than uniform alpha."""
        X, y_dict = simple_data
        alpha_dict = {
            "passing_yards": 0.01,
            "rushing_yards": 1.0,
            "passing_tds": 100.0,
            "rushing_tds": 10.0,
            "interceptions": 50.0,
            "fumbles_lost": 5.0,
        }
        m_per = RidgeMultiTarget(target_names=QB_TARGETS, alpha=alpha_dict)
        m_uniform = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        m_per.fit(X, y_dict)
        m_uniform.fit(X, y_dict)
        p_per = m_per.predict(X)
        p_uniform = m_uniform.predict(X)
        any_different = any(not np.allclose(p_per[t], p_uniform[t]) for t in QB_TARGETS)
        assert any_different

    def test_per_target_alpha_save_load(self, simple_data, tmp_path):
        """Save/load round-trips correctly with per-target alphas."""
        X, y_dict = simple_data
        alpha_dict = {
            "passing_yards": 0.1,
            "rushing_yards": 10.0,
            "passing_tds": 500.0,
            "rushing_tds": 50.0,
            "interceptions": 200.0,
            "fumbles_lost": 20.0,
        }
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=alpha_dict)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "per_target_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=QB_TARGETS)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_alpha_dict_missing_key_raises(self):
        """Missing target key in alpha dict raises ValueError."""
        with pytest.raises(ValueError, match="missing keys"):
            RidgeMultiTarget(
                target_names=QB_TARGETS,
                alpha={"passing_yards": 1.0, "rushing_yards": 1.0},
            )
