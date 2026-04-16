"""Tests for shared.models.RidgeMultiTarget (using DST targets).

DST has 3 targets: defensive_scoring, td_points, pts_allowed_bonus.
Note: pts_allowed_bonus can be negative (range [-4, +10]).
"""

import numpy as np
import pandas as pd
import pytest

from shared.models import RidgeMultiTarget

DST_TARGETS = ["defensive_scoring", "td_points", "pts_allowed_bonus"]


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "defensive_scoring": np.random.rand(n).astype(np.float32) * 10,
        "td_points": np.random.rand(n).astype(np.float32) * 6,
        # pts_allowed_bonus can be negative
        "pts_allowed_bonus": (np.random.rand(n).astype(np.float32) * 14) - 4,
    }
    return X, y_dict


@pytest.mark.unit
class TestRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == {"defensive_scoring", "td_points", "pts_allowed_bonus", "total"}
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_total_is_sum_of_components(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        expected = preds["defensive_scoring"] + preds["td_points"] + preds["pts_allowed_bonus"]
        np.testing.assert_allclose(preds["total"], expected, atol=1e-6)

    def test_predict_total_matches(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        np.testing.assert_allclose(total, preds["total"], atol=1e-6)

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
        assert preds["total"].shape == (1,)

    def test_feature_importance_keys(self, simple_data):
        X, y_dict = simple_data
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == {"defensive_scoring", "td_points", "pts_allowed_bonus"}
        for target, series in importance.items():
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
            "defensive_scoring": np.full(10, 7.0),
            "td_points": np.full(10, 0.0),
            "pts_allowed_bonus": np.full(10, 1.0),
        }
        model = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert np.allclose(preds["td_points"], 0.0, atol=1.0)

    def test_per_target_alpha_construction(self, simple_data):
        X, y_dict = simple_data
        alpha_dict = {"defensive_scoring": 0.01, "td_points": 1.0, "pts_allowed_bonus": 100.0}
        m_per = RidgeMultiTarget(target_names=DST_TARGETS, alpha=alpha_dict)
        m_uniform = RidgeMultiTarget(target_names=DST_TARGETS, alpha=1.0)
        m_per.fit(X, y_dict)
        m_uniform.fit(X, y_dict)
        p_per = m_per.predict(X)
        p_uniform = m_uniform.predict(X)
        any_different = any(
            not np.allclose(p_per[t], p_uniform[t]) for t in DST_TARGETS
        )
        assert any_different

    def test_per_target_alpha_save_load(self, simple_data, tmp_path):
        X, y_dict = simple_data
        alpha_dict = {"defensive_scoring": 0.1, "td_points": 10.0, "pts_allowed_bonus": 500.0}
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
                alpha={"defensive_scoring": 1.0, "td_points": 1.0},
            )
