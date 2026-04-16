"""Tests for shared.models — TwoStageRidge, OrdinalTDClassifier, GatedOrdinalTDClassifier, LightGBMMultiTarget."""

import numpy as np
import pandas as pd
import pytest

from shared.models import (
    TwoStageRidge,
    OrdinalTDClassifier,
    GatedOrdinalTDClassifier,
    LightGBMMultiTarget,
)

TARGETS = ["rushing_floor", "receiving_floor", "td_points"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_inflated_data():
    """Data where ~50% of y values are 0 (mimics TD points)."""
    np.random.seed(42)
    n, d = 40, 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.where(np.random.rand(n) > 0.5, np.random.rand(n) * 12, 0).astype(np.float32)
    return X, y


@pytest.fixture
def td_class_data():
    """Data with discrete TD-point-like values."""
    np.random.seed(42)
    n, d = 60, 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.choice([0.0, 6.0, 12.0, 18.0], size=n, p=[0.55, 0.25, 0.15, 0.05]).astype(np.float32)
    return X, y


@pytest.fixture
def multi_target_data():
    """Data for LightGBM multi-target tests."""
    np.random.seed(42)
    n, d = 80, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "rushing_floor": np.random.rand(n).astype(np.float32) * 10,
        "receiving_floor": np.random.rand(n).astype(np.float32) * 8,
        "td_points": np.random.rand(n).astype(np.float32) * 6,
    }
    return X, y_dict


# ---------------------------------------------------------------------------
# TwoStageRidge
# ---------------------------------------------------------------------------

class TestTwoStageRidge:
    def test_fit_and_predict_shapes(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_threshold_affects_predictions(self, zero_inflated_data):
        X, y = zero_inflated_data
        m_low = TwoStageRidge(threshold=0.1)
        m_high = TwoStageRidge(threshold=0.9)
        m_low.fit(X, y)
        m_high.fit(X, y)
        p_low = m_low.predict(X)
        p_high = m_high.predict(X)
        # High threshold produces more zeros
        assert (p_high == 0).sum() >= (p_low == 0).sum()

    def test_all_positive_samples_raises(self):
        """LogisticRegression requires 2 classes; all-positive y has only class 1."""
        np.random.seed(0)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.rand(20).astype(np.float32) * 10 + 1  # all > 0
        model = TwoStageRidge()
        with pytest.raises(ValueError, match="at least 2 classes"):
            model.fit(X, y)

    def test_save_and_load_roundtrip(self, zero_inflated_data, tmp_path):
        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "two_stage")
        model.save(model_dir)

        model2 = TwoStageRidge()
        model2.load(model_dir)
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_single_sample_prediction(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        preds = model.predict(X[:1])
        assert preds.shape == (1,)

    def test_high_threshold_mostly_zeros(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = TwoStageRidge(threshold=0.99)
        model.fit(X, y)
        preds = model.predict(X)
        # With very high threshold, most predictions should be zero
        assert (preds == 0).sum() > len(X) * 0.5


# ---------------------------------------------------------------------------
# OrdinalTDClassifier
# ---------------------------------------------------------------------------

class TestOrdinalTDClassifier:
    def test_fit_and_predict_shapes(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_class_point_values_computed(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        assert hasattr(model, "class_point_values_")
        assert len(model.class_point_values_) == model._n_classes

    def test_points_to_labels_fixed(self):
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        labels = model._points_to_labels(np.array([0, 6, 12, 18, 24]))
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3])  # 24 -> capped at 3

    def test_auto_class_values(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values="auto", n_classes=4)
        model.fit(X, y)
        assert hasattr(model, "class_point_values_")
        assert len(model.class_point_values_) >= 4

    def test_predict_proba_sums_to_one(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        proba = model._predict_proba(model.scaler_.transform(X))
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_and_load_roundtrip(self, td_class_data, tmp_path):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "ordinal_td")
        model.save(model_dir)

        model2 = OrdinalTDClassifier()
        model2.load(model_dir)
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_alpha_affects_predictions(self, td_class_data):
        X, y = td_class_data
        m1 = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4, alpha=0.01)
        m2 = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4, alpha=100.0)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# GatedOrdinalTDClassifier
# ---------------------------------------------------------------------------

class TestGatedOrdinalTDClassifier:
    def test_fit_and_predict_shapes(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_threshold_affects_predictions(self, td_class_data):
        X, y = td_class_data
        m_low = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4, threshold=0.1)
        m_high = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4, threshold=0.9)
        m_low.fit(X, y)
        m_high.fit(X, y)
        p_low = m_low.predict(X)
        p_high = m_high.predict(X)
        assert (p_high == 0).sum() >= (p_low == 0).sum()

    def test_save_and_load_roundtrip(self, td_class_data, tmp_path):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "gated_ordinal")
        model.save(model_dir)

        model2 = GatedOrdinalTDClassifier()
        model2.load(model_dir)
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)

    def test_save_writes_gated_flag(self, td_class_data, tmp_path):
        import json
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        model_dir = str(tmp_path / "gated_meta")
        model.save(model_dir)
        with open(f"{model_dir}/td_classifier_meta.json") as f:
            meta = json.load(f)
        assert meta["gated"] is True

    def test_single_sample_prediction(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X[:1])
        assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# LightGBMMultiTarget
# ---------------------------------------------------------------------------

class TestLightGBMMultiTarget:
    def test_fit_and_predict_shapes(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert set(preds.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_total_is_sum_of_components(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        expected = preds["rushing_floor"] + preds["receiving_floor"] + preds["td_points"]
        np.testing.assert_allclose(preds["total"], expected, atol=1e-6)

    def test_predictions_non_negative(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        for key in TARGETS:
            assert (preds[key] >= 0).all()

    def test_feature_importance_keys(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        model.fit(X, y_dict, feature_names=names)
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == set(TARGETS)
        for target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_with_validation_set(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=50)
        X_val = X[:20]
        y_val = {k: v[:20] for k, v in y_dict.items()}
        model.fit(X[20:], {k: v[20:] for k, v in y_dict.items()},
                  X_val=X_val, y_val_dict=y_val)
        preds = model.predict(X)
        assert preds["total"].shape == (len(X),)

    def test_without_validation_set(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert preds["total"].shape == (len(X),)

    def test_save_and_load_roundtrip(self, multi_target_data, tmp_path):
        X, y_dict = multi_target_data
        names = [f"feat_{i}" for i in range(X.shape[1])]
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict, feature_names=names)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "lgbm")
        model.save(model_dir)

        model2 = LightGBMMultiTarget(target_names=TARGETS)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_feature_names_stored(self, multi_target_data):
        X, y_dict = multi_target_data
        names = [f"feat_{i}" for i in range(X.shape[1])]
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict, feature_names=names)
        assert model._feature_names == names
