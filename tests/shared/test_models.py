"""Tests for src.shared.models — TwoStageRidge, OrdinalTDClassifier, GatedOrdinalTDClassifier, LightGBMMultiTarget."""

import numpy as np
import pandas as pd
import pytest

from src.shared.models import (
    GatedOrdinalTDClassifier,
    LightGBMMultiTarget,
    OrdinalTDClassifier,
    RidgeModel,
    RidgeMultiTarget,
    TwoStageRidge,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zero_inflated_data():
    """Data where ~50% of y values are 0 (mimics TD counts)."""
    np.random.seed(42)
    n, d = 40, 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.where(np.random.rand(n) > 0.5, np.random.rand(n) * 3, 0).astype(np.float32)
    return X, y


@pytest.fixture
def td_class_data():
    """Data with discrete raw TD counts."""
    np.random.seed(42)
    n, d = 60, 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.choice([0.0, 1.0, 2.0, 3.0], size=n, p=[0.55, 0.25, 0.15, 0.05]).astype(
        np.float32
    )
    return X, y


@pytest.fixture
def multi_target_data():
    """Data for LightGBM multi-target tests."""
    np.random.seed(42)
    n, d = 80, 5
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "rushing_yards": np.random.rand(n).astype(np.float32) * 10,
        "receiving_yards": np.random.rand(n).astype(np.float32) * 8,
        "rushing_tds": np.random.rand(n).astype(np.float32) * 6,
    }
    return X, y_dict


# ---------------------------------------------------------------------------
# TwoStageRidge
# ---------------------------------------------------------------------------


@pytest.mark.unit
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

    def test_save_and_load_roundtrip_non_default_threshold(self, zero_inflated_data, tmp_path):
        """Constructor hyperparams survive a round trip (#1433).

        save() used to persist only the four fitted objects, so a non-default
        threshold silently reverted to the constructor default 0.5 on load.
        """
        X, y = zero_inflated_data
        model = TwoStageRidge(clf_C=0.01, ridge_alpha=0.1, threshold=0.9)
        model.fit(X, y)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "two_stage")
        model.save(model_dir)

        model2 = TwoStageRidge()  # default threshold=0.5, as the wrapper's load() constructs
        model2.load(model_dir)
        assert model2.threshold == 0.9
        assert model2.clf_C == 0.01
        assert model2.ridge_alpha == 0.1
        np.testing.assert_allclose(preds_before, model2.predict(X), atol=1e-6)

    def test_load_without_meta_sidecar_keeps_constructor_defaults(
        self, zero_inflated_data, tmp_path
    ):
        """Pre-sidecar artifacts (no two_stage_meta.json) load with prior behavior."""
        import os

        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        model_dir = str(tmp_path / "two_stage")
        model.save(model_dir)
        os.remove(f"{model_dir}/two_stage_meta.json")  # simulate an old artifact

        model2 = TwoStageRidge()
        model2.load(model_dir)
        assert model2.threshold == 0.5
        assert model2.clf_C == 0.001
        assert model2.ridge_alpha == 0.01

    def test_multi_target_wrapper_two_stage_threshold_roundtrip(self, zero_inflated_data, tmp_path):
        """The wrapper's load() reconstructs ``TwoStageRidge()`` with default
        args whenever classifier.pkl exists; the sidecar must restore the
        configured non-default threshold through that path too (#1433)."""
        X, y_zero_inflated = zero_inflated_data
        y_dict = {
            "rushing_yards": np.abs(X[:, 0] * 10 + 20).astype(np.float32),
            "rushing_tds": y_zero_inflated,
        }
        model = RidgeMultiTarget(
            target_names=["rushing_yards", "rushing_tds"],
            alpha=1.0,
            two_stage_targets={
                "rushing_tds": {"clf_C": 0.01, "ridge_alpha": 0.1, "threshold": 0.9}
            },
        )
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "multi")
        model.save(model_dir)

        restored = RidgeMultiTarget(target_names=["rushing_yards", "rushing_tds"], alpha=1.0)
        restored.load(model_dir)
        assert restored._models["rushing_tds"].threshold == 0.9
        np.testing.assert_allclose(
            preds_before["rushing_tds"], restored.predict(X)["rushing_tds"], atol=1e-6
        )

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

    def test_classifier_trained_on_all_rows_regressor_on_positives(self, zero_inflated_data):
        """Stage 1 (gate) sees every row; Stage 2 (regressor) only sees y > 0."""
        X, y = zero_inflated_data
        model = TwoStageRidge()
        model.fit(X, y)
        # Classifier sees all rows and distinguishes 2 classes (zero vs positive)
        assert model.clf.classes_.tolist() == [0, 1]
        # Classifier's scaler was fit on full X (mean vector shape matches feature count)
        assert model.scaler_clf.mean_.shape == (X.shape[1],)
        # Regressor's scaler was fit only on the positive subset — its mean
        # must match the mean of the positive rows (not the full dataset).
        pos_mask = y > 0
        np.testing.assert_allclose(
            model.scaler_reg.mean_,
            X[pos_mask].mean(axis=0),
            atol=1e-6,
        )
        # The two scalers are therefore distinct objects with distinct means.
        assert model.scaler_clf is not model.scaler_reg


# ---------------------------------------------------------------------------
# OrdinalTDClassifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrdinalTDClassifier:
    def test_fit_and_predict_shapes(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_class_point_values_computed(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        assert hasattr(model, "class_point_values_")
        assert len(model.class_point_values_) == model._n_classes

    def test_points_to_labels_fixed(self):
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        labels = model._points_to_labels(np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3])  # 4 -> capped at 3

    def test_auto_class_values(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values="auto", n_classes=4)
        model.fit(X, y)
        assert hasattr(model, "class_point_values_")
        assert len(model.class_point_values_) >= 4

    def test_predict_proba_sums_to_one(self, td_class_data):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        proba = model._predict_proba(model.scaler_.transform(X))
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_and_load_roundtrip(self, td_class_data, tmp_path):
        X, y = td_class_data
        model = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
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
        m1 = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4, alpha=0.01)
        m2 = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4, alpha=100.0)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# GatedOrdinalTDClassifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatedOrdinalTDClassifier:
    def test_fit_and_predict_shapes(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_non_negative(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_threshold_affects_predictions(self, td_class_data):
        X, y = td_class_data
        m_low = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4, threshold=0.1)
        m_high = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4, threshold=0.9)
        m_low.fit(X, y)
        m_high.fit(X, y)
        p_low = m_low.predict(X)
        p_high = m_high.predict(X)
        assert (p_high == 0).sum() >= (p_low == 0).sum()

    def test_save_and_load_roundtrip(self, td_class_data, tmp_path):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
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
        model = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        model_dir = str(tmp_path / "gated_meta")
        model.save(model_dir)
        with open(f"{model_dir}/td_classifier_meta.json") as f:
            meta = json.load(f)
        assert meta["gated"] is True

    def test_single_sample_prediction(self, td_class_data):
        X, y = td_class_data
        model = GatedOrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        model.fit(X, y)
        preds = model.predict(X[:1])
        assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# LightGBMMultiTarget
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLightGBMMultiTarget:
    def test_fit_and_predict_shapes(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert set(preds.keys()) == {"rushing_yards", "receiving_yards", "rushing_tds"}
        for key in preds:
            assert preds[key].shape == (len(X),)

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
        for _target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_with_validation_set(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=50)
        X_val = X[:20]
        y_val = {k: v[:20] for k, v in y_dict.items()}
        model.fit(X[20:], {k: v[20:] for k, v in y_dict.items()}, X_val=X_val, y_val_dict=y_val)
        preds = model.predict(X)
        for t in TARGETS:
            assert preds[t].shape == (len(X),)

    def test_without_validation_set(self, multi_target_data):
        X, y_dict = multi_target_data
        model = LightGBMMultiTarget(target_names=TARGETS, n_estimators=10)
        model.fit(X, y_dict)
        preds = model.predict(X)
        for t in TARGETS:
            assert preds[t].shape == (len(X),)

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

    def test_predict_falls_back_to_instance_non_negative_targets(self, multi_target_data):
        """predict() with no kwarg uses ``self.non_negative_targets`` (audit #393).

        The constructor default still clamps every head; an explicit subset must
        be honored at predict-time even when the caller (serving) omits the
        kwarg. Force a target negative so the clamp is observable.
        """
        X, y_dict = multi_target_data
        # Train so every head learns a strongly-negative constant for one target.
        y_neg = dict(y_dict)
        y_neg["rushing_tds"] = np.full(len(X), -5.0, dtype=np.float32)
        model = LightGBMMultiTarget(
            target_names=TARGETS,
            n_estimators=10,
            non_negative_targets={"rushing_yards", "receiving_yards"},  # excludes rushing_tds
        )
        model.fit(X, y_neg)
        preds = model.predict(X)  # no kwarg -> falls back to instance set
        # Excluded head keeps its negative learned value (not clamped to 0).
        assert (preds["rushing_tds"] < 0).any()
        # Clamped heads stay >= 0.
        assert (preds["rushing_yards"] >= 0).all()

    def test_non_negative_targets_survive_save_load(self, multi_target_data, tmp_path):
        """The per-head clamp set round-trips through save/load (audit #393).

        ``meta.json`` previously omitted ``non_negative_targets``, so a reload
        constructed with the default and reverted to clamp-every-head. The loaded
        model (default constructor — the serving pattern) must honor the subset.
        """
        X, y_dict = multi_target_data
        subset = {"rushing_yards"}
        model = LightGBMMultiTarget(
            target_names=TARGETS, n_estimators=10, non_negative_targets=subset
        )
        model.fit(X, y_dict)
        model_dir = str(tmp_path / "lgbm")
        model.save(model_dir)

        loaded = LightGBMMultiTarget(target_names=TARGETS)  # default = clamp-all
        loaded.load(model_dir)
        assert loaded.non_negative_targets == subset


@pytest.mark.unit
class TestRidgeModelFloat64:
    """RidgeModel runs the StandardScaler -> PCA -> Ridge path in float64
    end-to-end, even though the incoming splits are float32 — no
    float32->float64->float32 round-trip across the PCA SVD."""

    @staticmethod
    def _data(n=120, d=12):
        rng = np.random.RandomState(0)
        X = rng.randn(n, d).astype(np.float32)
        y = (X @ rng.randn(d) + 0.5 * rng.randn(n)).astype(np.float32)
        return X, y

    def test_scaler_fit_in_float64_no_pca(self):
        X, y = self._data()
        m = RidgeModel(alpha=1.0)
        m.fit(X, y)
        # StandardScaler stores stats in its input dtype; float64 here proves
        # the entry cast took effect (a float32 X would give float32 stats).
        assert m.scaler.mean_.dtype == np.float64
        assert m.pca is None

    def test_pca_path_float64_and_predicts_on_float32(self):
        X, y = self._data()
        m = RidgeModel(alpha=1.0, pca_n_components=6)
        m.fit(X, y)
        assert m.scaler.mean_.dtype == np.float64
        assert m.pca is not None
        # predict() must accept the float32 splits and run float64 internally.
        preds = m.predict(X)
        assert np.isfinite(preds).all()
        assert preds.shape == (len(X),)
        # Sanity: the linear signal is recovered (positive correlation).
        assert np.corrcoef(preds, y)[0, 1] > 0.5

    def test_predict_dtype_invariant(self):
        """float32 vs float64 input to predict() give the same result (both are
        upcast to float64 at entry)."""
        X, y = self._data()
        m = RidgeModel(alpha=1.0, pca_n_components=6)
        m.fit(X, y)
        p32 = m.predict(X)
        p64 = m.predict(X.astype(np.float64))
        np.testing.assert_array_equal(p32, p64)

    def test_multitarget_propagates_float64(self):
        X, _ = self._data()
        y_dict = {t: (X[:, i] * 3.0).astype(np.float32) for i, t in enumerate(TARGETS)}
        m = RidgeMultiTarget(target_names=TARGETS, alpha=1.0, pca_n_components=6)
        m.fit(X, y_dict)
        # Each per-target RidgeModel inherits the float64 path.
        for t in TARGETS:
            assert m._models[t].scaler.mean_.dtype == np.float64
        preds = m.predict(X)
        for t in TARGETS:
            assert np.isfinite(preds[t]).all()
