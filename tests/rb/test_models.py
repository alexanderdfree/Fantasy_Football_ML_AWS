"""Tests for shared.models.RidgeMultiTarget (using RB raw-stat targets)."""

import numpy as np
import pandas as pd
import pytest

from src.RB.rb_config import RB_GATED_ORDINAL_TARGETS, RB_ORDINAL_TARGETS, RB_TARGETS
from src.shared.models import (
    GatedOrdinalTDClassifier,
    OrdinalTDClassifier,
    RidgeMultiTarget,
)


@pytest.mark.unit
class TestRidgeMultiTarget:
    def test_fit_and_predict_shapes(self, simple_ridge_data):
        X, y_dict = simple_ridge_data
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)

        assert set(preds.keys()) == set(RB_TARGETS)
        for key in preds:
            assert preds[key].shape == (len(X),)

    def test_predict_total_matches(self, simple_ridge_data):
        X, y_dict = simple_ridge_data
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        total = model.predict_total(X)
        preds = model.predict(X)
        np.testing.assert_allclose(total, sum(preds[t] for t in RB_TARGETS), atol=1e-6)

    def test_different_alphas(self, simple_ridge_data):
        """Different regularization strengths produce different predictions."""
        X, y_dict = simple_ridge_data
        m1 = RidgeMultiTarget(target_names=RB_TARGETS, alpha=0.01)
        m2 = RidgeMultiTarget(target_names=RB_TARGETS, alpha=100.0)
        m1.fit(X, y_dict)
        m2.fit(X, y_dict)
        p1 = m1.predict_total(X)
        p2 = m2.predict_total(X)
        assert not np.allclose(p1, p2)

    def test_single_sample_prediction(self, simple_ridge_data):
        X, y_dict = simple_ridge_data
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        single = X[:1]
        preds = model.predict(single)
        for t in RB_TARGETS:
            assert preds[t].shape == (1,)

    def test_feature_importance_keys(self, simple_ridge_data):
        X, y_dict = simple_ridge_data
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        names = [f"feat_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(names)
        assert set(importance.keys()) == set(RB_TARGETS)
        for _target, series in importance.items():
            assert isinstance(series, pd.Series)
            assert len(series) == X.shape[1]

    def test_save_and_load(self, simple_ridge_data, tmp_path):
        X, y_dict = simple_ridge_data
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "rb_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_constant_target(self):
        """Model should handle constant target (zero variance)."""
        np.random.seed(0)
        X = np.random.randn(10, 3).astype(np.float32)
        y_dict = {
            "rushing_tds": np.full(10, 0.0),
            "receiving_tds": np.full(10, 0.0),
            "rushing_yards": np.full(10, 50.0),
            "receiving_yards": np.full(10, 30.0),
            "receptions": np.full(10, 2.0),
            "fumbles_lost": np.full(10, 0.0),
        }
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        model.fit(X, y_dict)
        preds = model.predict(X)
        assert np.allclose(preds["rushing_tds"], 0.0, atol=1.0)

    def test_per_target_alpha_construction(self, simple_ridge_data):
        """Per-target alphas produce different results than uniform alpha."""
        X, y_dict = simple_ridge_data
        alpha_dict = {t: 1.0 for t in RB_TARGETS}
        alpha_dict["rushing_yards"] = 0.01
        alpha_dict["receiving_yards"] = 100.0
        m_per = RidgeMultiTarget(target_names=RB_TARGETS, alpha=alpha_dict)
        m_uniform = RidgeMultiTarget(target_names=RB_TARGETS, alpha=1.0)
        m_per.fit(X, y_dict)
        m_uniform.fit(X, y_dict)
        p_per = m_per.predict(X)
        p_uniform = m_uniform.predict(X)
        any_different = any(not np.allclose(p_per[t], p_uniform[t]) for t in RB_TARGETS)
        assert any_different

    def test_per_target_alpha_save_load(self, simple_ridge_data, tmp_path):
        """Save/load round-trips correctly with per-target alphas."""
        X, y_dict = simple_ridge_data
        alpha_dict = {t: 1.0 for t in RB_TARGETS}
        alpha_dict["rushing_yards"] = 0.1
        alpha_dict["fumbles_lost"] = 500.0
        model = RidgeMultiTarget(target_names=RB_TARGETS, alpha=alpha_dict)
        model.fit(X, y_dict)
        preds_before = model.predict(X)

        model_dir = str(tmp_path / "per_target_model")
        model.save(model_dir)

        model2 = RidgeMultiTarget(target_names=RB_TARGETS)
        model2.load(model_dir)
        preds_after = model2.predict(X)

        for key in preds_before:
            np.testing.assert_allclose(preds_before[key], preds_after[key], atol=1e-6)

    def test_alpha_dict_missing_key_raises(self):
        """Missing target key in alpha dict raises ValueError."""
        partial = {t: 1.0 for t in RB_TARGETS[:2]}
        with pytest.raises(ValueError, match="missing keys"):
            RidgeMultiTarget(target_names=RB_TARGETS, alpha=partial)


# ---------------------------------------------------------------------------
# Ordinal / Gated-Ordinal classifiers on raw TD counts
# ---------------------------------------------------------------------------


def _make_td_data(n=200, mean_tds=0.6, seed=0):
    """Synthetic regression data where target is a Poisson-ish TD count."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    lam = np.clip(mean_tds + 0.3 * X[:, 0], 0.05, 3.0)
    y = rng.poisson(lam).astype(np.float32)
    return X, y


@pytest.mark.unit
class TestOrdinalTDClassifierRawCounts:
    def test_fit_predict_with_count_class_values(self):
        """class_values=[0,1,2,3] (raw TD counts) — predictions bounded."""
        X, y = _make_td_data()
        clf = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4, alpha=1.0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert preds.min() >= 0
        # Empirical cap: expected TD value across the dataset should stay
        # well below the upper class_value even with noisy fits.
        assert preds.max() <= 5.0

    def test_labels_match_count_step_1(self):
        """With step=1 class_values, y=2.1 rounds to label 2."""
        clf = OrdinalTDClassifier(class_values=[0, 1, 2, 3], n_classes=4)
        labels = clf._points_to_labels(np.array([0.0, 0.4, 1.0, 2.1, 3.8]))
        np.testing.assert_array_equal(labels, [0, 0, 1, 2, 3])

    def test_labels_backward_compatible_step_6(self):
        """Legacy config (step=6 points) still maps correctly for old models."""
        clf = OrdinalTDClassifier(class_values=[0, 6, 12, 18], n_classes=4)
        labels = clf._points_to_labels(np.array([0.0, 5.9, 6.0, 11.9, 18.0]))
        np.testing.assert_array_equal(labels, [0, 1, 1, 2, 3])

    def test_uses_config_from_rb_ordinal_targets(self):
        """Both rushing_tds and receiving_tds entries have class_values=[0,1,2,3]."""
        for target, cfg in RB_ORDINAL_TARGETS.items():
            assert cfg["class_values"] == [0, 1, 2, 3], (
                f"{target} ordinal class_values must be raw counts"
            )


@pytest.mark.unit
class TestGatedOrdinalRawCounts:
    def test_fit_predict_raw_counts(self):
        X, y = _make_td_data(n=300, seed=1)
        clf = GatedOrdinalTDClassifier(
            class_values=[0, 1, 2, 3], n_classes=4, alpha=1.0, clf_C=0.001
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert preds.min() >= 0

    def test_rb_gated_ordinal_targets_shape(self):
        """Both TD targets retargeted with class_values=[0,1,2,3]."""
        assert set(RB_GATED_ORDINAL_TARGETS) == {"rushing_tds", "receiving_tds"}
        for cfg in RB_GATED_ORDINAL_TARGETS.values():
            assert cfg["type"] == "gated_ordinal"
            assert cfg["class_values"] == [0, 1, 2, 3]


@pytest.mark.unit
class TestRidgeMultiTargetWithGatedOrdinal:
    """RidgeMultiTarget dispatches rushing_tds and receiving_tds to the gated
    ordinal classifier per ``RB_GATED_ORDINAL_TARGETS``."""

    def test_construction_with_two_gated_td_targets(self):
        model = RidgeMultiTarget(
            target_names=RB_TARGETS,
            alpha={t: 1.0 for t in RB_TARGETS},
            classification_targets=RB_GATED_ORDINAL_TARGETS,
        )
        # Both TD targets wrapped in GatedOrdinalTDClassifier.
        from src.shared.models import GatedOrdinalTDClassifier as _Gated

        assert isinstance(model._models["rushing_tds"], _Gated)
        assert isinstance(model._models["receiving_tds"], _Gated)
