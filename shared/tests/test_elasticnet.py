"""Tests for ElasticNetModel and ElasticNetMultiTarget."""

import json

import numpy as np
import pytest

from shared.models import ElasticNetMultiTarget
from src.models.elastic_net import ElasticNetModel

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]


@pytest.fixture
def linear_data():
    np.random.seed(42)
    n, d = 120, 8
    X = np.random.randn(n, d).astype(np.float32)
    # Linear signal on a few features + noise
    y = (2.0 * X[:, 0] - 1.5 * X[:, 2] + 0.5 * np.random.randn(n)).astype(np.float32)
    return X, y


@pytest.fixture
def multi_target_data():
    np.random.seed(42)
    n, d = 80, 6
    X = np.random.randn(n, d).astype(np.float32)
    y_dict = {
        "rushing_yards": (X[:, 0] * 5 + np.random.randn(n)).astype(np.float32),
        "receiving_yards": (X[:, 1] * 3 + np.random.randn(n)).astype(np.float32),
        "rushing_tds": np.abs(X[:, 2] + np.random.randn(n) * 0.3).astype(np.float32),
    }
    return X, y_dict


@pytest.mark.unit
class TestElasticNetModel:
    def test_fit_and_predict_shapes(self, linear_data):
        X, y = linear_data
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_convergence_flag_set(self, linear_data):
        X, y = linear_data
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        model.fit(X, y)
        # On well-conditioned data this should converge cleanly.
        assert model.converged is True
        assert model.n_iter > 0

    def test_non_convergence_flagged(self, linear_data):
        X, y = linear_data
        # max_iter=1 forces a non-convergence warning on all but trivial data.
        model = ElasticNetModel(alpha=0.001, l1_ratio=0.7, max_iter=1, tol=1e-12)
        model.fit(X, y)
        assert model.converged is False

    def test_l1_sparsity(self, linear_data):
        X, y = linear_data
        # Strong L1 should zero out some coefficients.
        model = ElasticNetModel(alpha=1.0, l1_ratio=0.9)
        model.fit(X, y)
        n_zeros = int(np.sum(np.abs(model.model.coef_) < 1e-10))
        assert n_zeros > 0

    def test_save_and_load_roundtrip(self, linear_data, tmp_path):
        X, y = linear_data
        model = ElasticNetModel(alpha=0.5, l1_ratio=0.3)
        model.fit(X, y)
        preds = model.predict(X)

        model.save(str(tmp_path))
        # meta.json carries the CV-picked hyperparameters + convergence info
        # so a reviewer can tell whether the solver actually finished.
        assert (tmp_path / "meta.json").exists()
        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["alpha"] == pytest.approx(0.5)
        assert meta["l1_ratio"] == pytest.approx(0.3)
        assert "converged" in meta
        assert "n_iter" in meta

        loaded = ElasticNetModel()
        loaded.load(str(tmp_path))
        np.testing.assert_allclose(loaded.predict(X), preds, rtol=1e-6)
        assert loaded.alpha == pytest.approx(0.5)
        assert loaded.l1_ratio == pytest.approx(0.3)

    def test_feature_importance_returns_series(self, linear_data):
        X, y = linear_data
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)
        model.fit(X, y)
        names = [f"f{i}" for i in range(X.shape[1])]
        imp = model.get_feature_importance(names)
        assert len(imp) == X.shape[1]
        # Highest-importance feature should be one of the true drivers (f0 or f2)
        top = imp.index[0]
        assert top in {"f0", "f2"}


@pytest.mark.unit
class TestElasticNetMultiTarget:
    def test_fit_and_predict_shapes(self, multi_target_data):
        X, y_dict = multi_target_data
        model = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.1 for t in TARGETS},
            l1_ratio={t: 0.5 for t in TARGETS},
        )
        model.fit(X, y_dict)
        preds = model.predict(X)
        for t in TARGETS:
            assert preds[t].shape == y_dict[t].shape

    def test_predict_total_sums_heads(self, multi_target_data):
        X, y_dict = multi_target_data
        model = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.1 for t in TARGETS},
            l1_ratio={t: 0.5 for t in TARGETS},
        )
        model.fit(X, y_dict)
        preds = model.predict(X)
        total = model.predict_total(X)
        np.testing.assert_allclose(total, sum(preds[t] for t in TARGETS), rtol=1e-6)

    def test_non_negative_clamping_default(self, multi_target_data):
        X, y_dict = multi_target_data
        model = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.1 for t in TARGETS},
            l1_ratio={t: 0.5 for t in TARGETS},
        )
        model.fit(X, y_dict)
        preds = model.predict(X)
        for t in TARGETS:
            assert (preds[t] >= 0).all()

    def test_convergence_report(self, multi_target_data):
        X, y_dict = multi_target_data
        model = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.1 for t in TARGETS},
            l1_ratio={t: 0.5 for t in TARGETS},
        )
        model.fit(X, y_dict)
        report = model.convergence_report()
        assert set(report.keys()) == set(TARGETS)
        for info in report.values():
            assert "converged" in info
            assert "n_iter" in info

    def test_save_and_load_roundtrip(self, multi_target_data, tmp_path):
        X, y_dict = multi_target_data
        model = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.3 for t in TARGETS},
            l1_ratio={t: 0.4 for t in TARGETS},
        )
        model.fit(X, y_dict)
        preds = model.predict(X)

        model.save(str(tmp_path))
        loaded = ElasticNetMultiTarget(
            target_names=TARGETS,
            alpha={t: 0.0 for t in TARGETS},  # will be overwritten by load
            l1_ratio={t: 0.0 for t in TARGETS},
        )
        loaded.load(str(tmp_path))
        loaded_preds = loaded.predict(X)
        for t in TARGETS:
            np.testing.assert_allclose(loaded_preds[t], preds[t], rtol=1e-6)

    def test_missing_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha dict missing keys"):
            ElasticNetMultiTarget(
                target_names=TARGETS,
                alpha={"rushing_yards": 0.1},  # missing keys
                l1_ratio={t: 0.5 for t in TARGETS},
            )

    def test_missing_l1_ratio_raises(self):
        with pytest.raises(ValueError, match="l1_ratio dict missing keys"):
            ElasticNetMultiTarget(
                target_names=TARGETS,
                alpha={t: 0.1 for t in TARGETS},
                l1_ratio={"rushing_yards": 0.5},
            )
