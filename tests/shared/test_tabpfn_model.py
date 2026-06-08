"""Tests for src.shared.models.TabPFNMultiTarget — the opt-in (default-off) 5th
model variant.

TabPFN's weight-download dependency (``tabpfn``) is optional and intentionally
NOT installed in CI: no production position enables it, so the package would only
bloat the image. The wrapper logic (per-target loop, non-negative clamp, optional
PCA reduction, save/load round-trip) is therefore exercised with a lightweight
fake regressor injected over the lazy ``_new_regressor`` import; a single
end-to-end test runs only when ``tabpfn`` is importable.
"""

import numpy as np
import pytest

from src.shared.models import TabPFNMultiTarget

TARGETS = ["receiving_yards", "receptions", "fumbles_lost"]


class _FakeRegressor:
    """Stand-in for ``TabPFNRegressor``: predicts a per-target constant.

    Module-level (so the save/load round-trip can ``joblib.dump`` it). ``fill``
    forces a fixed output (incl. negatives) to make the clamp assertions
    deterministic; otherwise it learns the train mean at ``fit``.
    """

    def __init__(self, fill=None):
        self.fill = fill
        self._value = 0.0

    def fit(self, X, y):
        self._value = float(np.mean(y)) if self.fill is None else self.fill
        return self

    def predict(self, X):
        return np.full(len(X), self._value, dtype=float)


@pytest.fixture
def tabpfn_data():
    rng = np.random.default_rng(0)
    n, d = 60, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = {
        "receiving_yards": rng.normal(40, 15, n).clip(0).astype(np.float32),
        "receptions": rng.normal(4, 2, n).clip(0).astype(np.float32),
        # includes negatives so the non-negative clamp has something to do
        "fumbles_lost": rng.normal(0.0, 0.3, n).astype(np.float32),
    }
    return X, y


def _patch_fake(monkeypatch, model, fill=None):
    monkeypatch.setattr(model, "_new_regressor", lambda: _FakeRegressor(fill=fill))


@pytest.mark.unit
class TestTabPFNMultiTarget:
    def test_importable_and_constructible_without_tabpfn(self):
        # 5/6 positions never enable TabPFN and CI doesn't install it, so the
        # class must import + construct with the dep absent (the ``tabpfn``
        # import is lazy, inside _new_regressor at fit time only).
        model = TabPFNMultiTarget(target_names=TARGETS)
        assert model.target_names == TARGETS
        assert model.non_negative_targets == set(TARGETS)

    def test_fit_predict_shapes_and_keys(self, monkeypatch, tabpfn_data):
        X, y = tabpfn_data
        model = TabPFNMultiTarget(target_names=TARGETS)
        _patch_fake(monkeypatch, model)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds) == set(TARGETS)
        for t in TARGETS:
            assert preds[t].shape == (len(X),)

    def test_non_negative_clamp_all_heads(self, monkeypatch, tabpfn_data):
        X, y = tabpfn_data
        model = TabPFNMultiTarget(target_names=TARGETS)
        _patch_fake(monkeypatch, model, fill=-5.0)  # force negative outputs
        model.fit(X, y)
        preds = model.predict(X)
        for t in TARGETS:
            assert (preds[t] >= 0).all()

    def test_non_negative_subset_lets_signed_head_through(self, monkeypatch, tabpfn_data):
        X, y = tabpfn_data
        # Only clamp receptions; a position with a signed head opts out by
        # passing a subset rather than flipping the clamp globally.
        model = TabPFNMultiTarget(target_names=TARGETS, non_negative_targets={"receptions"})
        _patch_fake(monkeypatch, model, fill=-1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds["receptions"] >= 0).all()
        assert (preds["fumbles_lost"] < 0).all()

    def test_pca_reduction_feeds_reduced_matrix(self, monkeypatch, tabpfn_data):
        X, y = tabpfn_data  # 6 raw features
        model = TabPFNMultiTarget(target_names=TARGETS, pca_n_components=3)
        captured = {}

        class _DimCheckReg(_FakeRegressor):
            def fit(self, Xf, yv):
                captured["n_features"] = Xf.shape[1]
                return super().fit(Xf, yv)

        monkeypatch.setattr(model, "_new_regressor", lambda: _DimCheckReg())
        model.fit(X, y)
        assert captured["n_features"] == 3  # PCA applied before the regressor
        assert model.pca is not None and model.scaler is not None
        assert model.predict(X)["receptions"].shape == (len(X),)

    def test_no_pca_passes_raw_matrix(self, monkeypatch, tabpfn_data):
        X, y = tabpfn_data
        model = TabPFNMultiTarget(target_names=TARGETS)  # pca_n_components=None
        captured = {}

        class _DimCheckReg(_FakeRegressor):
            def fit(self, Xf, yv):
                captured["n_features"] = Xf.shape[1]
                return super().fit(Xf, yv)

        monkeypatch.setattr(model, "_new_regressor", lambda: _DimCheckReg())
        model.fit(X, y)
        assert captured["n_features"] == X.shape[1]  # raw, no reduction
        assert model.pca is None and model.scaler is None

    def test_get_feature_importance_is_empty(self):
        model = TabPFNMultiTarget(target_names=TARGETS)
        assert model.get_feature_importance(["a", "b"]) == {}

    def test_save_load_roundtrip(self, monkeypatch, tabpfn_data, tmp_path):
        X, y = tabpfn_data
        model = TabPFNMultiTarget(target_names=TARGETS, pca_n_components=3)
        _patch_fake(monkeypatch, model)
        model.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])
        before = model.predict(X)

        model_dir = str(tmp_path / "m")
        model.save(model_dir)

        restored = TabPFNMultiTarget(target_names=TARGETS)
        restored.load(model_dir)
        after = restored.predict(X)
        for t in TARGETS:
            np.testing.assert_allclose(before[t], after[t], atol=1e-6)
        # Meta round-trips the knobs serving would rely on.
        assert restored.pca_n_components == 3
        assert restored.non_negative_targets == set(TARGETS)
        assert restored.target_names == TARGETS

    def test_end_to_end_real_tabpfn(self, tabpfn_data):
        # Real weight-loading path; only when the optional dep is present
        # (skipped in CI). Keeps the wrapper honest against the real estimator.
        pytest.importorskip("tabpfn")
        X, y = tabpfn_data
        model = TabPFNMultiTarget(target_names=TARGETS, device="cpu", n_estimators=2)
        model.fit(X, y)
        preds = model.predict(X)
        for t in TARGETS:
            assert preds[t].shape == (len(X),)
            assert np.isfinite(preds[t]).all()


@pytest.mark.unit
def test_config_plumbing_defaults_off_for_every_position():
    # build_pipeline_config must expose train_tabpfn=False by default (dormant
    # infra) plus the tabpfn_* hyperparam keys, for every position. Guards
    # against accidentally shipping TabPFN enabled.
    from src.shared.position_pipeline import build_pipeline_config
    from src.shared.registry import Position

    for pos in Position.values():
        module = __import__(f"src.{pos.lower()}.config", fromlist=["POSITION_CONFIG"])
        cfg = build_pipeline_config(pos, module.POSITION_CONFIG)
        assert cfg["train_tabpfn"] is False, f"{pos} ships TabPFN enabled"
        assert "tabpfn_n_estimators" in cfg
        assert "tabpfn_pca_components" in cfg
        assert "tabpfn_ignore_pretraining_limits" in cfg
