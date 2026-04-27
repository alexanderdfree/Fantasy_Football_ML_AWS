"""Smoke test for analysis_shap_lgbm.py."""

import importlib
import json
import os
import sys

# SHAP's TreeExplainer JITs via numba/llvmlite, which bundles its own libomp.
# When torch (loaded by earlier tests in the suite) has already initialized a
# different libomp, the second registration segfaults on macOS. This escape
# hatch is documented in numba + intel MKL issues; it is a no-op when only
# one OpenMP runtime is present.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.mark.unit
def test_sample_background_deterministic():
    """Same (seed, n) must yield identical rows across runs — or SHAP diffs
    would reflect sampling noise instead of real model change."""
    mod = importlib.import_module("src.analysis.analysis_shap_lgbm")

    X = np.arange(500 * 4, dtype=np.float32).reshape(500, 4)
    bg1 = mod._sample_background(X, n_samples=50, seed=42)
    bg2 = mod._sample_background(X, n_samples=50, seed=42)
    bg3 = mod._sample_background(X, n_samples=50, seed=7)

    assert bg1.shape == (50, 4)
    np.testing.assert_array_equal(bg1, bg2)
    # Different seed must produce different rows (with overwhelming probability).
    assert not np.array_equal(bg1, bg3)


@pytest.mark.unit
def test_sample_background_caps_at_population():
    mod = importlib.import_module("src.analysis.analysis_shap_lgbm")
    X = np.ones((30, 3), dtype=np.float32)
    bg = mod._sample_background(X, n_samples=1000, seed=0)
    assert bg.shape == (30, 3)


@pytest.mark.unit
def test_cli_rejects_tiny_background(monkeypatch, capsys):
    """Smoke check on argparse: --background-samples 50 must fail."""
    mod = importlib.import_module("src.analysis.analysis_shap_lgbm")
    monkeypatch.setattr(sys, "argv", ["src/analysis/analysis_shap_lgbm.py", "QB", "--background-samples", "50"])
    with pytest.raises(SystemExit):
        mod.main()


@pytest.mark.integration
@pytest.mark.skipif(
    "torch" in sys.modules,
    reason=(
        "SHAP TreeExplainer segfaults on macOS when numba's libomp loads after "
        "torch's. Run this test file in isolation (pytest tests/test_shap_analysis.py)."
    ),
)
def test_run_shap_end_to_end(tmp_path, monkeypatch):
    """End-to-end SHAP flow on a synthetic LightGBM model.

    Bypasses the real train matrix and position-config imports with
    monkeypatches so the test doesn't need ``data/splits/*.parquet`` or a
    real runner module. Verifies that PNG summary plots and the ranking
    JSON are emitted with the expected shape.
    """
    from src.shared.models import LightGBMMultiTarget

    mod = importlib.import_module("src.analysis.analysis_shap_lgbm")

    rng = np.random.default_rng(0)
    targets = ["yards", "tds"]
    feature_cols = [f"f{i}" for i in range(5)]
    n = 200
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y_dict = {
        "yards": (X[:, 0] * 3 + rng.standard_normal(n)).astype(np.float32),
        "tds": (X[:, 1] + rng.standard_normal(n) * 0.5).astype(np.float32),
    }

    model = LightGBMMultiTarget(target_names=targets, n_estimators=30, num_leaves=7)
    model.fit(X, y_dict, feature_names=feature_cols)
    outputs = tmp_path / "outputs"
    os.makedirs(outputs / "models", exist_ok=True)
    model.save(str(outputs / "models"))

    # Stub the position loader + train-matrix rebuilder so the test stays
    # hermetic. The real implementations hit disk and runner imports.
    fake_cfg = {"targets": targets, "random_seed": 0}
    monkeypatch.setattr(mod, "_load_position_config", lambda pos: fake_cfg)
    monkeypatch.setattr(mod, "build_train_matrix", lambda pos, cfg: (X, y_dict, feature_cols))

    ranking_path = mod._run_shap_for_position(
        "QB",
        target_filter=None,
        background_samples=100,
        seed=42,
        output_dir=str(outputs),
    )

    # PNG per target.
    assert (outputs / "figures" / "qb_shap_summary_yards.png").exists()
    assert (outputs / "figures" / "qb_shap_summary_tds.png").exists()

    # JSON structure: meta block + one target dict per target, with feature keys.
    assert os.path.exists(ranking_path)
    with open(ranking_path) as f:
        ranking = json.load(f)
    assert "_meta" in ranking
    assert ranking["_meta"]["position"] == "QB"
    assert ranking["_meta"]["seed"] == 42
    assert ranking["_meta"]["background_samples"] == 100
    assert "shap_computed_at" in ranking["_meta"]
    assert "model_trained_at" in ranking["_meta"]

    for target in targets:
        assert target in ranking
        assert set(ranking[target].keys()) == set(feature_cols)
        # Ranking sorted descending — first feature has the highest mean |SHAP|.
        values = list(ranking[target].values())
        assert values == sorted(values, reverse=True)


@pytest.mark.integration
@pytest.mark.skipif(
    "torch" in sys.modules,
    reason=(
        "SHAP TreeExplainer segfaults on macOS when numba's libomp loads after "
        "torch's. Run this test file in isolation (pytest tests/test_shap_analysis.py)."
    ),
)
def test_target_filter_rejects_unknown(tmp_path, monkeypatch):
    """--targets with a name not in the model must raise with a useful message."""
    from src.shared.models import LightGBMMultiTarget

    mod = importlib.import_module("src.analysis.analysis_shap_lgbm")

    rng = np.random.default_rng(0)
    targets = ["yards"]
    feature_cols = ["f0", "f1"]
    X = rng.standard_normal((100, 2)).astype(np.float32)
    y_dict = {"yards": rng.standard_normal(100).astype(np.float32)}

    model = LightGBMMultiTarget(target_names=targets, n_estimators=10, num_leaves=5)
    model.fit(X, y_dict, feature_names=feature_cols)
    outputs = tmp_path / "outputs"
    os.makedirs(outputs / "models", exist_ok=True)
    model.save(str(outputs / "models"))

    monkeypatch.setattr(mod, "_load_position_config", lambda pos: {"targets": targets})
    monkeypatch.setattr(mod, "build_train_matrix", lambda pos, cfg: (X, y_dict, feature_cols))

    with pytest.raises(ValueError, match="no targets named"):
        mod._run_shap_for_position(
            "QB",
            target_filter=["not_a_real_target"],
            background_samples=100,
            seed=0,
            output_dir=str(outputs),
        )
