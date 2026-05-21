"""Smoke test for analysis_shap_lgbm.py."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# SHAP's TreeExplainer JITs via numba/llvmlite, which bundles its own libomp.
# When torch (loaded by the root conftest's pytest_configure) has already
# initialized a different libomp, the second registration triggers an OpenMP
# error. KMP_DUPLICATE_LIB_OK=TRUE silences the duplicate-registration check,
# which is sufficient for the unit tests below (they don't actually invoke
# SHAP). The integration tests subprocess into a fresh interpreter so the
# child never loads torch at all — the env var alone proved unreliable on
# macOS once torch had been imported first.
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
    monkeypatch.setattr(
        sys, "argv", ["src/analysis/analysis_shap_lgbm.py", "QB", "--background-samples", "50"]
    )
    with pytest.raises(SystemExit):
        mod.main()


# ---------------------------------------------------------------------------
# Integration tests: run SHAP in a subprocess so the child interpreter
# starts with no torch loaded. The lazy-import refactor in
# src/analysis/analysis_shap_lgbm.py keeps that module's import torch-free;
# the monkeypatched ``build_train_matrix`` below short-circuits the lazy load.
# ---------------------------------------------------------------------------


_SHAP_DRIVER_SCRIPT = r"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ["REPO_ROOT"])

from src.shared.models import LightGBMMultiTarget
from src.analysis import analysis_shap_lgbm

params = json.loads(os.environ["SHAP_PARAMS"])
output_dir = os.path.join(params["tmp_path"], "outputs")
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

rng = np.random.default_rng(0)
targets = params["targets"]
feature_cols = params["feature_cols"]
n = params["n_samples"]
n_features = len(feature_cols)
X = rng.standard_normal((n, n_features)).astype(np.float32)
y_dict = {}
for i, t in enumerate(targets):
    coef_col = i % n_features
    y_dict[t] = (X[:, coef_col] * (i + 2.0) + rng.standard_normal(n) * 0.5).astype(np.float32)

model = LightGBMMultiTarget(
    target_names=targets,
    n_estimators=params["n_estimators"],
    num_leaves=params["num_leaves"],
)
model.fit(X, y_dict, feature_names=feature_cols)
model.save(os.path.join(output_dir, "models"))

# Replicate the monkeypatches from the original pytest scope: hand the runner
# fake training data so it doesn't need data/splits/*.parquet, and bypass the
# lazy build_train_matrix import so torch never enters this child interpreter.
analysis_shap_lgbm._load_position_config = lambda pos: {"targets": targets, "random_seed": 0}
analysis_shap_lgbm.build_train_matrix = lambda pos, cfg: (X, y_dict, feature_cols)

try:
    ranking_path = analysis_shap_lgbm._run_shap_for_position(
        "QB",
        target_filter=params.get("target_filter"),
        background_samples=params["background_samples"],
        seed=params["seed"],
        output_dir=output_dir,
    )
    out = {"ok": True, "ranking_path": ranking_path}
except Exception as exc:
    out = {"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}
print("__SHAP_RESULT__:" + json.dumps(out))
"""


def _run_shap_driver(tmp_path: Path, **params) -> dict:
    """Execute the SHAP driver script in a fresh Python interpreter.

    A child interpreter starts with no torch loaded; combined with the
    lazy-import refactor in src/analysis/analysis_shap_lgbm.py, the SHAP
    TreeExplainer call runs without the OpenMP/libomp conflict that fires
    when torch is loaded first in the parent process (see root conftest.py
    pytest_configure, which imports torch unconditionally).
    """
    repo_root = Path(__file__).resolve().parents[1]
    payload = {"tmp_path": str(tmp_path), **params}
    env = {
        **os.environ,
        "REPO_ROOT": str(repo_root),
        "SHAP_PARAMS": json.dumps(payload),
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _SHAP_DRIVER_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    if proc.returncode != 0:
        pytest.fail(
            "SHAP driver subprocess exited non-zero.\n"
            f"returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    marker = "__SHAP_RESULT__:"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    pytest.fail(
        f"SHAP driver did not emit result marker.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.mark.integration
def test_run_shap_end_to_end(tmp_path):
    """End-to-end SHAP flow on a synthetic LightGBM model.

    Runs in a subprocess so torch (loaded by the root conftest's
    pytest_configure) can't poison numba's libomp in the SHAP path.
    Verifies PNG plots and the ranking JSON are emitted with the expected
    shape. Bypasses the real train matrix and position-config imports with
    in-child monkeypatches so the test doesn't need ``data/splits/*.parquet``
    or a real runner module.
    """
    targets = ["yards", "tds"]
    feature_cols = [f"f{i}" for i in range(5)]
    result = _run_shap_driver(
        tmp_path,
        targets=targets,
        feature_cols=feature_cols,
        n_samples=200,
        n_estimators=30,
        num_leaves=7,
        background_samples=100,
        seed=42,
        target_filter=None,
    )
    assert result["ok"], f"SHAP driver reported error: {result}"
    ranking_path = result["ranking_path"]

    outputs = tmp_path / "outputs"
    assert (outputs / "figures" / "qb_shap_summary_yards.png").exists()
    assert (outputs / "figures" / "qb_shap_summary_tds.png").exists()

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
def test_target_filter_rejects_unknown(tmp_path):
    """--targets with a name not in the model must raise with a useful message."""
    result = _run_shap_driver(
        tmp_path,
        targets=["yards"],
        feature_cols=["f0", "f1"],
        n_samples=100,
        n_estimators=10,
        num_leaves=5,
        background_samples=100,
        seed=0,
        target_filter=["not_a_real_target"],
    )
    assert not result["ok"], f"Expected SHAP driver to error; got {result}"
    assert result["error_type"] == "ValueError", result
    assert "no targets named" in result["error_message"], result
