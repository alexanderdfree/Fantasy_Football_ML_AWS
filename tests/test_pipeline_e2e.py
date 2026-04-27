"""End-to-end pipeline smoke tests across all 6 positions.

Exercises ``src.shared.pipeline.run_pipeline`` for each position with a shrunk
config (1 epoch, 2-layer x 8-unit NN, no attention, no LightGBM). Asserts the
orchestration completes, predictions are finite and correctly shaped, and
model artifacts land on disk. Budget: < 20s per position on CPU.

Companion file: ``tests/test_reproducibility.py`` covers bit-identical
predictions across seeded re-runs; this file focuses on completion + shape.

All four test functions share a single module-scoped pipeline invocation
per position (indirect parametrization), so a position trains once per
module — not four times — saving ~3x wall clock vs. the previous setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._pipeline_e2e_utils import (
    ALL_POSITIONS,
    build_tiny_config,
    load_tiny_splits,
    run_pipeline_in_tmp,
)

# Markers (unit / integration / e2e / regression) are registered in
# ``tests/conftest.py`` so no local pytest_configure is needed here.


# ---------------------------------------------------------------------------
# Fixtures — one module-scoped pipeline run per position, shared across tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(request, tmp_path_factory):
    """Run the pipeline once per position; all tests in this module reuse it.

    Indirect parametrization supplies the position via ``request.param``.
    Returns a dict with:
      - ``result``: the ``run_pipeline`` output
      - ``workdir``: the tmp dir the pipeline wrote artifacts into (used by
        the artifact-check test that inspects ``{POS}/outputs/models``)
      - ``cfg``: the tiny config used, so per-test assertions can iterate
        over its target list without rebuilding it.
    """
    position = request.param
    splits_root = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not (splits_root / "train.parquet").exists():
        pytest.skip(f"Real splits not present at {splits_root}; skipping E2E")

    splits = load_tiny_splits(position)
    cfg = build_tiny_config(position)
    workdir = tmp_path_factory.mktemp(f"e2e_{position}")
    result = run_pipeline_in_tmp(position, cfg, splits, workdir, seed=42)
    return {"result": result, "workdir": workdir, "cfg": cfg}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.parametrize(
    "pipeline_run,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_run"],
)
def test_pipeline_runs_without_exception(pipeline_run, position):
    """run_pipeline completes end-to-end; result dict contains expected keys."""
    result = pipeline_run["result"]
    assert "ridge_metrics" in result
    assert "nn_metrics" in result
    assert "test_df" in result
    assert "per_target_preds" in result


@pytest.mark.e2e
@pytest.mark.parametrize(
    "pipeline_run,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_run"],
)
def test_pipeline_predictions_finite_and_shaped(pipeline_run, position):
    """Ridge and NN predictions are finite and match the test-set row count."""
    result = pipeline_run["result"]
    cfg = pipeline_run["cfg"]

    n_test = len(result["test_df"])
    assert n_test > 0, f"{position}: test split is empty — tiny dataset is broken"

    targets = list(cfg["targets"])
    for model_name in ("ridge", "nn"):
        preds = result["per_target_preds"][model_name]
        # Predictions shape: per-target arrays of shape (n_test,)
        for key in targets:
            arr = np.asarray(preds[key])
            assert arr.shape == (n_test,), (
                f"{position} {model_name}.{key}: shape {arr.shape} != ({n_test},)"
            )
            assert np.all(np.isfinite(arr)), f"{position} {model_name}.{key}: contains NaN/Inf"


@pytest.mark.e2e
@pytest.mark.parametrize(
    "pipeline_run,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_run"],
)
def test_pipeline_writes_model_artifacts(pipeline_run, position):
    """Pipeline saves Ridge model + NN state_dict + scaler to the outputs dir."""
    workdir = pipeline_run["workdir"]
    cfg = pipeline_run["cfg"]

    pos_lower = position.lower()
    models_dir = workdir / position / "outputs" / "models"
    assert models_dir.is_dir(), f"{position}: models dir not created"

    nn_path = models_dir / f"{pos_lower}_multihead_nn.pt"
    assert nn_path.exists(), f"{position}: NN state_dict not saved at {nn_path}"

    scaler_path = models_dir / "nn_scaler.pkl"
    assert scaler_path.exists(), f"{position}: NN scaler not saved at {scaler_path}"

    # Ridge models are saved as per-target subdirectories. Every target must
    # have its own dir containing at least one .pkl artefact — the specific
    # files differ between the plain Ridge, two-stage classifier, and ordinal
    # variants, but RidgeMultiTarget.save() iterates over all targets, so a
    # missing dir means something failed to persist.
    targets = list(cfg["targets"])
    for t in targets:
        tdir = models_dir / t
        assert tdir.is_dir(), (
            f"{position}: per-target ridge dir for {t!r} missing at {tdir}; "
            f"present entries: {[p.name for p in models_dir.iterdir()]}"
        )
        pkls = list(tdir.glob("*.pkl"))
        assert pkls, f"{position}: no .pkl artifacts saved for target {t} at {tdir}"


@pytest.mark.e2e
@pytest.mark.parametrize(
    "pipeline_run,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_run"],
)
def test_pipeline_predictions_dataframe_size(pipeline_run, position):
    """The per_target_preds dict carries predictions whose row count matches test_df."""
    result = pipeline_run["result"]
    cfg = pipeline_run["cfg"]

    n_test_rows = len(result["test_df"])
    n_targets = len(cfg["targets"])

    # Assemble a (n_test_rows x n_targets) predictions matrix from the NN output
    # to validate the shape contract described in the unit spec.
    nn_preds = result["per_target_preds"]["nn"]
    preds_matrix = np.column_stack([nn_preds[t] for t in cfg["targets"]])
    assert preds_matrix.shape == (n_test_rows, n_targets), (
        f"{position}: preds matrix shape {preds_matrix.shape} != ({n_test_rows}, {n_targets})"
    )
    assert np.all(np.isfinite(preds_matrix)), f"{position}: preds matrix contains NaN/Inf"


@pytest.mark.e2e
def test_pipeline_trains_elasticnet_when_enabled(tmp_path_factory):
    """With ``train_elasticnet=True``, run_pipeline fits ElasticNet alongside
    Ridge and reports its metrics + per-target preds. Smoke-tests the full
    tune→fit→predict→save chain on QB so a future regression in the
    ElasticNet code path can't silently skip CI.
    """
    splits_root = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not (splits_root / "train.parquet").exists():
        pytest.skip(f"Real splits not present at {splits_root}; skipping E2E")

    cfg = build_tiny_config("QB")
    cfg["train_elasticnet"] = True  # flip the switch the tiny override turned off
    cfg.setdefault("enet_l1_ratios", [0.5])  # single ratio keeps the tune fast
    splits = load_tiny_splits("QB")
    workdir = tmp_path_factory.mktemp("e2e_QB_enet")
    result = run_pipeline_in_tmp("QB", cfg, splits, workdir, seed=42)

    assert "elasticnet_metrics" in result
    assert "elasticnet_ranking" in result
    assert "elasticnet" in result["per_target_preds"]

    n_test = len(result["test_df"])
    for target in cfg["targets"]:
        arr = np.asarray(result["per_target_preds"]["elasticnet"][target])
        assert arr.shape == (n_test,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0)  # non-negative clamping enforced

    # ElasticNet artifacts land under models/elasticnet/<target>/.
    enet_dir = workdir / "QB" / "outputs" / "models" / "elasticnet"
    assert enet_dir.is_dir()
    for target in cfg["targets"]:
        tdir = enet_dir / target
        assert tdir.is_dir(), f"ElasticNet per-target dir missing for {target}"
        # vanilla targets persist the alpha / l1_ratio / convergence meta.
        meta = tdir / "meta.json"
        if meta.exists():
            import json as _json

            info = _json.loads(meta.read_text())
            assert "alpha" in info
            assert "l1_ratio" in info
            assert "converged" in info
