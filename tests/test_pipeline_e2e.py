"""End-to-end pipeline smoke tests across all 6 positions.

Exercises ``shared.pipeline.run_pipeline`` for each position with a shrunk
config (1 epoch, 2-layer x 8-unit NN, no attention, no LightGBM). Asserts the
orchestration completes, predictions are finite and correctly shaped, and
model artifacts land on disk. Budget: < 20s per position on CPU.

Companion file: ``tests/test_reproducibility.py`` covers bit-identical
predictions across seeded re-runs; this file focuses on completion + shape.
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_splits(request):
    """(train, val, test) tuple sized for the given position — module-scoped."""
    position = request.param
    splits_root = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not (splits_root / "train.parquet").exists():
        pytest.skip(f"Real splits not present at {splits_root}; skipping E2E")
    return load_tiny_splits(position)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.parametrize(
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_pipeline_runs_without_exception(tiny_splits, position, tmp_path):
    """run_pipeline completes end-to-end; result dict contains expected keys."""
    cfg = build_tiny_config(position)
    result = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path, seed=42)

    assert "ridge_metrics" in result
    assert "nn_metrics" in result
    assert "test_df" in result
    assert "per_target_preds" in result


@pytest.mark.e2e
@pytest.mark.parametrize(
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_pipeline_predictions_finite_and_shaped(tiny_splits, position, tmp_path):
    """Ridge and NN predictions are finite and match the test-set row count."""
    cfg = build_tiny_config(position)
    result = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path, seed=42)

    n_test = len(result["test_df"])
    assert n_test > 0, f"{position}: test split is empty — tiny dataset is broken"

    targets = list(cfg["targets"])
    for model_name in ("ridge", "nn"):
        preds = result["per_target_preds"][model_name]
        # Predictions shape: per-target arrays of shape (n_test,)
        for key in targets + ["total"]:
            arr = np.asarray(preds[key])
            assert arr.shape == (n_test,), (
                f"{position} {model_name}.{key}: shape {arr.shape} != ({n_test},)"
            )
            assert np.all(np.isfinite(arr)), (
                f"{position} {model_name}.{key}: contains NaN/Inf"
            )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_pipeline_writes_model_artifacts(tiny_splits, position, tmp_path):
    """Pipeline saves Ridge model + NN state_dict + scaler to the outputs dir."""
    cfg = build_tiny_config(position)
    run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path, seed=42)

    pos_lower = position.lower()
    models_dir = tmp_path / position / "outputs" / "models"
    assert models_dir.is_dir(), f"{position}: models dir not created"

    nn_path = models_dir / f"{pos_lower}_multihead_nn.pt"
    assert nn_path.exists(), f"{position}: NN state_dict not saved at {nn_path}"

    scaler_path = models_dir / "nn_scaler.pkl"
    assert scaler_path.exists(), f"{position}: NN scaler not saved at {scaler_path}"

    # Ridge models are saved as per-target subdirectories. Each target's dir
    # must contain at least one .pkl artefact — the specific files differ
    # between the plain Ridge, two-stage classifier, and ordinal variants.
    targets = list(cfg["targets"])
    target_dirs = [models_dir / t for t in targets if (models_dir / t).is_dir()]
    assert target_dirs, (
        f"{position}: no per-target ridge dirs found in {models_dir}; "
        f"present entries: {[p.name for p in models_dir.iterdir()]}"
    )
    for tdir in target_dirs:
        pkls = list(tdir.glob("*.pkl"))
        assert pkls, (
            f"{position}: no .pkl artifacts saved for target {tdir.name} at {tdir}"
        )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_pipeline_predictions_dataframe_size(tiny_splits, position, tmp_path):
    """The per_target_preds dict carries predictions whose row count matches test_df."""
    cfg = build_tiny_config(position)
    result = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path, seed=42)

    n_val_rows = len(result["test_df"])
    n_targets = len(cfg["targets"])

    # Assemble a (n_val_rows x n_targets) predictions matrix from the NN output
    # to validate the shape contract described in the unit spec.
    nn_preds = result["per_target_preds"]["nn"]
    preds_matrix = np.column_stack([nn_preds[t] for t in cfg["targets"]])
    assert preds_matrix.shape == (n_val_rows, n_targets), (
        f"{position}: preds matrix shape {preds_matrix.shape} != "
        f"({n_val_rows}, {n_targets})"
    )
    assert np.all(np.isfinite(preds_matrix)), (
        f"{position}: preds matrix contains NaN/Inf"
    )
