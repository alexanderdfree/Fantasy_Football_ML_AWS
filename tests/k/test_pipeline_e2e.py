"""End-to-end pipeline smoke test for the K (Kicker) position.

Runs the full shared-kernel pipeline (`run_pipeline`) against a tiny
synthetic dataset with a shrunk neural-net config (2 layers x 8 units,
1 epoch). Asserts:
  - No exceptions.
  - Predictions are finite.
  - Output shapes match test-set size.
  - Two runs with the same seed produce BIT-IDENTICAL predictions
    (atol=0, rtol=0) — reproducibility guard for ridge + NN + baseline.

Budget: < 20s on CPU.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.k.config import CONFIG_TINY
from src.k.data import filter_to_position
from src.k.features import (
    add_specific_features,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.k.targets import compute_targets
from src.shared.pipeline import run_pipeline


def _build_e2e_config() -> dict:
    """Complete the CONFIG_TINY dict with the callables run_pipeline needs."""
    cfg = dict(CONFIG_TINY)
    cfg.update(
        {
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "compute_adjustment_fn": None,
        }
    )
    return cfg


@pytest.fixture(scope="module")
def prepared_splits(tiny_splits):
    """Tiny kicker splits with targets+features computed on the full frame.

    The real pipeline computes K features before splitting (rolling features
    need prior weeks across split boundaries). We do the same here, then
    return train/val/test frames already enriched.
    """
    train, val, test = tiny_splits
    full = pd.concat([train, val, test], ignore_index=True)
    full = compute_targets(full)
    compute_features(full)
    return (
        full[full["season"] <= 2023].copy(),
        full[full["season"] == 2024].copy(),
        full[full["season"] == 2025].copy(),
    )


@pytest.fixture(scope="module")
def e2e_outputs_dir(tmp_path_factory):
    """Redirect the pipeline's hard-coded `k/outputs` writes into a tmp dir.

    The pipeline writes model/figure artifacts relative to cwd as a side
    effect; we chdir into a throwaway directory so the real repo is untouched.
    """
    cwd = os.getcwd()
    tmp_dir = tmp_path_factory.mktemp("k_e2e_outputs")
    (tmp_dir / "k" / "outputs").mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(tmp_dir)
        yield tmp_dir
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Module-scoped pipeline runs — one shared run for smoke/shape assertions,
# a second cached run for the cross-run bit-identity check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(prepared_splits, e2e_outputs_dir):
    """Single pipeline invocation shared across tests (saves ~6s per test)."""
    train, val, test = prepared_splits
    cfg = _build_e2e_config()
    return run_pipeline("K", cfg, train.copy(), val.copy(), test.copy(), seed=42)


@pytest.fixture(scope="module")
def pipeline_run_repeat(prepared_splits, e2e_outputs_dir, pipeline_run):
    """Second pipeline invocation with the same seed for bit-identity checks."""
    train, val, test = prepared_splits
    cfg = _build_e2e_config()
    return run_pipeline("K", cfg, train.copy(), val.copy(), test.copy(), seed=42)


@pytest.mark.e2e
def test_pipeline_e2e_runs_without_exception(pipeline_run):
    """Smoke: pipeline completes end-to-end with tiny config + synthetic data."""
    result = pipeline_run
    assert "ridge_metrics" in result
    assert "nn_metrics" in result
    assert "test_df" in result
    assert "per_target_preds" in result


@pytest.mark.e2e
def test_pipeline_predictions_finite_and_shaped(pipeline_run):
    """Predictions must be finite and shaped like the test set."""
    result = pipeline_run

    n_test = len(result["test_df"])
    assert n_test > 0, "Test split empty — dataset builder is broken"

    for model_name in ("ridge", "nn"):
        preds = result["per_target_preds"][model_name]
        for key in ("fg_yard_points", "pat_points", "fg_misses", "xp_misses"):
            arr = preds[key]
            assert arr.shape == (n_test,), f"{model_name} {key} shape {arr.shape} != ({n_test},)"
            assert np.all(np.isfinite(arr)), f"{model_name} {key} has NaN/Inf"


@pytest.mark.e2e
def test_pipeline_bit_identical_across_seeded_runs(pipeline_run, pipeline_run_repeat):
    """Reproducibility: two runs with seed=42 produce bit-identical predictions.

    Covers the reviewer concern that "training is reproducible" is unverified.
    atol=0, rtol=0 — any non-determinism in Ridge, NN weights, or data
    pipeline would show up here.
    """
    for model_name in ("ridge", "nn"):
        p1 = pipeline_run["per_target_preds"][model_name]
        p2 = pipeline_run_repeat["per_target_preds"][model_name]
        for key in ("fg_yard_points", "pat_points", "fg_misses", "xp_misses"):
            np.testing.assert_array_equal(
                p1[key],
                p2[key],
                err_msg=f"{model_name} {key} differs across seeded runs",
            )
