"""End-to-end smoke test for the TE pipeline.

Exercises `shared.pipeline.run_pipeline(..., TE_CONFIG_TINY)` on a deterministic
tiny synthetic dataset (50 players x 2 seasons x 17 weeks, seed=42) using
shrunken hyperparameters (2-layer x 8-unit NN, 1 epoch). Asserts:

  - pipeline completes without exception,
  - test predictions exist, have correct shape, and are finite,
  - two independent runs with seed=42 produce bit-identical predictions.

Budget: < 20 seconds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shared.pipeline import run_pipeline
from TE.te_config import TE_CONFIG_TINY, TE_TARGETS
from TE.te_data import filter_to_te
from TE.te_features import (
    add_te_specific_features,
    fill_te_nans,
    get_te_feature_columns,
)
from TE.te_targets import compute_te_targets
from TE.tests.conftest import _build_tiny_te_splits

pytestmark = [
    pytest.mark.e2e,
    # Silence PerformanceWarning from build_features on tiny synthetic input.
    pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning"),
]


def _build_tiny_cfg() -> dict:
    """Bundle TE_CONFIG_TINY with the TE callables required by run_pipeline."""
    return {
        **TE_CONFIG_TINY,
        "filter_fn": filter_to_te,
        "compute_targets_fn": compute_te_targets,
        "add_features_fn": add_te_specific_features,
        "fill_nans_fn": fill_te_nans,
        "get_feature_columns_fn": get_te_feature_columns,
    }


def _run_tiny_pipeline(seed: int = 42):
    """Build fresh tiny splits and run the TE pipeline. Returns result dict."""
    train, val, test = _build_tiny_te_splits(seed=seed)
    # Pandas fragmentation warnings are signal noise from build_features on
    # tiny synthetic data — suppress only inside this helper to keep pytest
    # output readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline("TE", _build_tiny_cfg(), train, val, test, seed=seed)


class TestPipelineE2E:
    def test_pipeline_completes_and_produces_finite_predictions(self, te_tiny_splits):
        """Smoke: pipeline runs cleanly, predictions finite and correctly shaped."""
        train, val, test = te_tiny_splits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_pipeline("TE", _build_tiny_cfg(), train, val, test, seed=42)

        # Structural assertions.
        assert "test_df" in result
        assert "ridge_metrics" in result
        assert "nn_metrics" in result
        test_df = result["test_df"]

        # Prediction columns exist and match test row count.
        for pred_col in ("pred_ridge_total", "pred_nn_total", "pred_baseline"):
            assert pred_col in test_df.columns, f"missing {pred_col}"
            arr = test_df[pred_col].values
            assert arr.shape == (len(test),), f"{pred_col} shape {arr.shape} != ({len(test)},)"
            assert np.isfinite(arr).all(), f"{pred_col} has NaN/Inf"

        # Per-target predictions also present and finite.
        for target in TE_TARGETS:
            for model in ("ridge", "nn"):
                col = f"pred_{model}_{target}"
                assert col in test_df.columns
                assert np.isfinite(test_df[col].values).all()

    def test_pipeline_is_bit_identical_across_runs(self):
        """Two runs with seed=42 yield bit-identical predictions.

        Guards end-to-end reproducibility: a single nondeterministic op
        anywhere in the pipeline (Ridge, NN init, data loader shuffle)
        would break this.
        """
        result_a = _run_tiny_pipeline(seed=42)
        result_b = _run_tiny_pipeline(seed=42)

        test_a = result_a["test_df"]
        test_b = result_b["test_df"]

        # Row count and order must match.
        assert len(test_a) == len(test_b)

        # Every prediction column must be bit-identical (atol=0).
        for col in test_a.columns:
            if not col.startswith("pred_"):
                continue
            a = test_a[col].values
            b = test_b[col].values
            np.testing.assert_array_equal(
                a,
                b,
                err_msg=f"{col} differs across seeded runs",
            )
