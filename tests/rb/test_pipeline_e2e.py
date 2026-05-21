"""End-to-end smoke for the RB pipeline.

Exercises `src.shared.pipeline.run_pipeline` on a tiny synthetic RB dataset with
a shrunk neural net (2-layer x 8-unit backbone, 1 epoch) and asserts:

* No exceptions.
* Predictions are finite and the expected shapes.
* **Bit-identical** outputs across two runs with the same seed
  (`np.testing.assert_array_equal` for Ridge, `torch.testing.assert_close`
  atol=0 for the NN).

Budget: < 20s on CPU.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from src.rb.config import POSITION_CONFIG
from src.shared.position_pipeline import build_pipeline_config
from tests._pipeline_e2e_utils import _TINY_OVERRIDES


def _build_tiny_config() -> dict:
    """Shrunk CONFIG for the E2E smoke.

    Routes through ``build_pipeline_config("RB", POSITION_CONFIG, ...)`` so
    every cfg field the runtime pipeline consumes — including the RB-only
    ``two_stage_targets`` / ``classification_targets`` keys assembled by
    ``_rb_two_stage_targets`` / ``_rb_classification_targets`` — comes
    from the same factory the production runner exercises. The legacy
    hand-rolled dict drifted as new cfg keys were added (it bypassed
    ``build_pipeline_config`` and missed e.g. ``aggregate_fn``,
    ``nn_non_negative_targets``); matching the QB pattern fixes that.

    The shared ``_TINY_OVERRIDES`` (in ``tests/_pipeline_e2e_utils.py``)
    layers the shrunken NN/scheduler/ridge knobs on top, so the E2E budget
    stays under 20s. ``cv_split_column="week"`` is forced because the
    synthetic RB dataset only spans two seasons.
    """
    return build_pipeline_config(
        "RB",
        POSITION_CONFIG,
        **_TINY_OVERRIDES,
        cv_split_column="week",
    )


def _find_data_raw_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(here, "data", "raw")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError("Could not locate data/raw/ relative to test file")


def _run_pipeline_in_tmp(train_df, val_df, test_df, seed: int, workdir: str) -> dict:
    from src.shared.pipeline import run_pipeline

    cfg = _build_tiny_config()
    original_cwd = os.getcwd()
    data_raw_src = _find_data_raw_dir()
    try:
        os.chdir(workdir)
        os.makedirs("src/rb/outputs/models", exist_ok=True)
        os.makedirs("src/rb/outputs/figures", exist_ok=True)
        dst = os.path.join(workdir, "data", "raw")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(data_raw_src, dst)
        return run_pipeline(
            "RB",
            cfg,
            train_df=train_df.copy(),
            val_df=val_df.copy(),
            test_df=test_df.copy(),
            seed=seed,
        )
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Module-scoped pipeline runs — share one (or two) full run(s) across tests
# so we don't retrain from scratch per assertion.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(synthetic_splits):
    """Single pipeline invocation shared across tests.

    Held open in a TemporaryDirectory for the life of the module so the
    pipeline's per-run artifacts stay intact if a test later reads them.
    """
    train_df, val_df, test_df = synthetic_splits
    with tempfile.TemporaryDirectory() as tmp:
        yield _run_pipeline_in_tmp(train_df, val_df, test_df, seed=42, workdir=tmp)


@pytest.fixture(scope="module")
def pipeline_run_repeat(synthetic_splits, pipeline_run):
    """Second pipeline invocation with the same seed for bit-identity checks.

    Depends on pipeline_run so module-scoped ordering is deterministic.
    """
    train_df, val_df, test_df = synthetic_splits
    with tempfile.TemporaryDirectory() as tmp:
        yield _run_pipeline_in_tmp(train_df, val_df, test_df, seed=42, workdir=tmp)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_pipeline_runs_to_completion(pipeline_run):
    """Smoke test: the pipeline must finish with finite predictions."""
    result = pipeline_run

    assert "ridge_metrics" in result
    assert "nn_metrics" in result
    assert "per_target_preds" in result

    preds = result["per_target_preds"]
    for model_name in ("ridge", "nn"):
        assert model_name in preds, f"{model_name} missing from per_target_preds"
        for target in POSITION_CONFIG.targets:
            vec = preds[model_name][target]
            assert vec.ndim == 1
            assert np.isfinite(vec).all(), f"{model_name}/{target} has non-finite predictions"

    n_test = preds["ridge"][POSITION_CONFIG.targets[0]].shape[0]
    assert n_test > 0


@pytest.mark.e2e
@pytest.mark.timeout(180)
def test_pipeline_bit_identical_same_seed(pipeline_run, pipeline_run_repeat):
    """Two runs with seed=42 must produce bit-identical Ridge + NN outputs.

    Timeout bumped to 180s: runs the full pipeline twice end-to-end (Ridge +
    NN training + plot save ×2), and raw-stat target scale means NN training
    uses more epochs than the legacy fantasy-point scale.
    """
    preds_a = pipeline_run["per_target_preds"]
    preds_b = pipeline_run_repeat["per_target_preds"]

    for target in POSITION_CONFIG.targets:
        np.testing.assert_array_equal(
            preds_a["ridge"][target],
            preds_b["ridge"][target],
            err_msg=f"Ridge predictions drifted for {target}",
        )

    for target in POSITION_CONFIG.targets:
        a = torch.from_numpy(preds_a["nn"][target])
        b = torch.from_numpy(preds_b["nn"][target])
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
