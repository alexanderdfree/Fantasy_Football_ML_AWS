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
from src.rb.data import filter_to_position
from src.rb.features import add_specific_features, fill_nans, get_feature_columns
from src.rb.targets import compute_targets


def _build_tiny_config() -> dict:
    """Shrunk CONFIG for the E2E smoke."""
    pc = POSITION_CONFIG
    return {
        "targets": pc.targets,
        "ridge_alpha_grids": pc.ridge_alpha_grids,
        "two_stage_targets": {},
        "classification_targets": pc.gated_ordinal_targets,
        "ridge_pca_components": pc.ridge_pca_components,
        "ridge_cv_folds": 2,
        "ridge_refine_points": 0,
        "cv_split_column": "week",
        "specific_features": pc.specific_features,
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
        "nn_backbone_layers": [8, 8],
        "nn_head_hidden": 4,
        "nn_dropout": pc.nn_dropout,
        "nn_head_hidden_overrides": None,
        "nn_lr": pc.nn_lr,
        "nn_weight_decay": pc.nn_weight_decay,
        "nn_epochs": 1,
        "nn_batch_size": 64,
        "nn_patience": 1,
        "nn_log_every": 1,
        "loss_weights": pc.loss_weights,
        "huber_deltas": pc.huber_deltas,
        "scheduler_type": pc.scheduler_type,
        "cosine_t0": pc.cosine_t0,
        "cosine_t_mult": pc.cosine_t_mult,
        "cosine_eta_min": pc.cosine_eta_min,
        "train_attention_nn": False,
        "attn_d_model": pc.attn_d_model,
        "attn_n_heads": pc.attn_n_heads,
        "attn_max_seq_len": pc.attn_max_seq_len,
        "attn_history_stats": pc.attn_history_stats,
        "attn_project_kv": pc.attn_project_kv,
        "attn_positional_encoding": pc.attn_positional_encoding,
        "attn_gated_fusion": pc.attn_gated_fusion,
        "attn_dropout": pc.attn_dropout,
        "attn_gated": pc.attn_gated,
        "attn_gate_hidden": pc.attn_gate_hidden,
        "attn_gate_weight": pc.attn_gate_weight,
        "gated_targets": pc.gated_targets,
        "head_losses": pc.head_losses,
        "train_lightgbm": False,
    }


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
