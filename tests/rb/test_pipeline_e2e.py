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

from src.rb.config import (
    ATTN_D_MODEL,
    ATTN_DROPOUT,
    ATTN_GATE_HIDDEN,
    ATTN_GATE_WEIGHT,
    ATTN_GATED,
    ATTN_GATED_FUSION,
    ATTN_HISTORY_STATS,
    ATTN_MAX_SEQ_LEN,
    ATTN_N_HEADS,
    ATTN_POSITIONAL_ENCODING,
    ATTN_PROJECT_KV,
    COSINE_ETA_MIN,
    COSINE_T0,
    COSINE_T_MULT,
    GATED_ORDINAL_TARGETS,
    GATED_TARGETS,
    HEAD_LOSSES,
    HUBER_DELTAS,
    LOSS_WEIGHTS,
    NN_BACKBONE_LAYERS_TINY,
    NN_BATCH_SIZE_TINY,
    NN_DROPOUT,
    NN_EPOCHS_TINY,
    NN_HEAD_HIDDEN_TINY,
    NN_LR,
    NN_PATIENCE_TINY,
    NN_WEIGHT_DECAY,
    RIDGE_ALPHA_GRIDS,
    RIDGE_PCA_COMPONENTS,
    SCHEDULER_TYPE,
    SPECIFIC_FEATURES,
    TARGETS,
)
from src.rb.data import filter_to_position
from src.rb.features import add_specific_features, fill_nans, get_feature_columns
from src.rb.targets import compute_targets


def _build_tiny_config() -> dict:
    """Shrunk CONFIG for the E2E smoke."""
    return {
        "targets": TARGETS,
        "ridge_alpha_grids": RIDGE_ALPHA_GRIDS,
        "two_stage_targets": {},
        "classification_targets": GATED_ORDINAL_TARGETS,
        "ridge_pca_components": RIDGE_PCA_COMPONENTS,
        "ridge_cv_folds": 2,
        "ridge_refine_points": 0,
        "cv_split_column": "week",
        "specific_features": SPECIFIC_FEATURES,
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
        "nn_backbone_layers": NN_BACKBONE_LAYERS_TINY,
        "nn_head_hidden": NN_HEAD_HIDDEN_TINY,
        "nn_dropout": NN_DROPOUT,
        "nn_head_hidden_overrides": None,
        "nn_lr": NN_LR,
        "nn_weight_decay": NN_WEIGHT_DECAY,
        "nn_epochs": NN_EPOCHS_TINY,
        "nn_batch_size": NN_BATCH_SIZE_TINY,
        "nn_patience": NN_PATIENCE_TINY,
        "nn_log_every": 1,
        "loss_weights": LOSS_WEIGHTS,
        "huber_deltas": HUBER_DELTAS,
        "scheduler_type": SCHEDULER_TYPE,
        "cosine_t0": COSINE_T0,
        "cosine_t_mult": COSINE_T_MULT,
        "cosine_eta_min": COSINE_ETA_MIN,
        "train_attention_nn": False,
        "attn_d_model": ATTN_D_MODEL,
        "attn_n_heads": ATTN_N_HEADS,
        "attn_max_seq_len": ATTN_MAX_SEQ_LEN,
        "attn_history_stats": ATTN_HISTORY_STATS,
        "attn_project_kv": ATTN_PROJECT_KV,
        "attn_positional_encoding": ATTN_POSITIONAL_ENCODING,
        "attn_gated_fusion": ATTN_GATED_FUSION,
        "attn_dropout": ATTN_DROPOUT,
        "attn_gated": ATTN_GATED,
        "attn_gate_hidden": ATTN_GATE_HIDDEN,
        "attn_gate_weight": ATTN_GATE_WEIGHT,
        "gated_targets": GATED_TARGETS,
        "head_losses": HEAD_LOSSES,
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
        for target in TARGETS:
            vec = preds[model_name][target]
            assert vec.ndim == 1
            assert np.isfinite(vec).all(), f"{model_name}/{target} has non-finite predictions"

    n_test = preds["ridge"][TARGETS[0]].shape[0]
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

    for target in TARGETS:
        np.testing.assert_array_equal(
            preds_a["ridge"][target],
            preds_b["ridge"][target],
            err_msg=f"Ridge predictions drifted for {target}",
        )

    for target in TARGETS:
        a = torch.from_numpy(preds_a["nn"][target])
        b = torch.from_numpy(preds_b["nn"][target])
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
