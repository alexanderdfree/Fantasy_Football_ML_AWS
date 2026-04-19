"""End-to-end smoke for the RB pipeline.

Exercises `shared.pipeline.run_pipeline` on a tiny synthetic RB dataset with
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
import shutil
import tempfile

import numpy as np
import pytest
import torch

from RB.rb_config import (
    RB_ATTN_D_MODEL,
    RB_ATTN_DROPOUT,
    RB_ATTN_GATED_FUSION,
    RB_ATTN_GATED_TD,
    RB_ATTN_HISTORY_STATS,
    RB_ATTN_MAX_SEQ_LEN,
    RB_ATTN_N_HEADS,
    RB_ATTN_POSITIONAL_ENCODING,
    RB_ATTN_PROJECT_KV,
    RB_ATTN_TD_GATE_HIDDEN,
    RB_ATTN_TD_GATE_WEIGHT,
    RB_COSINE_ETA_MIN,
    RB_COSINE_T0,
    RB_COSINE_T_MULT,
    RB_GATED_ORDINAL_TARGETS,
    RB_HUBER_DELTAS,
    RB_LOSS_W_TOTAL,
    RB_LOSS_WEIGHTS,
    RB_NN_BACKBONE_LAYERS_TINY,
    RB_NN_BATCH_SIZE_TINY,
    RB_NN_DROPOUT,
    RB_NN_EPOCHS_TINY,
    RB_NN_HEAD_HIDDEN_TINY,
    RB_NN_LR,
    RB_NN_PATIENCE_TINY,
    RB_NN_WEIGHT_DECAY,
    RB_RIDGE_ALPHA_GRIDS,
    RB_RIDGE_PCA_COMPONENTS,
    RB_SCHEDULER_TYPE,
    RB_SPECIFIC_FEATURES,
    RB_TARGETS,
)
from RB.rb_data import filter_to_rb
from RB.rb_features import add_rb_specific_features, fill_rb_nans, get_rb_feature_columns
from RB.rb_targets import compute_fumble_adjustment, compute_rb_targets


def _build_tiny_config() -> dict:
    """Shrunk RB_CONFIG for the E2E smoke: 1 epoch, 2-layer x 8-unit NN.

    Disables the attention NN and LightGBM to keep the budget under 20s.
    Uses ``cv_split_column="week"`` so Ridge alpha tuning has enough
    distinct folds even with a single training season.
    """
    return {
        "targets": RB_TARGETS,
        "ridge_alpha_grids": RB_RIDGE_ALPHA_GRIDS,
        "two_stage_targets": {},
        "classification_targets": RB_GATED_ORDINAL_TARGETS,
        "ridge_pca_components": RB_RIDGE_PCA_COMPONENTS,
        "ridge_cv_folds": 2,
        "ridge_refine_points": 0,
        "cv_split_column": "week",
        "specific_features": RB_SPECIFIC_FEATURES,
        "filter_fn": filter_to_rb,
        "compute_targets_fn": compute_rb_targets,
        "add_features_fn": add_rb_specific_features,
        "fill_nans_fn": fill_rb_nans,
        "get_feature_columns_fn": get_rb_feature_columns,
        "compute_adjustment_fn": compute_fumble_adjustment,
        "nn_backbone_layers": RB_NN_BACKBONE_LAYERS_TINY,
        "nn_head_hidden": RB_NN_HEAD_HIDDEN_TINY,
        "nn_dropout": RB_NN_DROPOUT,
        "nn_head_hidden_overrides": None,
        "nn_lr": RB_NN_LR,
        "nn_weight_decay": RB_NN_WEIGHT_DECAY,
        "nn_epochs": RB_NN_EPOCHS_TINY,
        "nn_batch_size": RB_NN_BATCH_SIZE_TINY,
        "nn_patience": RB_NN_PATIENCE_TINY,
        "nn_log_every": 1,
        "loss_weights": RB_LOSS_WEIGHTS,
        "loss_w_total": RB_LOSS_W_TOTAL,
        "huber_deltas": RB_HUBER_DELTAS,
        "scheduler_type": RB_SCHEDULER_TYPE,
        "cosine_t0": RB_COSINE_T0,
        "cosine_t_mult": RB_COSINE_T_MULT,
        "cosine_eta_min": RB_COSINE_ETA_MIN,
        "train_attention_nn": False,
        "attn_d_model": RB_ATTN_D_MODEL,
        "attn_n_heads": RB_ATTN_N_HEADS,
        "attn_max_seq_len": RB_ATTN_MAX_SEQ_LEN,
        "attn_history_stats": RB_ATTN_HISTORY_STATS,
        "attn_project_kv": RB_ATTN_PROJECT_KV,
        "attn_positional_encoding": RB_ATTN_POSITIONAL_ENCODING,
        "attn_gated_fusion": RB_ATTN_GATED_FUSION,
        "attn_dropout": RB_ATTN_DROPOUT,
        "attn_gated_td": RB_ATTN_GATED_TD,
        "attn_td_gate_hidden": RB_ATTN_TD_GATE_HIDDEN,
        "attn_td_gate_weight": RB_ATTN_TD_GATE_WEIGHT,
        "train_lightgbm": False,
    }


def _find_data_raw_dir() -> str:
    """Locate the real data/raw/ directory.

    The test may run from a git worktree (under .claude/worktrees/agent-XXX/)
    which does NOT contain the parquet files. Walk up from the test file until
    we find a sibling `data/raw/schedules_*.parquet` — that's the main repo.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # Walk up a few levels — the worktree layout puts tests four levels below
    # the superproject root.
    for _ in range(8):
        candidate = os.path.join(here, "data", "raw")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError("Could not locate data/raw/ relative to test file")


def _run_pipeline_in_tmp(train_df, val_df, test_df, seed: int, workdir: str) -> dict:
    """Run `shared.pipeline.run_pipeline` from `workdir` so the artefact writes
    (RB/outputs/models, ...) land in the tmp directory and get cleaned up.

    The pipeline writes plots to disk. We chdir into a tmp dir for the duration
    of the run to isolate side effects.
    """
    from shared.pipeline import run_pipeline

    cfg = _build_tiny_config()
    original_cwd = os.getcwd()
    data_raw_src = _find_data_raw_dir()
    try:
        os.chdir(workdir)
        os.makedirs("RB/outputs/models", exist_ok=True)
        os.makedirs("RB/outputs/figures", exist_ok=True)
        # The pipeline's weather merge expects data/raw/schedules_*.parquet
        # relative to cwd. Symlink the real one from the main repo.
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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_pipeline_runs_to_completion(synthetic_rb_splits):
    """Smoke test: the pipeline must finish with finite predictions."""
    train_df, val_df, test_df = synthetic_rb_splits

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_pipeline_in_tmp(train_df, val_df, test_df, seed=42, workdir=tmp)

    assert "ridge_metrics" in result
    assert "nn_metrics" in result
    assert "per_target_preds" in result

    preds = result["per_target_preds"]
    for model_name in ("ridge", "nn"):
        assert model_name in preds, f"{model_name} missing from per_target_preds"
        for target in list(RB_TARGETS) + ["total"]:
            vec = preds[model_name][target]
            assert vec.ndim == 1
            assert np.isfinite(vec).all(), f"{model_name}/{target} has non-finite predictions"

    # Expected shape: one row per test row (rows that survive filter_to_rb).
    n_test = preds["ridge"]["total"].shape[0]
    assert n_test > 0


@pytest.mark.e2e
def test_pipeline_bit_identical_same_seed(synthetic_rb_splits):
    """Two runs with seed=42 must produce bit-identical Ridge + NN outputs.

    This is the reproducibility guarantee flagged by the code review — any
    drift (e.g. a new source of non-determinism in training) will fail here.
    """
    train_df, val_df, test_df = synthetic_rb_splits

    with tempfile.TemporaryDirectory() as tmp_a:
        result_a = _run_pipeline_in_tmp(train_df, val_df, test_df, seed=42, workdir=tmp_a)
    with tempfile.TemporaryDirectory() as tmp_b:
        result_b = _run_pipeline_in_tmp(train_df, val_df, test_df, seed=42, workdir=tmp_b)

    preds_a = result_a["per_target_preds"]
    preds_b = result_b["per_target_preds"]

    # Ridge is closed-form → identical bytes expected.
    for target in list(RB_TARGETS) + ["total"]:
        np.testing.assert_array_equal(
            preds_a["ridge"][target],
            preds_b["ridge"][target],
            err_msg=f"Ridge predictions drifted for {target}",
        )

    # NN is CPU-only with matched seeds and single-threaded training, so
    # outputs should match within torch's default tolerance. Use atol=0 to
    # flag the tiniest drift.
    for target in list(RB_TARGETS) + ["total"]:
        a = torch.from_numpy(preds_a["nn"][target])
        b = torch.from_numpy(preds_b["nn"][target])
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
