"""End-to-end test for the K attention-NN pipeline variant.

Runs the shared pipeline with K_CONFIG_TINY_ATTN + synthetic per-kick data,
asserts the attention branch produces finite predictions, and verifies that
Ridge + base NN metrics are bit-identical to a non-attention run (proves the
L1 rolling features don't leak into the non-attention code path).
"""

import functools
import os

import numpy as np
import pandas as pd
import pytest

from src.K.k_config import (
    K_ATTN_KICK_STATS,
    K_ATTN_MAX_GAMES,
    K_ATTN_MAX_KICKS_PER_GAME,
    K_CONFIG_TINY,
    K_CONFIG_TINY_ATTN,
)
from src.K.k_data import filter_to_k
from src.K.k_features import (
    add_k_specific_features,
    build_nested_kick_history,
    compute_k_features,
    fill_k_nans,
    get_k_feature_columns,
)
from src.K.k_targets import compute_k_targets
from src.shared.pipeline import run_pipeline


@pytest.fixture(scope="module")
def prepared_splits(tiny_k_dataset):
    df = tiny_k_dataset.copy()
    df = compute_k_targets(df)
    compute_k_features(df)
    return (
        df[df["season"] <= 2023].copy(),
        df[df["season"] == 2024].copy(),
        df[df["season"] == 2025].copy(),
    )


@pytest.fixture(scope="module")
def outputs_dir(tmp_path_factory):
    """Redirect pipeline artifact writes to a tmp dir."""
    cwd = os.getcwd()
    tmp_dir = tmp_path_factory.mktemp("k_attn_e2e_outputs")
    (tmp_dir / "K" / "outputs").mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(tmp_dir)
        yield tmp_dir
    finally:
        os.chdir(cwd)


def _callables() -> dict:
    return {
        "filter_fn": filter_to_k,
        "compute_targets_fn": compute_k_targets,
        "add_features_fn": add_k_specific_features,
        "fill_nans_fn": fill_k_nans,
        "get_feature_columns_fn": get_k_feature_columns,
        "compute_adjustment_fn": None,
    }


def _attn_config(kicks_df: pd.DataFrame) -> dict:
    """Build the attention-NN E2E config with a builder closure over kicks_df."""
    builder = functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=K_ATTN_KICK_STATS,
        # Shrink the window aggressively for the E2E budget.
        max_games=4,
        max_kicks_per_game=3,
    )
    cfg = dict(K_CONFIG_TINY_ATTN)
    cfg.update(
        {
            **_callables(),
            "attn_history_builder_fn": builder,
        }
    )
    return cfg


def _base_config() -> dict:
    """Identical to _attn_config but with attention disabled — for diff checks."""
    cfg = dict(K_CONFIG_TINY)
    cfg.update(_callables())
    return cfg


@pytest.fixture(scope="module")
def attn_pipeline_run(prepared_splits, tiny_k_kicks, outputs_dir):
    train, val, test = prepared_splits
    cfg = _attn_config(tiny_k_kicks)
    return run_pipeline("K", cfg, train.copy(), val.copy(), test.copy(), seed=42)


@pytest.fixture(scope="module")
def base_pipeline_run(prepared_splits, outputs_dir):
    train, val, test = prepared_splits
    cfg = _base_config()
    return run_pipeline("K", cfg, train.copy(), val.copy(), test.copy(), seed=42)


@pytest.mark.e2e
def test_attention_nn_row_present(attn_pipeline_run):
    """The comparison and test-df should carry attention-NN outputs."""
    result = attn_pipeline_run
    test_df = result["test_df"]
    assert "pred_attn_nn_total" in test_df.columns, (
        "Attention NN predictions missing — the attention branch didn't run"
    )


@pytest.mark.e2e
def test_attention_predictions_finite_and_shaped(attn_pipeline_run):
    result = attn_pipeline_run
    n_test = len(result["test_df"])
    assert n_test > 0
    attn_total = result["test_df"]["pred_attn_nn_total"].to_numpy()
    assert attn_total.shape == (n_test,)
    assert np.all(np.isfinite(attn_total)), "attention total has NaN/Inf"
    for t in ("fg_yard_points", "pat_points", "fg_misses", "xp_misses"):
        col = f"pred_attn_nn_{t}"
        assert col in result["test_df"].columns, f"missing {col}"
        arr = result["test_df"][col].to_numpy()
        assert np.all(np.isfinite(arr)), f"{col} has NaN/Inf"


@pytest.mark.e2e
def test_ridge_and_base_nn_bit_identical_across_attention_toggle(
    attn_pipeline_run, base_pipeline_run
):
    """Critical: toggling attention on/off must not perturb Ridge or base NN.

    If the L1 features leaked into Ridge/base NN feature_cols (the exact bug
    the attn_static_from_df path is designed to prevent), this would fail.
    """
    attn_test = attn_pipeline_run["test_df"]
    base_test = base_pipeline_run["test_df"]

    for prefix in ("pred_ridge", "pred_nn"):
        for t in ("fg_yard_points", "pat_points", "fg_misses", "xp_misses", "total"):
            col = f"{prefix}_{t}"
            np.testing.assert_array_equal(
                attn_test[col].to_numpy(),
                base_test[col].to_numpy(),
                err_msg=(
                    f"{col} differs between attention-on and attention-off runs — "
                    f"L1 features or the attention code path leaked into the base "
                    f"Ridge/NN predictions."
                ),
            )
