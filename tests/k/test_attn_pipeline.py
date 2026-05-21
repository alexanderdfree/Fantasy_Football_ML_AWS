"""End-to-end test for the K attention-NN pipeline variant.

Runs the shared pipeline with CONFIG_TINY_ATTN + synthetic per-kick data,
asserts the attention branch produces finite predictions, and verifies that
Ridge + base NN metrics are bit-identical to a non-attention run (proves the
attention history features (``attn_history_stats``, ``attn_kick_stats``,
``attn_static_features``) don't leak into the non-attention code path).
"""

import functools
import os

import numpy as np
import pandas as pd
import pytest

from src.k.config import CONFIG_TINY, CONFIG_TINY_ATTN, POSITION_CONFIG

ATTN_KICK_STATS = POSITION_CONFIG.attn_kick_stats
ATTN_MAX_GAMES = POSITION_CONFIG.attn_max_games
ATTN_MAX_KICKS_PER_GAME = POSITION_CONFIG.attn_max_kicks_per_game
from src.k.data import filter_to_position
from src.k.features import (
    add_specific_features,
    build_nested_kick_history,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.k.targets import compute_targets
from src.shared.pipeline import run_pipeline


@pytest.fixture(scope="module")
def prepared_splits(tiny_dataset):
    df = tiny_dataset.copy()
    df = compute_targets(df)
    compute_features(df)
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
    (tmp_dir / "k" / "outputs").mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(tmp_dir)
        yield tmp_dir
    finally:
        os.chdir(cwd)


def _callables() -> dict:
    return {
        "filter_fn": filter_to_position,
        "compute_targets_fn": compute_targets,
        "add_features_fn": add_specific_features,
        "fill_nans_fn": fill_nans,
        "get_feature_columns_fn": get_feature_columns,
        "compute_adjustment_fn": None,
    }


def _attn_config(kicks_df: pd.DataFrame) -> dict:
    """Build the attention-NN E2E config with a builder closure over kicks_df."""
    builder = functools.partial(
        build_nested_kick_history,
        kicks_df=kicks_df,
        kick_stats=ATTN_KICK_STATS,
        # Shrink the window aggressively for the E2E budget.
        max_games=4,
        max_kicks_per_game=3,
    )
    cfg = dict(CONFIG_TINY_ATTN)
    cfg.update(
        {
            **_callables(),
            "attn_history_builder_fn": builder,
        }
    )
    return cfg


def _base_config() -> dict:
    """Identical to _attn_config but with attention disabled — for diff checks."""
    cfg = dict(CONFIG_TINY)
    cfg.update(_callables())
    return cfg


@pytest.fixture(scope="module")
def attn_pipeline_run(prepared_splits, tiny_kicks, outputs_dir):
    train, val, test = prepared_splits
    cfg = _attn_config(tiny_kicks)
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

    If the attention history features leaked into Ridge/base NN feature_cols
    (the exact bug the attn_static_from_df path is designed to prevent), this
    would fail.
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
                    f"attention history features or the attention code path leaked "
                    f"into the base Ridge/NN predictions."
                ),
            )


# ---------------------------------------------------------------------------
# Tuner-shape skip path — Ridge/ENet/LGBM/base NN all gated off, attention only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tuner_shape_pipeline_run(prepared_splits, tiny_kicks, outputs_dir):
    """Pipeline run matching what src/tuning/tune_nn.py does per Optuna trial:
    only the attention NN trains. Ridge / ElasticNet / LightGBM / base NN are
    gated off via cfg, which triggers the short-circuit early return in
    run_pipeline that skips comparison / ranking / backtest / save artifacts.
    """
    train, val, test = prepared_splits
    cfg = _attn_config(tiny_kicks)
    cfg["train_ridge"] = False
    cfg["train_elasticnet"] = False
    cfg["train_lightgbm"] = False
    cfg["train_base_nn"] = False
    return run_pipeline("K", cfg, train.copy(), val.copy(), test.copy(), seed=42)


@pytest.mark.e2e
def test_tuner_shape_returns_attn_history_and_skips_other_models(tuner_shape_pipeline_run):
    """src/tuning/tune_nn.py reads result["attn_history"]["val_loss"] for its
    Optuna objective. With the four skip gates set, the pipeline short-circuits
    after the CPU/GPU join; this asserts the minimal result still surfaces
    attn_history and that Ridge / base NN metrics are None.
    """
    result = tuner_shape_pipeline_run

    # Short-circuit fired: downstream artifacts not in the result.
    assert "test_df" not in result, "short-circuit did not fire — downstream ran"
    assert "sim_results" not in result, "short-circuit did not fire — backtest ran"
    assert "per_target_preds" not in result

    # Skipped models report None.
    assert result["ridge_metrics"] is None
    assert result["nn_metrics"] is None
    assert "elasticnet_metrics" not in result
    assert "lgbm_metrics" not in result

    # Attention NN ran — what the tuner objective reads.
    assert "attn_history" in result
    val_losses = result["attn_history"]["val_loss"]
    assert len(val_losses) > 0
    assert all(np.isfinite(v) for v in val_losses)
