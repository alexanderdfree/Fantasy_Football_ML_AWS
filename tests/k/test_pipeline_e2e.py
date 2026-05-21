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

from src.k.config import CONFIG_TINY, POSITION_CONFIG
from src.k.features import compute_features
from src.k.targets import compute_targets
from src.shared.pipeline import run_pipeline
from src.shared.position_pipeline import build_pipeline_config


def _build_e2e_config() -> dict:
    """Build the e2e cfg dict via ``build_pipeline_config``.

    Routes through ``build_pipeline_config("K", POSITION_CONFIG, ...)`` so
    the test exercises the same factory the production runner
    (``src/k/run_pipeline.py``) uses to assemble its CONFIG. Previously the
    test bypassed the factory by reading ``CONFIG_TINY`` directly + appending
    the per-position callables manually, which left PR #287's K-specific
    ``aggregate_fn`` injection (the fix that subtracts ``fg_misses`` /
    ``xp_misses`` from ``_total(preds)``) uncovered at the integration level.

    The shrunken ``CONFIG_TINY`` knobs (1-epoch NN, 2x8 backbone, single-alpha
    ridge grid, no LightGBM) are layered as overrides so the test still fits
    the <20s CPU budget. ``train_attention_nn`` is forced False explicitly —
    ``POSITION_CONFIG`` has it True (production trains nested attention) but
    the e2e fixture doesn't inject the runtime ``attn_history_builder_fn``
    closure that nested attention requires. ``compute_adjustment_fn`` is
    forced None matching K/DST convention in ``tests/_pipeline_e2e_utils.py``.
    """
    return build_pipeline_config(
        "K",
        POSITION_CONFIG,
        **CONFIG_TINY,
        train_attention_nn=False,
        compute_adjustment_fn=None,
    )


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


# ---------------------------------------------------------------------------
# Regression test: H1 — K's aggregate_fn must subtract miss penalties.
#
# Before the fix, ``build_pipeline_config`` skipped ``cfg["aggregate_fn"]`` for
# K, and ``_total(preds)`` in ``src/shared/pipeline.py`` fell back to
# ``sum(preds[t] for t in targets)`` — which added ``fg_misses`` and
# ``xp_misses`` as positive, inflating every K ``pred_*_total`` column,
# corrupting ``compute_ranking_metrics``, and corrupting the K row of
# ``benchmark_history/{run_id}.json``. Serving was unaffected (already
# applied ``target_signs`` directly).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_pipeline_config_registers_k_aggregate_fn():
    """``build_pipeline_config('K', POSITION_CONFIG)`` must populate
    ``cfg['aggregate_fn']`` with a callable that subtracts ``fg_misses``
    and ``xp_misses`` (matching K's ``target_signs = [+1, +1, -1, -1]``).

    Without this, ``_total(preds)`` falls back to ``sum(preds[t] for t in
    targets)`` and silently adds miss penalties as positive — inflating
    every K ``pred_*_total`` column.
    """
    from src.k.config import POSITION_CONFIG
    from src.shared.position_pipeline import build_pipeline_config

    cfg = build_pipeline_config("K", POSITION_CONFIG)
    assert "aggregate_fn" in cfg, (
        "K is missing cfg['aggregate_fn']; _total(preds) will fall back to "
        "sum() and add miss penalties as positive (H1 regression)."
    )

    preds = {
        "fg_yard_points": np.array([10.0, 6.0, 3.0]),
        "pat_points": np.array([2.0, 1.0, 4.0]),
        "fg_misses": np.array([1.0, 0.0, 2.0]),
        "xp_misses": np.array([0.0, 1.0, 1.0]),
    }
    expected = (
        preds["fg_yard_points"] + preds["pat_points"] - preds["fg_misses"] - preds["xp_misses"]
    )
    actual = cfg["aggregate_fn"](preds)
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            "K aggregate_fn must apply target_signs [+1, +1, -1, -1]; "
            "got naive-sum behaviour instead."
        ),
    )

    # And confirm the sign vector actually matters: the buggy fallback would
    # return the naive sum, which is strictly larger here (misses are >= 0).
    naive_sum = sum(preds[t] for t in ("fg_yard_points", "pat_points", "fg_misses", "xp_misses"))
    assert np.all(actual <= naive_sum), (
        "Sanity guard: the correct aggregator must produce totals <= naive sum "
        "when miss counts are non-negative."
    )
    assert np.any(actual < naive_sum), (
        "Sanity guard: at least one synthetic row should have non-zero misses "
        "so the test actually exercises the sign flip."
    )
