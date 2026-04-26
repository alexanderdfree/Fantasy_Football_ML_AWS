"""E2E smoke test that drives the ElasticNet branch in ``run_pipeline``.

The default DST E2E test (``test_dst_pipeline_e2e.py``) leaves
``train_elasticnet=False`` — so the ~40-stmt ElasticNet block in
``shared/pipeline.py::run_pipeline`` is never executed by the rest of
the suite. This file flips the flag on, runs the pipeline once on the
same tiny synthetic DST dataset, and asserts the ElasticNet outputs
are wired into ``result["elasticnet_metrics"]`` + ranking dicts.

Borrows the schedule + tiny-cwd machinery from ``test_dst_pipeline_e2e``
to avoid duplicating the chdir + monkeypatch dance.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from DST.dst_config import (
    DST_CONFIG_TINY,
    DST_HUBER_DELTAS,
    DST_LOSS_WEIGHTS,
    DST_POISSON_TARGETS,
    DST_RIDGE_ALPHA_GRIDS,
    DST_SPECIFIC_FEATURES,
    DST_TARGETS,
)
from DST.dst_data import filter_to_dst
from DST.dst_features import (
    add_dst_specific_features,
    compute_dst_features,
    fill_dst_nans,
    get_dst_feature_columns,
)
from DST.dst_targets import compute_dst_targets
from DST.tests.conftest import _build_tiny_dst_dataset
from DST.tests.test_dst_pipeline_e2e import _build_synthetic_schedules
from shared.pipeline import run_pipeline


def _make_dst_elasticnet_cfg() -> dict:
    cfg = {
        "targets": DST_TARGETS,
        "ridge_alpha_grids": DST_RIDGE_ALPHA_GRIDS,
        "specific_features": DST_SPECIFIC_FEATURES,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": None,
        "loss_weights": DST_LOSS_WEIGHTS,
        "huber_deltas": DST_HUBER_DELTAS,
        "poisson_targets": DST_POISSON_TARGETS,
    }
    cfg.update(DST_CONFIG_TINY)
    cfg["train_elasticnet"] = True
    cfg["enet_l1_ratios"] = [0.5]  # single l1_ratio keeps the tune step fast
    return cfg


def _build_tiny_splits(seed: int = 42):
    df = _build_tiny_dst_dataset(seed=seed)
    df = compute_dst_targets(df)
    compute_dst_features(df)
    train = df[df["season"].isin([2022, 2023])].copy()
    val = df[df["season"] == 2024].copy()
    test = df[df["season"] == 2025].copy()
    return train, val, test


@pytest.mark.e2e
@pytest.mark.timeout(120)
def test_run_pipeline_with_train_elasticnet_true(tmp_path_factory):
    """``cfg["train_elasticnet"] = True`` → run_pipeline produces an
    ``elasticnet_metrics`` block alongside ridge / nn metrics, and the
    comparison table includes the ElasticNet row."""
    tiny_dst = _build_tiny_dst_dataset(seed=42)
    sched = _build_synthetic_schedules(tiny_dst)

    mp = pytest.MonkeyPatch()
    tmp_path = tmp_path_factory.mktemp("dst_enet_e2e")
    mp.chdir(tmp_path)
    os.makedirs(tmp_path / "DST" / "outputs" / "models", exist_ok=True)
    os.makedirs(tmp_path / "DST" / "outputs" / "figures", exist_ok=True)

    from shared import weather_features as _wf

    mp.setattr(_wf, "_schedule_cache", sched)
    mp.setattr(_wf, "_load_schedules", lambda: sched)

    try:
        train, val, test = _build_tiny_splits(seed=42)
        cfg = _make_dst_elasticnet_cfg()
        result = run_pipeline("DST", cfg, train, val, test, seed=42)

        assert "elasticnet_metrics" in result
        enet_metrics = result["elasticnet_metrics"]
        # Per-target + total rows must all carry MAE.
        assert "total" in enet_metrics
        assert np.isfinite(enet_metrics["total"]["mae"])
        for t in DST_TARGETS:
            assert t in enet_metrics
            assert np.isfinite(enet_metrics[t]["mae"])

        # Per-target preds must be present and finite.
        enet_preds = result["per_target_preds"]["elasticnet"]
        n_test = len(result["test_df"])
        for t in DST_TARGETS:
            assert enet_preds[t].shape == (n_test,)
            assert np.isfinite(enet_preds[t]).all()
    finally:
        mp.undo()
