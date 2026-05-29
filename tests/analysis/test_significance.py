"""Unit tests for src/analysis/significance.py (paired block bootstrap).

Pure-function coverage on synthetic test frames — no data splits, no model training, no
torch. ``run_for_position`` (which re-runs the pipeline) is out of scope for unit tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import significance as sig

pytestmark = pytest.mark.unit


def _synth_test_df(
    seed: int = 0,
    ridge_sigma: float = 5.0,
    nn_sigma: float = 1.0,
    weeks: int = 18,
    players: int = 20,
):
    """Synthetic test season: NN error sigma < Ridge error sigma => NN is genuinely better."""
    rng = np.random.default_rng(seed)
    rows = []
    for wk in range(1, weeks + 1):
        for p in range(players):
            true = rng.normal(12, 6)
            rows.append(
                {
                    "week": wk,
                    "player_id": f"p{p}",
                    "fantasy_points": true,
                    "pred_baseline": true + rng.normal(0, 8),
                    "pred_ridge_total": true + rng.normal(0, ridge_sigma),
                    "pred_nn_total": true + rng.normal(0, nn_sigma),
                }
            )
    return pd.DataFrame(rows)


_COLS = {"Season Avg": "pred_baseline", "Ridge": "pred_ridge_total", "Neural Net": "pred_nn_total"}


def test_module_imports_cleanly():
    assert hasattr(sig, "paired_bootstrap")
    assert hasattr(sig, "compact_significance")
    assert hasattr(sig, "run_for_position")
    assert hasattr(sig, "main")
    assert sig.CANONICAL_PRED_COLUMNS["Ridge"] == "pred_ridge_total"


def test_obvious_gap_is_significant():
    df = _synth_test_df(nn_sigma=1.0, ridge_sigma=8.0)
    res = sig.paired_bootstrap(df, _COLS, n_boot=500)
    mae = next(
        p
        for p in res["pairs"]
        if p["model"] == "Neural Net" and p["other"] == "Ridge" and p["metric"] == "mae"
    )
    assert mae["significant"] is True
    assert mae["model_better"] is True
    assert mae["delta"] > 0  # NN has lower MAE


def test_no_gap_is_within_noise():
    df = _synth_test_df()
    df["pred_nn_total"] = df["pred_ridge_total"]  # identical predictions => zero gap
    res = sig.paired_bootstrap(df, _COLS, n_boot=400)
    mae = next(
        p
        for p in res["pairs"]
        if p["model"] == "Neural Net" and p["other"] == "Ridge" and p["metric"] == "mae"
    )
    assert mae["significant"] is False
    assert mae["delta"] == pytest.approx(0.0, abs=1e-9)


def test_seed_reproducible():
    df = _synth_test_df()
    a = sig.paired_bootstrap(df, _COLS, n_boot=300, seed=7)
    b = sig.paired_bootstrap(df, _COLS, n_boot=300, seed=7)
    assert a["pairs"] == b["pairs"]


def test_week_with_few_players_skipped_for_topk():
    df = _synth_test_df()
    # Shrink one week below top_k=12 — it must be skipped for hit rate, not crash.
    df = pd.concat([df[df["week"] != 5], df[df["week"] == 5].head(5)], ignore_index=True)
    res = sig.paired_bootstrap(df, _COLS, n_boot=200, top_k=12)
    mae = next(p for p in res["pairs"] if p["metric"] == "mae" and p["model"] == "Neural Net")
    assert np.isfinite(mae["delta"])


def test_p_value_floored_never_zero():
    df = _synth_test_df(nn_sigma=0.5, ridge_sigma=10.0)
    res = sig.paired_bootstrap(df, _COLS, n_boot=200)
    for p in res["pairs"]:
        assert p["p_value"] >= 1.0 / 200
        assert p["p_value"] > 0.0


def test_reference_not_present_raises():
    df = _synth_test_df()
    with pytest.raises(ValueError):
        sig.paired_bootstrap(df, _COLS, reference="Nonexistent")


def test_player_unit_forces_mae_only():
    df = _synth_test_df()
    res = sig.paired_bootstrap(df, _COLS, n_boot=200, unit="player")
    assert all(p["metric"] == "mae" for p in res["pairs"])
    assert res["unit"] == "player"


def test_compact_significance_json_serializable():
    df = _synth_test_df()
    res = sig.paired_bootstrap(df, _COLS, n_boot=200)
    compact = sig.compact_significance(res)
    assert compact["best_model"] == "Neural Net"
    json.dumps(compact)  # must not raise (native floats only)
    json.dumps(res)


def test_pred_columns_from_test_df_subset():
    df = _synth_test_df()  # has baseline/ridge/nn but no enet/attn/lgbm
    cols = sig.pred_columns_from_test_df(df)
    assert set(cols) == {"Season Avg", "Ridge", "Neural Net"}
