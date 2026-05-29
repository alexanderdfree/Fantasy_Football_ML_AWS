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
    # Layer 2 — model-vs-model.
    assert hasattr(sig, "paired_bootstrap")
    assert hasattr(sig, "compact_significance")
    assert hasattr(sig, "run_for_position")
    assert hasattr(sig, "main")
    assert sig.CANONICAL_PRED_COLUMNS["Ridge"] == "pred_ridge_total"
    # Layer 1 — paired-error primitives (model-vs-expert), consumed by
    # analysis_expert_comparison.py; a signature break here would only surface
    # at PR-review time otherwise.
    assert hasattr(sig, "diebold_mariano_test")
    assert hasattr(sig, "paired_bootstrap_metric_ci")


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


# ---------- Layer 1 primitives (from PR #447 — model-vs-expert) --------------
# Diebold-Mariano


def test_dm_model_strictly_better_is_significant():
    rng = np.random.default_rng(0)
    e_expert = rng.normal(0.0, 5.0, size=300)
    e_model = 0.3 * e_expert  # same sign, smaller magnitude everywhere
    res = sig.diebold_mariano_test(e_model, e_expert, power=1)
    assert res["mean_loss_diff"] < 0  # |e_model| - |e_expert| < 0
    assert res["dm_stat"] < 0
    assert res["p_value"] < 0.01
    assert res["favored"] == "model"
    assert res["n"] == 300


def test_dm_identical_errors_is_tie():
    e = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    res = sig.diebold_mariano_test(e, e.copy(), power=1)
    assert res["favored"] == "tie"
    assert res["p_value"] == 1.0
    assert res["dm_stat"] == 0.0


def test_dm_power_two_matches_rmse_loss_direction():
    rng = np.random.default_rng(1)
    e_expert = rng.normal(0.0, 4.0, size=200)
    e_model = 0.5 * e_expert
    res = sig.diebold_mariano_test(e_model, e_expert, power=2)
    assert res["favored"] == "model"
    assert res["p_value"] < 0.05


def test_dm_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sig.diebold_mariano_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_dm_too_few_obs_raises():
    with pytest.raises(ValueError):
        sig.diebold_mariano_test([1.0], [2.0])


# Paired bootstrap (clustered)


def test_bootstrap_model_better_ci_excludes_zero():
    rng = np.random.default_rng(2)
    e_expert = rng.normal(0.0, 5.0, size=400)
    e_model = 0.3 * e_expert
    res = sig.paired_bootstrap_metric_ci(e_model, e_expert, metric="mae", n_boot=300, seed=7)
    assert res["delta"] < 0  # model MAE < expert MAE
    assert res["hi"] < 0  # whole CI below zero ⇒ significant
    assert res["p_value"] < 0.05
    assert res["metric"] == "mae"


def test_bootstrap_identical_errors_ci_contains_zero():
    e = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 6.0])
    res = sig.paired_bootstrap_metric_ci(e, e.copy(), metric="rmse", n_boot=200, seed=0)
    assert res["delta"] == 0.0
    assert res["lo"] <= 0.0 <= res["hi"]
    assert res["p_value"] == 1.0


def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(3)
    e_model = rng.normal(0.0, 2.0, size=150)
    e_expert = rng.normal(0.0, 3.0, size=150)
    a = sig.paired_bootstrap_metric_ci(e_model, e_expert, n_boot=200, seed=42)
    b = sig.paired_bootstrap_metric_ci(e_model, e_expert, n_boot=200, seed=42)
    assert a == b


def test_bootstrap_clustered_runs_and_matches_observed_delta():
    rng = np.random.default_rng(4)
    e_model = rng.normal(0.0, 2.0, size=300)
    e_expert = rng.normal(0.0, 4.0, size=300)
    groups = np.arange(300) // 3  # 100 clusters of 3 player-weeks each
    clustered = sig.paired_bootstrap_metric_ci(
        e_model, e_expert, metric="mae", groups=groups, n_boot=200, seed=1
    )
    plain = sig.paired_bootstrap_metric_ci(e_model, e_expert, metric="mae", n_boot=200, seed=1)
    # Observed point estimate is identical; only the resampling (hence CI) differs.
    assert clustered["delta"] == pytest.approx(plain["delta"])
    assert np.isfinite(clustered["lo"]) and np.isfinite(clustered["hi"])


def test_bootstrap_invalid_metric_raises():
    with pytest.raises(ValueError):
        sig.paired_bootstrap_metric_ci([1.0, 2.0], [3.0, 4.0], metric="mape")


def test_bootstrap_groups_length_mismatch_raises():
    with pytest.raises(ValueError):
        sig.paired_bootstrap_metric_ci([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], groups=[0, 1], n_boot=10)
