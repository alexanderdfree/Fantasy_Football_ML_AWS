"""Coverage tests for ``src/shared/evaluation.py``.

Exercises the gate-metric diagnostics (``_sigmoid``, ``_gate_metrics``,
``build_gate_info``), the gate-info-enabled branch of
``compute_target_metrics``, the standalone fantasy-points MAE helper,
ranking metrics (happy path + empty frame), the gated-entries table in
``print_comparison_table``, and ``plot_pred_vs_actual`` at both single
and multi-target shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.shared.aggregate_targets import infer_position
from src.shared.evaluation import (
    _gate_metrics,
    _sigmoid,
    build_gate_info,
    compute_fantasy_points_mae,
    compute_ranking_metrics,
    compute_target_metrics,
    plot_pred_vs_actual,
    print_comparison_table,
)

# --------------------------------------------------------------------------
# _sigmoid / _gate_metrics / build_gate_info
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_sigmoid_is_stable_at_extremes():
    """_sigmoid clips inputs to avoid overflow at ±large values."""
    assert _sigmoid(np.array([1000.0])) == pytest.approx(1.0)
    assert _sigmoid(np.array([-1000.0])) == pytest.approx(0.0)
    assert _sigmoid(np.array([0.0])) == pytest.approx(0.5)


@pytest.mark.unit
def test_gate_metrics_both_classes_present():
    """Mixed y>0 / y==0 → AUC + Brier + cond_mae all finite."""
    y_true = np.array([0.0, 0.0, 5.0, 10.0, 0.0, 8.0])
    gate_logit = np.array([-2.0, -1.0, 1.5, 2.0, -0.5, 0.8])
    value_mu = np.array([0.0, 0.0, 4.5, 10.5, 0.0, 7.8])

    m = _gate_metrics(y_true, gate_logit, value_mu)
    assert m["gate_auc"] is not None and 0.0 <= m["gate_auc"] <= 1.0
    assert m["gate_brier"] >= 0.0
    assert m["positive_rate"] == pytest.approx(3 / 6)
    assert m["cond_mae"] is not None
    assert m["cond_rmse"] is not None


@pytest.mark.unit
def test_gate_metrics_single_class_auc_none():
    """All-zero y_true → AUC is None (undefined), cond_mae is None too."""
    y_true = np.zeros(5)
    gate_logit = np.array([-1.0, -2.0, 0.5, -0.3, 0.1])
    value_mu = np.zeros(5)

    m = _gate_metrics(y_true, gate_logit, value_mu)
    assert m["gate_auc"] is None
    assert m["cond_mae"] is None
    assert m["cond_rmse"] is None
    # Brier + positive_rate still populated.
    assert m["positive_rate"] == 0.0


@pytest.mark.unit
def test_build_gate_info_none_for_empty_gated_targets():
    assert build_gate_info({}, []) is None
    assert build_gate_info({"x": 1}, []) is None


@pytest.mark.unit
def test_build_gate_info_filters_missing_keys():
    """Targets without BOTH gate_logit AND value_mu get skipped silently."""
    preds = {
        "passing_tds_gate_logit": np.array([0.1, 0.2]),
        "passing_tds_value_mu": np.array([1.0, 2.0]),
        # rushing_tds has only the logit — must be skipped.
        "rushing_tds_gate_logit": np.array([0.5, 0.6]),
    }
    info = build_gate_info(preds, ["passing_tds", "rushing_tds"])
    assert info == {
        "passing_tds": {
            "gate_logit": pytest.approx(np.array([0.1, 0.2])),
            "value_mu": pytest.approx(np.array([1.0, 2.0])),
        }
    }


@pytest.mark.unit
def test_build_gate_info_returns_none_when_no_valid_keys():
    """All gated targets missing → return None (not an empty dict)."""
    info = build_gate_info({"unrelated": 1}, ["passing_tds"])
    assert info is None


# --------------------------------------------------------------------------
# compute_target_metrics
# --------------------------------------------------------------------------


def _qb_preds(n: int = 10, seed: int = 0) -> tuple[dict, dict]:
    """Synthetic QB true/pred target dicts."""
    rng = np.random.default_rng(seed)
    targets = (
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "interceptions",
        "fumbles_lost",
    )
    y_true = {t: rng.uniform(0, 300 if "yards" in t else 4, n).astype(float) for t in targets}
    y_pred = {t: y_true[t] + rng.normal(0, 1, n) for t in targets}
    return y_true, y_pred


@pytest.mark.unit
def test_compute_target_metrics_infers_qb_position_and_aggregates_total():
    """QB target set is recognized → ``total`` computed via fantasy-point aggregator."""
    y_true, y_pred = _qb_preds()
    targets = [
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "interceptions",
        "fumbles_lost",
    ]
    results = compute_target_metrics(y_true, y_pred, targets)
    assert "total" in results
    assert results["total"]["unit"] == "pts"
    # Each target has its own metrics + unit.
    for t in targets:
        assert "mae" in results[t]
        assert "unit" in results[t]


@pytest.mark.unit
def test_compute_target_metrics_unknown_position_sums_plainly():
    """Target set that doesn't match any position → ``total`` = sum of per-target."""
    y_true = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    y_pred = {"a": np.array([1.0, 2.5]), "b": np.array([2.5, 4.5])}
    results = compute_target_metrics(y_true, y_pred, ["a", "b"])
    # total is computed on sum a+b = [4, 6] vs preds sum = [3.5, 7.0]
    expected_mae = float(np.mean(np.abs(np.array([4, 6]) - np.array([3.5, 7.0]))))
    assert results["total"]["mae"] == pytest.approx(expected_mae)


@pytest.mark.unit
def test_compute_target_metrics_with_gate_info_adds_diagnostics():
    """gate_info → gated-target metrics gain gate_auc/brier/cond_mae fields."""
    y_true = {
        "passing_tds": np.array([0.0, 0.0, 1.0, 2.0, 0.0]),
        "passing_yards": np.array([200.0, 250.0, 300.0, 150.0, 210.0]),
    }
    y_pred = {
        "passing_tds": np.array([0.1, 0.2, 0.9, 1.8, 0.4]),
        "passing_yards": np.array([220.0, 240.0, 305.0, 160.0, 200.0]),
    }
    gate_info = {
        "passing_tds": {
            "gate_logit": np.array([-1.5, -1.0, 1.0, 2.0, -0.5]),
            "value_mu": np.array([0.0, 0.0, 1.1, 1.9, 0.0]),
        }
    }
    results = compute_target_metrics(
        y_true, y_pred, ["passing_tds", "passing_yards"], gate_info=gate_info
    )
    assert "gate_brier" in results["passing_tds"]
    assert "gate_auc" in results["passing_tds"]
    # Non-gated target has no gate_* fields.
    assert "gate_brier" not in results["passing_yards"]


# --------------------------------------------------------------------------
# infer_position
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_infer_position_matches_qb():
    qb_targets = [
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "interceptions",
        "fumbles_lost",
    ]
    assert infer_position(qb_targets) == "QB"


@pytest.mark.unit
def test_infer_position_matches_k():
    """K and DST target sets must route to their bespoke aggregators rather
    than fall through to the plain-sum branch in compute_target_metrics."""
    from src.k.config import TARGETS as K_CONFIG_TARGETS

    assert infer_position(K_CONFIG_TARGETS) == "K"
    # Order shouldn't matter.
    assert infer_position(list(reversed(K_CONFIG_TARGETS))) == "K"


@pytest.mark.unit
def test_infer_position_matches_dst():
    from src.dst.config import TARGETS as DST_CONFIG_TARGETS

    assert infer_position(DST_CONFIG_TARGETS) == "DST"


@pytest.mark.unit
def test_infer_position_returns_none_on_novel_set():
    assert infer_position(["alpha", "beta"]) is None


@pytest.mark.unit
def test_compute_target_metrics_k_uses_signed_aggregation():
    """K total must be signed sum (fg_yard + pat - fg_miss - xp_miss), not
    the unsigned plain-sum fallback."""
    y_true = {
        "fg_yard_points": np.array([3.0, 5.5, 4.0]),
        "pat_points": np.array([2.0, 3.0, 1.0]),
        "fg_misses": np.array([0.0, 1.0, 0.0]),
        "xp_misses": np.array([0.0, 0.0, 1.0]),
    }
    y_pred = {k: v.copy() for k, v in y_true.items()}
    targets = list(y_true.keys())

    # Perfect preds → 0 MAE.
    assert compute_target_metrics(y_true, y_pred, targets)["total"]["mae"] == pytest.approx(0.0)

    # Perturb fg_misses by +1: signed aggregator yields |Δtotal| = 1 per row.
    # Plain-sum fallback would also give 1 here, so the discriminating check
    # is the fg_yard_points test below.
    y_pred_off = {k: v.copy() for k, v in y_true.items()}
    y_pred_off["fg_misses"] = y_pred_off["fg_misses"] + 1.0
    assert compute_target_metrics(y_true, y_pred_off, targets)["total"]["mae"] == pytest.approx(1.0)

    # Discriminating check: perturb fg_yard_points by +2 AND fg_misses by +2.
    # Signed: total error = +2 - 2 = 0. Plain-sum: total error = +2 + 2 = 4.
    y_pred_cancel = {k: v.copy() for k, v in y_true.items()}
    y_pred_cancel["fg_yard_points"] = y_pred_cancel["fg_yard_points"] + 2.0
    y_pred_cancel["fg_misses"] = y_pred_cancel["fg_misses"] + 2.0
    assert compute_target_metrics(y_true, y_pred_cancel, targets)["total"]["mae"] == pytest.approx(
        0.0
    )


@pytest.mark.unit
def test_compute_target_metrics_dst_uses_proper_aggregation():
    """DST total must use ``_dst_predictions_to_fantasy_points`` (linear stats
    + tier-mapped PA/YA bonuses), not the plain raw-stat sum that's dominated
    by yards_allowed (scale 0-600)."""
    y_true = {
        "def_sacks": np.array([3.0, 0.0, 5.0]),
        "def_ints": np.array([1.0, 0.0, 2.0]),
        "def_fumble_rec": np.array([1.0, 0.0, 1.0]),
        "def_fumbles_forced": np.array([2.0, 1.0, 3.0]),
        "def_safeties": np.array([0.0, 0.0, 1.0]),
        "def_tds": np.array([1.0, 0.0, 2.0]),
        "def_blocked_kicks": np.array([0.0, 0.0, 1.0]),
        "special_teams_tds": np.array([1.0, 0.0, 0.0]),
        "points_allowed": np.array([10.0, 35.0, 0.0]),
        "yards_allowed": np.array([220.0, 420.0, 80.0]),
    }
    y_pred = {k: v.copy() for k, v in y_true.items()}
    targets = list(y_true.keys())

    # Perfect preds → 0 MAE.
    assert compute_target_metrics(y_true, y_pred, targets)["total"]["mae"] == pytest.approx(0.0)

    # Perturb yards_allowed by +100. Proper aggregator: bounded tier-bonus
    # shift (≤ 10 fp/row across the whole YA range). Plain-sum: 100 fp/row.
    y_pred_off = {k: v.copy() for k, v in y_true.items()}
    y_pred_off["yards_allowed"] = y_pred_off["yards_allowed"] + 100.0
    assert compute_target_metrics(y_true, y_pred_off, targets)["total"]["mae"] < 10.0


# --------------------------------------------------------------------------
# compute_fantasy_points_mae
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_fantasy_points_mae_qb_ppr():
    """Returns a non-negative float scalar."""
    y_true, y_pred = _qb_preds(n=20)
    mae = compute_fantasy_points_mae("QB", y_true, y_pred, scoring_format="ppr")
    assert isinstance(mae, float)
    assert mae >= 0


# --------------------------------------------------------------------------
# compute_ranking_metrics
# --------------------------------------------------------------------------


def _ranking_df(n_weeks: int = 4, players_per_week: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for w in range(1, n_weeks + 1):
        for p in range(players_per_week):
            rows.append(
                {
                    "week": w,
                    "player_id": f"P{p:02d}",
                    "fantasy_points": float(rng.uniform(0, 30)),
                    "pred_total": float(rng.uniform(0, 30)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_compute_ranking_metrics_happy_path():
    df = _ranking_df()
    out = compute_ranking_metrics(df, top_k=12)
    assert "weekly" in out
    assert len(out["weekly"]) == 4
    for entry in out["weekly"]:
        assert 0 <= entry["top_k_hit_rate"] <= 1.0
    assert 0 <= out["season_avg_hit_rate"] <= 1.0


@pytest.mark.unit
def test_compute_ranking_metrics_skips_weeks_below_top_k():
    """Weeks with < top_k players drop out (no ranking meaningful)."""
    df = pd.DataFrame(
        {
            "week": [1] * 5 + [2] * 20,
            "player_id": [f"P{i}" for i in range(25)],
            "fantasy_points": np.linspace(0, 30, 25),
            "pred_total": np.linspace(0, 30, 25),
        }
    )
    out = compute_ranking_metrics(df, top_k=12)
    # Week 1 (n=5) skipped, only week 2 remains.
    assert len(out["weekly"]) == 1


@pytest.mark.unit
def test_compute_ranking_metrics_all_weeks_below_top_k_returns_zero_averages():
    """Empty weekly_results → season averages default to 0.0."""
    df = pd.DataFrame(
        {
            "week": [1, 1],
            "player_id": ["A", "B"],
            "fantasy_points": [10.0, 20.0],
            "pred_total": [12.0, 18.0],
        }
    )
    out = compute_ranking_metrics(df, top_k=12)
    assert out["weekly"] == []
    assert out["season_avg_hit_rate"] == 0.0
    assert out["season_avg_spearman"] == 0.0


@pytest.mark.unit
def test_compute_ranking_metrics_constant_predictions_nan_spearman(capsys):
    """Constant predictions → Spearman is NaN; the warning fires AND the
    nanmean fallback returns NaN."""
    rows = [
        {"week": 1, "player_id": f"P{i}", "fantasy_points": float(i), "pred_total": 5.0}
        for i in range(15)
    ]
    df = pd.DataFrame(rows)
    out = compute_ranking_metrics(df, top_k=12)
    # Spearman was NaN → warning printed
    assert "Spearman correlation is NaN" in capsys.readouterr().out
    assert np.isnan(out["season_avg_spearman"])


# --------------------------------------------------------------------------
# print_comparison_table
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_print_comparison_table_basic(capsys):
    """Two models, three targets, no gated-head info."""
    targets = ["passing_yards", "passing_tds"]
    results = {
        "Ridge": {
            "total": {"mae": 5.0, "rmse": 7.0, "r2": 0.3, "unit": "pts"},
            "passing_yards": {"mae": 30.0, "rmse": 40.0, "r2": 0.4, "unit": "yds"},
            "passing_tds": {"mae": 0.8, "rmse": 1.2, "r2": 0.2, "unit": "TDs"},
        },
        "NN": {
            "total": {"mae": 4.5, "rmse": 6.5, "r2": 0.35, "unit": "pts"},
            "passing_yards": {"mae": 25.0, "rmse": 35.0, "r2": 0.45, "unit": "yds"},
            "passing_tds": {"mae": 0.7, "rmse": 1.1, "r2": 0.25, "unit": "TDs"},
        },
    }
    print_comparison_table(results, "QB", targets)
    out = capsys.readouterr().out
    assert "Ridge" in out
    assert "NN" in out
    assert "Per-Target MAE" in out
    # No "Gated-Head Diagnostics" block because no model has gate_* keys
    assert "Gated-Head Diagnostics" not in out


@pytest.mark.unit
def test_print_comparison_table_includes_gated_head_block_when_present(capsys):
    """Model with gate_auc / gate_brier on a target → gated diagnostics table."""
    targets = ["passing_tds"]
    results = {
        "Attn NN": {
            "total": {"mae": 1.0, "rmse": 1.5, "r2": 0.1, "unit": "pts"},
            "passing_tds": {
                "mae": 0.5,
                "rmse": 0.7,
                "r2": 0.2,
                "unit": "TDs",
                "gate_auc": 0.72,
                "gate_brier": 0.18,
                "positive_rate": 0.35,
                "predicted_positive_rate": 0.40,
                "cond_mae": 0.4,
                "cond_rmse": 0.6,
            },
        },
    }
    print_comparison_table(results, "QB", targets)
    out = capsys.readouterr().out
    assert "Gated-Head Diagnostics" in out
    assert "Gate AUC" in out


@pytest.mark.unit
def test_print_comparison_table_gate_with_none_fields_renders_na(capsys):
    """cond_mae=None / gate_auc=None → table renders 'n/a' instead of crashing."""
    targets = ["passing_tds"]
    results = {
        "Attn NN": {
            "total": {"mae": 1.0, "rmse": 1.5, "r2": 0.1, "unit": "pts"},
            "passing_tds": {
                "mae": 0.5,
                "rmse": 0.7,
                "r2": 0.2,
                "unit": "TDs",
                "gate_auc": None,
                "gate_brier": 0.18,
                "positive_rate": 0.0,
                "predicted_positive_rate": 0.10,
                "cond_mae": None,
                "cond_rmse": None,
            },
        },
    }
    print_comparison_table(results, "QB", targets)
    out = capsys.readouterr().out
    assert "n/a" in out


# --------------------------------------------------------------------------
# plot_pred_vs_actual
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_pred_vs_actual_multi_target(tmp_path):
    """4-target grid renders and writes a PNG."""
    targets = ["passing_yards", "rushing_yards", "passing_tds", "rushing_tds"]
    n = 20
    rng = np.random.default_rng(0)
    y_true = {t: rng.uniform(0, 50, n) for t in targets}
    y_pred = {t: y_true[t] + rng.normal(0, 1, n) for t in targets}
    save_path = tmp_path / "pred_vs_actual.png"
    plot_pred_vs_actual(y_true, y_pred, targets, "Ridge", str(save_path))
    assert save_path.exists()


@pytest.mark.unit
def test_plot_pred_vs_actual_single_target(tmp_path):
    """Single-target path hits the ``axes = [axes]`` branch (np.asarray on a
    scalar Axes would otherwise 0-d-array)."""
    n = 15
    rng = np.random.default_rng(0)
    y_true = {"passing_yards": rng.uniform(0, 300, n)}
    y_pred = {"passing_yards": y_true["passing_yards"] + rng.normal(0, 1, n)}
    save_path = tmp_path / "single.png"
    plot_pred_vs_actual(y_true, y_pred, ["passing_yards"], "NN", str(save_path))
    assert save_path.exists()
