"""Unit tests for ``benchmark._cohorts_block`` (tracked cohort metrics, #1102/#1106).

The block computes per-cohort, per-model fantasy-point bias/MAE from
``result["test_df"]`` at benchmark time. These tests pin the mask semantics,
the column-presence guards (RB lacks ``is_returning_from_absence`` by design;
K/DST frames lack skill contextual columns), and the empty/degenerate shapes —
no model is trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmarking.benchmark import _cohorts_block

pytestmark = pytest.mark.unit

_N = 8


def _frame(**overrides) -> pd.DataFrame:
    actual = np.linspace(5.0, 19.0, _N)
    df = pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(_N)],
            "season": [2025] * _N,
            "week": [1, 1, 2, 3, 4, 5, 6, 7],
            "fantasy_points": actual,
            # Ridge predicts exactly +1 over actual on every row -> bias +1.0.
            "pred_ridge_total": actual + 1.0,
            "pred_nn_total": actual - 0.5,
            "pred_attn_nn_total": actual,
            "pred_lgbm_total": actual + 2.0,
            "is_returning_from_absence": [0, 1, 0, 1, 0, 0, 0, 0],
            "game_status": [1.0, 0.5, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0],
            "inherited_opportunity": [0.0, 3.2, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0],
            "prior_season_mean_fantasy_points": np.linspace(2.0, 16.0, _N),
        }
    )
    for col, val in overrides.items():
        df[col] = val
    return df


def test_block_masks_and_bias():
    block = _cohorts_block("QB", {"test_df": _frame()})
    assert set(block) == {"week1", "returning", "questionable", "inheritor", "elite_top24"}
    assert block["week1"]["n"] == 2
    assert block["returning"]["n"] == 2
    assert block["questionable"]["n"] == 2
    assert block["inheritor"]["n"] == 2
    # 8 distinct players < 24 -> everyone is "elite".
    assert block["elite_top24"]["n"] == _N
    ridge = block["week1"]["models"]["Ridge"]
    assert ridge["bias"] == pytest.approx(1.0)
    assert ridge["mae"] == pytest.approx(1.0)
    assert set(block["week1"]["models"]) == {"Ridge", "NN", "Attention NN", "LightGBM"}


def test_missing_column_omits_cohort_only():
    df = _frame().drop(columns=["inherited_opportunity", "is_returning_from_absence"])
    block = _cohorts_block("RB", {"test_df": df})
    assert "inheritor" not in block
    assert "returning" not in block
    assert block["week1"]["n"] == 2


def test_empty_cohort_reports_n_zero():
    block = _cohorts_block("QB", {"test_df": _frame(game_status=1.0)})
    assert block["questionable"] == {"n": 0, "models": {}}


def test_elite_top24_caps_at_24_players():
    n = 30
    actual = np.linspace(5.0, 25.0, n)
    df = pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(n)],
            "season": [2025] * n,
            "week": [w % 17 + 2 for w in range(n)],  # no week-1 rows needed here
            "fantasy_points": actual,
            "pred_ridge_total": actual + 1.0,
            "prior_season_mean_fantasy_points": np.arange(n, dtype=float),
        }
    )
    block = _cohorts_block("WR", {"test_df": df})
    assert block["elite_top24"]["n"] == 24


def test_degenerate_results_return_none():
    assert _cohorts_block("QB", {}) is None
    assert _cohorts_block("QB", {"test_df": None}) is None
    # No recognised prediction columns (e.g. a malformed result) -> None.
    no_preds = _frame().drop(
        columns=["pred_ridge_total", "pred_nn_total", "pred_attn_nn_total", "pred_lgbm_total"]
    )
    assert _cohorts_block("QB", {"test_df": no_preds}) is None
    # Frame without the fantasy_points truth column -> None.
    assert _cohorts_block("DST", {"test_df": _frame().drop(columns=["fantasy_points"])}) is None
