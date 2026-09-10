"""Behavioral regression tests for full-score, paired expert comparisons."""

import numpy as np
import pandas as pd
import pytest

from src.serving import comparison
from src.shared.evaluation_cohorts import REFERENCE_VERSION

pytestmark = pytest.mark.unit


def records(n=30):
    return pd.DataFrame(
        {
            "player_id": [f"p{i:02}" for i in range(n)],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "fantasy_points": np.arange(n, dtype=float) + 7,
            "ridge_pred_ppr": np.arange(n, dtype=float),
            "nn_pred_ppr": np.arange(n, dtype=float),
            "attn_nn_pred_ppr": np.arange(n, dtype=float),
            "lgbm_pred_ppr": np.arange(n, dtype=float),
            "nflcom_pred_ppr": np.arange(n, dtype=float),
            "rotowire_pred_ppr": np.arange(n, dtype=float),
            "espn_pred_ppr": np.arange(n, dtype=float),
        }
    )


def test_identical_forecasts_have_identical_errors_on_full_actuals():
    subsets, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    cells = subsets["all"]["WR"]
    assert all(cell["mae"] == 7 for cell in cells.values())
    assert len({cell["n"] for cell in cells.values()}) == 1
    assert coverage["all"]["WR"]["n"] == 30


def test_missing_expert_week_is_excluded_for_every_displayed_source():
    data = records()
    data.loc[0, "fantasy_points"] = 10000
    data.loc[0, "nflcom_pred_ppr"] = np.nan
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert all(cell["mae"] == 7 and cell["n"] == 29 for cell in subsets["all"]["WR"].values())
    assert coverage["all"]["WR"]["cohort_n"] == 30


def test_zero_projection_is_retained_and_infinite_prediction_is_excluded():
    data = records()
    data.loc[1, "nflcom_pred_ppr"] = np.inf
    subsets, _, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert subsets["all"]["WR"]["nflcom"]["n"] == 29


def test_weekly_list_is_reference_selected_before_coverage_filter():
    data = records()
    ref = data[["player_id", "position", "season", "week"]].copy()
    ref["reference_rank"] = np.arange(1, 31)
    ref["reference_version"] = REFERENCE_VERSION
    data.loc[0, "nflcom_pred_ppr"] = np.nan
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=ref)
    assert coverage["weekly_reference_top24"]["WR"]["cohort_n"] == 24
    assert all(cell["n"] == 23 for cell in subsets["weekly_reference_top24"]["WR"].values())


def test_postseason_cannot_change_season_leader_membership():
    data = records().assign(season_type="REG")
    post = data.iloc[[0]].assign(week=19, season_type="POST", fantasy_points=10000)
    subsets, _, _, _ = comparison.comparison_tables(
        pd.concat([data, post]), reference=pd.DataFrame()
    )
    assert subsets["top12"]["WR"]["ridge"]["n"] == 12
    assert subsets["top12"]["WR"]["ridge"]["mae"] == 7


def test_route_ignores_poisoned_static_expert_metrics(app_module, monkeypatch):
    monkeypatch.setattr(
        comparison,
        "_load_comparison_experts",
        lambda: {"subsets": {"all": {"WR": {"nflcom": {"mae": 0}}}}},
    )
    monkeypatch.setattr(comparison, "load_reference", lambda: None)
    app_module._cache.update(results=records(), loaded=True)
    monkeypatch.setattr("src.serving.core._ensure_metrics", lambda: None)
    with app_module.app.test_client() as client:
        body = client.get("/api/comparison").get_json()
    assert body["subsets"]["all"]["WR"]["nflcom"]["mae"] == 7
    assert body["sample_basis"] == "shared_player_weeks"


def test_offline_actual_scoring_includes_wr_rushing_and_qb_receiving():
    from src.analysis.analysis_nflcom_baseline import _aggregate_actuals_to_ppr

    wr = pd.DataFrame(
        {"receiving_yards": [50], "receptions": [5], "rushing_yards": [10], "rushing_tds": [1]}
    )
    qb = pd.DataFrame(
        {"passing_yards": [250], "receiving_yards": [10], "receptions": [1], "receiving_tds": [1]}
    )
    assert _aggregate_actuals_to_ppr(wr, "WR", "ppr")[0] == 17
    assert _aggregate_actuals_to_ppr(qb, "QB", "ppr")[0] == 18
