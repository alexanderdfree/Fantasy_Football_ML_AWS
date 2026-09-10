"""Behavioral regression tests for shared-component, paired expert comparisons."""

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
            "actual_receiving_yards": np.arange(n, dtype=float) * 10,
            "actual_receptions": 7.0,
            "actual_receiving_tds": 0.0,
            "actual_fumbles_lost": 0.0,
            "ridge_pred_ppr": np.arange(n, dtype=float),
            "nn_pred_ppr": np.arange(n, dtype=float),
            "attn_nn_pred_ppr": np.arange(n, dtype=float),
            "lgbm_pred_ppr": np.arange(n, dtype=float),
            "nflcom_pred_ppr": np.arange(n, dtype=float),
            "rotowire_pred_ppr": np.arange(n, dtype=float),
            "espn_pred_ppr": np.arange(n, dtype=float),
        }
    )


def test_identical_forecasts_have_identical_errors_on_shared_actuals():
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


def test_available_reference_cannot_hide_an_empty_comparison():
    data = records()
    ref = data[["player_id", "position", "season", "week"]].copy()
    ref["reference_rank"] = np.arange(1, 31)
    ref["reference_version"] = REFERENCE_VERSION
    # This source exists in the position, but not for any of the reference top 24.
    data.loc[:23, "nflcom_pred_ppr"] = np.nan
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=ref)
    cell = coverage["weekly_reference_top24"]["WR"]
    assert cell["n"] == 0
    assert cell["reference_status"] == "available"
    assert cell["status"] == "unavailable"
    assert all(value is None for value in subsets["weekly_reference_top24"]["WR"].values())


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


def test_offline_actual_scoring_excludes_wr_rushing_and_qb_receiving():
    from src.analysis.analysis_nflcom_baseline import _aggregate_actuals_to_ppr

    wr = pd.DataFrame(
        {"receiving_yards": [50], "receptions": [5], "rushing_yards": [10], "rushing_tds": [1]}
    )
    qb = pd.DataFrame(
        {"passing_yards": [250], "receiving_yards": [10], "receptions": [1], "receiving_tds": [1]}
    )
    assert _aggregate_actuals_to_ppr(wr, "WR", "ppr")[0] == 10
    assert _aggregate_actuals_to_ppr(qb, "QB", "ppr")[0] == 10


def test_unprojected_actual_stats_cannot_change_any_comparison_or_cohort():
    data = records()
    baseline = comparison.comparison_tables(data, reference=pd.DataFrame())
    data["fantasy_points"] += np.arange(len(data))[::-1] * 100
    data["actual_rushing_yards"] = 1000
    data["actual_rushing_tds"] = 50
    assert comparison.comparison_tables(data, reference=pd.DataFrame()) == baseline


def test_missing_actual_components_are_explicit_and_never_use_full_score():
    data = records().drop(columns="actual_receptions")
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert all(value is None for value in subsets["all"]["WR"].values())
    assert coverage["all"]["WR"]["reason"] == "shared_actual_components_missing"
    data = records()
    data.loc[0, "actual_receptions"] = np.nan
    subsets, _, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert all(value["n"] == 29 for value in subsets["all"]["WR"].values())


def test_incompatible_kicker_total_is_excluded_from_errors_and_quartiles():
    data = records().assign(
        position="K",
        actual_fg_yard_points=5.0,
        actual_pat_points=3.0,
        actual_fg_misses=0.0,
        actual_xp_misses=0.0,
        nflcom_pred_ppr=1000,
        rotowire_pred_ppr=np.nan,
        espn_pred_ppr=8.0,
    )
    subsets, coverage, quartiles, rankings = comparison.comparison_tables(
        data, reference=pd.DataFrame()
    )
    assert subsets["all"]["K"]["nflcom"] is None
    assert subsets["all"]["K"]["espn"]["mae"] == 0
    assert "nflcom" in coverage["all"]["K"]["excluded_sources"]
    assert "nflcom" not in rankings["K"]
    assert all(row.get("nflcom") is None for row in (quartiles["K"] or {}).values())
