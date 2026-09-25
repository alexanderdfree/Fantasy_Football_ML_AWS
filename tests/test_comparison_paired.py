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
            "nflcom_comparison_pred_ppr": np.arange(n, dtype=float),
            "rotowire_comparison_pred_ppr": np.arange(n, dtype=float),
            "espn_comparison_pred_ppr": np.arange(n, dtype=float),
        }
    )


GRADED = {"ridge", "nn", "attn_nn", "lgbm", "rotowire", "espn"}


def graded(cells):
    """Cells for graded sources; NFL.com offense is displayed but never graded."""
    assert cells["nflcom"] is None
    return {source: cell for source, cell in cells.items() if cell is not None}


def test_identical_forecasts_have_identical_errors_on_shared_actuals():
    subsets, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    cells = graded(subsets["all"]["WR"])
    assert set(cells) == GRADED
    assert all(cell["mae"] == 7 and cell["bias"] == -7 for cell in cells.values())
    assert len({cell["n"] for cell in cells.values()}) == 1
    assert coverage["all"]["WR"]["n"] == 30
    # Identical errors cannot produce a winner.
    assert coverage["all"]["WR"]["uncertainty"]["winner"] == "tie"


def test_nflcom_offense_is_excluded_as_a_duplicate_stale_rotowire_series():
    data = records()
    data["nflcom_comparison_pred_ppr"] = np.nan  # even a wholly missing NFL.com
    data.loc[:4, "nflcom_comparison_pred_ppr"] = 1000.0  # or a wildly wrong one
    subsets, coverage, quartiles, rankings = comparison.comparison_tables(
        data, reference=pd.DataFrame()
    )
    cell = coverage["all"]["WR"]
    assert cell["n"] == 30  # never narrows or poisons the graded slate
    assert "nflcom" not in cell["sources"] and "nflcom" not in cell["unavailable_sources"]
    assert "RotoWire" in cell["excluded_sources"]["nflcom"]
    assert subsets["all"]["WR"]["nflcom"] is None
    assert "nflcom" not in rankings["WR"]
    assert all(row.get("nflcom") is None for row in quartiles["WR"].values())


def test_missing_expert_week_is_excluded_for_every_displayed_source():
    data = records()
    data.loc[0, "fantasy_points"] = 10000
    data.loc[0, "rotowire_comparison_pred_ppr"] = np.nan
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    cells = graded(subsets["all"]["WR"])
    assert all(cell["mae"] == 7 and cell["n"] == 29 for cell in cells.values())
    assert coverage["all"]["WR"]["cohort_n"] == 30


def test_zero_projection_is_retained_and_infinite_prediction_is_excluded():
    data = records()
    data.loc[1, "rotowire_comparison_pred_ppr"] = np.inf
    data.loc[2, "rotowire_comparison_pred_ppr"] = 0.0
    subsets, _, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert subsets["all"]["WR"]["rotowire"]["n"] == 29


def depth_records():
    data = records()
    data["depth_chart_rank"] = [1.0] * 10 + [2.0] * 10 + [-1.0] * 5 + [np.nan] * 5
    return data


def test_depth_chart_starters_use_neither_outcomes_nor_forecasts():
    data = depth_records()
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    cell = coverage["weekly_depth_starters"]["WR"]
    assert cell["status"] == "available" and cell["n"] == cell["cohort_n"] == 10
    assert cell["selection_basis"] == "pregame_depth_chart_rank_1"
    baseline = [value["n"] for value in graded(subsets["weekly_depth_starters"]["WR"]).values()]
    # Reversing every forecast and every outcome cannot move the membership.
    flipped = data.copy()
    for column in [c for c in data if c.endswith("_ppr")] + ["actual_receiving_yards"]:
        flipped[column] = flipped[column].to_numpy()[::-1]
    _, flipped_coverage, _, _ = comparison.comparison_tables(flipped, reference=pd.DataFrame())
    assert flipped_coverage["weekly_depth_starters"]["WR"]["n"] == 10
    assert baseline == [10] * len(GRADED)


def test_depth_starter_rows_without_outcomes_are_dropped_not_replaced():
    data = depth_records()
    data.loc[0, "actual_receptions"] = np.nan  # a starter with no recorded outcome
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert coverage["weekly_depth_starters"]["WR"]["n"] == 9


def test_depth_starter_cohort_is_unavailable_without_depth_charts():
    _, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    cell = coverage["weekly_depth_starters"]["WR"]
    assert cell["status"] == "unavailable" and cell["n"] == 0
    assert cell["reason"] == "depth_chart_missing"


def test_every_kicker_is_a_depth_chart_starter():
    data = records().assign(
        position="K",
        actual_fg_yard_points=5.0,
        actual_pat_points=3.0,
        actual_fg_misses=0.0,
        actual_xp_misses=0.0,
        rotowire_comparison_pred_ppr=np.nan,
    )
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    cell = coverage["weekly_depth_starters"]["K"]
    assert cell["n"] == 30 and cell["selection_basis"] == "one_unit_per_team_game"


def test_elite_cohort_uses_prior_season_importance_only():
    data = pd.concat([records().assign(week=week) for week in (1, 2)], ignore_index=True)
    number = data["player_id"].str[1:].astype(int)
    data["prior_season_mean_shared_component_points"] = -number.astype(float)  # p00 highest
    data.loc[number >= 24, "ridge_pred_ppr"] = 1000.0  # poisons ridge only if non-members enter
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    cell = coverage["elite_top24"]["WR"]
    assert cell["status"] == "available" and cell["selected_players"] == 24
    assert cell["n"] == 48  # 24 distinct players, both weeks
    cells = graded(subsets["elite_top24"]["WR"])
    assert all(value["n"] == 48 and value["mae"] == 7 for value in cells.values())


def test_elite_cohort_is_unavailable_without_prior_season_scores():
    _, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    cell = coverage["elite_top24"]["WR"]
    assert cell["status"] == "unavailable" and cell["reason"] == "prior_season_scores_missing"


def test_clear_model_advantage_under_both_metrics_is_declared():
    data = records()
    for model in ("ridge", "nn", "attn_nn", "lgbm"):
        data[f"{model}_pred_ppr"] = data["actual_receiving_yards"] * 0.1 + 7  # exact
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    gaps = coverage["all"]["WR"]["uncertainty"]
    assert gaps["status"] == "available" and gaps["players"] == 30
    assert gaps["mae"]["verdict"] == gaps["rmse"]["verdict"] == gaps["winner"] == "models"
    assert gaps["mae"]["ci"][1] < 0 and gaps["mae"]["best_expert"] in {"rotowire", "espn"}


def test_source_with_forecasts_only_on_ungraded_rows_cannot_blank_the_position():
    data = records()
    data.loc[:4, "actual_receptions"] = np.nan  # ungraded rows
    data["espn_comparison_pred_ppr"] = np.nan
    data.loc[:4, "espn_comparison_pred_ppr"] = 1.0  # ESPN forecasts only where nothing is graded
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    cell = coverage["all"]["WR"]
    assert cell["status"] == "available" and cell["n"] == 25
    assert "espn" not in cell["sources"] and cell["unavailable_sources"] == ["espn"]


def test_wholly_unavailable_source_is_named_not_silently_dropped():
    data = records()
    data["espn_comparison_pred_ppr"] = np.nan
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    assert "espn" not in coverage["all"]["WR"]["sources"]
    assert coverage["all"]["WR"]["unavailable_sources"] == ["espn"]


def test_weekly_list_is_reference_selected_before_coverage_filter():
    data = records()
    ref = data[["player_id", "position", "season", "week"]].copy()
    ref["reference_rank"] = np.arange(1, 31)
    ref["reference_version"] = REFERENCE_VERSION
    data.loc[0, "rotowire_comparison_pred_ppr"] = np.nan
    subsets, coverage, _, _ = comparison.comparison_tables(data, reference=ref)
    assert coverage["weekly_reference_top24"]["WR"]["cohort_n"] == 24
    cells = graded(subsets["weekly_reference_top24"]["WR"])
    assert all(cell["n"] == 23 for cell in cells.values())


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
    data.loc[:23, "rotowire_comparison_pred_ppr"] = np.nan
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
        lambda: {"subsets": {"all": {"WR": {"rotowire": {"mae": 0}, "nflcom": {"mae": 0}}}}},
    )
    monkeypatch.setattr(comparison, "load_reference", lambda: None)
    app_module._cache.update(results=records(), loaded=True)
    monkeypatch.setattr("src.serving.core._ensure_metrics", lambda: None)
    with app_module.app.test_client() as client:
        body = client.get("/api/comparison").get_json()
    assert body["subsets"]["all"]["WR"]["rotowire"]["mae"] == 7
    assert body["subsets"]["all"]["WR"]["nflcom"] is None
    assert body["sample_basis"] == "shared_player_weeks"
    assert body["uncertainty_meta"]["method"] == "player_clustered_paired_bootstrap"


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
    assert all(value["n"] == 29 for value in graded(subsets["all"]["WR"]).values())


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
        rotowire_comparison_pred_ppr=np.nan,
        espn_comparison_pred_ppr=8.0,
    )
    subsets, coverage, quartiles, rankings = comparison.comparison_tables(
        data, reference=pd.DataFrame()
    )
    assert subsets["all"]["K"]["nflcom"] is None
    assert subsets["all"]["K"]["espn"]["mae"] == 0
    assert "nflcom" in coverage["all"]["K"]["excluded_sources"]
    assert "nflcom" not in rankings["K"]
    assert all(row.get("nflcom") is None for row in (quartiles["K"] or {}).values())


def test_hindsight_cohorts_report_cells_but_no_verdict():
    subsets, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    for name, reason in (
        ("top12", "selected_on_outcomes"),
        ("top30", "selected_on_outcomes"),
        ("weekly_reference_top24", "selected_by_graded_forecast"),
    ):
        assert coverage[name]["WR"]["uncertainty"] == {"status": "not_applicable", "reason": reason}
    assert subsets["top12"]["WR"]["ridge"]["mae"] == 7
    for name in ("weekly_depth_starters", "all", "elite_top24"):
        assert coverage[name]["WR"]["uncertainty"]["status"] != "not_applicable"


def test_headline_verdict_grades_the_served_model_not_the_best_of_four():
    data = records()
    data["ridge_pred_ppr"] = data["actual_receiving_yards"] * 0.1 + 7  # exact
    data["lgbm_pred_ppr"] = data["actual_receiving_yards"] * 0.1 + 7 + 9  # WR's served model, worse
    _, coverage, _, _ = comparison.comparison_tables(data, reference=pd.DataFrame())
    gaps = coverage["all"]["WR"]["uncertainty"]
    # Ridge is exact, so the family's best-of-four beats the experts...
    assert gaps["winner"] == "models" and gaps["mae"]["best_model"] == "ridge"
    # ...but the row verdict belongs to the model the site serves for WR.
    served = gaps["served_model"]
    assert served["status"] == "available" and served["model"] == "lgbm"
    assert served["winner"] == "experts"
    assert served["mae"]["minus_best_expert"] == 2 and served["mae"]["ci"] == [2, 2]
    assert served["mae"]["best_expert"] in {"rotowire", "espn"}


def test_served_model_comes_from_the_api_contract():
    from src.contracts.api import SERVED_MODEL, SERVED_MODEL_CHAIN

    _, coverage, _, _ = comparison.comparison_tables(records(), reference=pd.DataFrame())
    assert coverage["weekly_depth_starters"]["WR"]["uncertainty"]["served_model"]["model"] == "lgbm"
    assert SERVED_MODEL == {
        "QB": "attn_nn",
        "RB": "lgbm",
        "WR": "lgbm",
        "TE": "attn_nn",
        "K": "ridge",
        "DST": "attn_nn",
    }
    assert all(chain[0] == SERVED_MODEL[pos] for pos, chain in SERVED_MODEL_CHAIN.items())
