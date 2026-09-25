from copy import deepcopy

import pandas as pd
import pytest

from src.analysis.practice_cutoffs import compare_cutoff, select_cutoff_rows

pytestmark = pytest.mark.unit


def snapshot(time="2026-09-25T16:00:00Z", *, kickoff="2026-09-27T17:00:00Z", prediction=10.0):
    return {
        "schema_version": 1,
        "evidence_kind": "observed_live_snapshot",
        "season": 2026,
        "week": 3,
        "observed_at": time,
        "forecast_available_at": time,
        "games": [{"team": "BAL", "kickoff": kickoff}],
        "practice": {
            "observations": [
                {
                    "player_id": "a",
                    "season": 2026,
                    "week": 3,
                    "observed_at": time,
                    "reported_at": None,
                    "injury_descriptions": ["knee"],
                    "coverage": "reported",
                }
            ]
        },
        "forecast": {
            "available": True,
            "generated_at": time,
            "input_signature": "test",
            "evaluation_context": {
                "players": [
                    {
                        "player_id": "a",
                        "returning": True,
                        "game_status": 0.5,
                        "elite_top24": True,
                        "weekly_reference_top24": None,
                    }
                ]
            },
            "scoring": {
                "ppr": [
                    {"player_id": "a", "position": "RB", "team": "BAL", "ridge_pred": prediction}
                ]
            },
        },
    }


def test_latest_eligible_forecast_and_reports_never_leak_across_cutoff():
    early = snapshot()
    at_cutoff = snapshot("2026-09-25T17:00:00Z", prediction=12.0)
    late = snapshot("2026-09-25T17:00:01Z", prediction=99.0)
    assert select_cutoff_rows([late, early, at_cutoff]).ridge_pred.iloc[0] == 12.0
    assert select_cutoff_rows([early, late], hours=24).ridge_pred.iloc[0] == 99.0
    revised = deepcopy(early)
    revised["practice"]["observations"][0]["reported_at"] = "2026-09-25T18:00:00Z"
    assert select_cutoff_rows([revised]).empty
    revised = deepcopy(early)
    revised["forecast_available_at"] = "2026-09-25T18:00:00Z"
    assert select_cutoff_rows([revised]).empty


def test_cutoffs_are_relative_to_thursday_game_and_repeated_polls_are_one_row():
    thursday = snapshot("2026-09-22T23:00:00Z", kickoff="2026-09-25T00:15:00Z")
    poll = snapshot("2026-09-23T00:00:00Z", kickoff="2026-09-25T00:15:00Z")
    late = snapshot("2026-09-23T01:00:00Z", kickoff="2026-09-25T00:15:00Z", prediction=99)
    selected = select_cutoff_rows([thursday, poll, late])
    assert len(selected) == 1
    assert selected.ridge_pred.iloc[0] == 10
    assert not any("practice_count" in c for c in selected)


def test_final_weekly_reports_and_missing_times_cannot_be_live_evidence():
    historical = snapshot()
    historical["evidence_kind"] = "historical_weekly_report"
    with pytest.raises(ValueError, match="observed live"):
        select_cutoff_rows([historical])
    missing = snapshot()
    missing.pop("observed_at")
    with pytest.raises(ValueError, match="availability time"):
        select_cutoff_rows([missing])


def test_comparison_uses_identical_rows_and_projected_actual_components():
    baseline = select_cutoff_rows([snapshot(prediction=10)])
    candidate = select_cutoff_rows([snapshot(prediction=12)])
    actuals = pd.DataFrame(
        {
            "player_id": ["a"],
            "position": ["RB"],
            "season": [2026],
            "week": [3],
            "rushing_yards": [100.0],
            "receiving_yards": [0.0],
            "rushing_tds": [0.0],
            "receiving_tds": [0.0],
            "receptions": [2.0],
            "fumbles_lost": [0.0],
            "fantasy_points": [99.0],
        }
    )
    report = compare_cutoff(baseline, candidate, actuals)
    metrics = report["positions"]["RB"]["ridge_pred"]
    assert metrics["base"]["mae"] == 2.0
    assert metrics["candidate"]["mae"] == 0.0
    assert report["coverage"]["both"] == 1
    assert report["positions"]["RB"]["nn_pred"]["unavailable"] == 1
    cohorts = report["positions"]["RB"]["cohorts"]
    for name in ("injured", "returning", "elite_top24"):
        assert cohorts[name]["models"]["ridge_pred"]["delta"]["mae"] == -2.0
        assert cohorts[name]["sparse"] is True
    assert cohorts["weekly_reference_top24"]["status"] == "unavailable"
    assert cohorts["rest_only"]["n"] == 0
    disagreed = candidate.copy()
    disagreed["elite_top24"] = False
    mismatch = compare_cutoff(baseline, disagreed, actuals)
    assert mismatch["positions"]["RB"]["cohorts"]["elite_top24"]["unknown_or_disagreed"] == 1
    missing = compare_cutoff(baseline, candidate, actuals.drop(columns="fumbles_lost"))
    assert missing["missing_actuals"] == 1
    assert missing["positions"]["RB"]["ridge_pred"]["n"] == 0


def test_empty_archives_report_no_evidence():
    rows = select_cutoff_rows([])
    actuals = pd.DataFrame(columns=["player_id", "season", "week", "position"])
    report = compare_cutoff(rows, rows, actuals)
    assert report["positions"] == {}
