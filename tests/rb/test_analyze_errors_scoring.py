"""RB diagnostic totals use the same raw components as model predictions."""

import sys

import numpy as np
import pandas as pd
import pytest

from src.shared.comparison_scoring import score_actual_components

pytestmark = pytest.mark.unit


def _frame():
    frame = pd.DataFrame(
        {
            "rushing_tds": [1.0, 0.0],
            "receiving_tds": [0.0, 1.0],
            "rushing_yards": [45.0, 32.0],
            "receiving_yards": [10.0, 24.0],
            "receptions": [2.0, 3.0],
            "fumbles_lost": [0.0, 1.0],
        }
    )
    frame["pred_nn_total"] = score_actual_components(frame, "RB")
    frame["fantasy_points"] = frame["pred_nn_total"] + [4.12, 0.0]
    return frame


def _run_diagnostic(frame, monkeypatch):
    from src.rb import analyze_errors as cli

    captured = {}
    analyze = cli.run_stratified_analysis

    def strata(df, targets):
        for name in cli.STRATA_COLS:
            df[name] = "all"

    def inspect(df, models, targets, strata):
        captured["actual_total"] = df[targets["total"]].copy()
        captured["metrics"] = analyze(df, models, targets, strata)
        return captured["metrics"]

    monkeypatch.setattr(sys, "argv", ["analyze_errors", "--no-plots"])
    monkeypatch.setattr(cli, "run", lambda: {"test_df": frame})
    monkeypatch.setattr(cli, "add_stratification_columns", strata)
    monkeypatch.setattr(cli, "run_stratified_analysis", inspect)
    monkeypatch.setattr(cli, "print_stratified_table", lambda *args: None)
    monkeypatch.setattr(cli, "find_top_error_sources", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli, "print_top_error_sources", lambda *args: None)
    cli.main()
    return captured


def test_perfect_predictions_ignore_unprojected_passing_points(monkeypatch):
    frame = _frame()
    original = frame.copy(deep=True)
    result = _run_diagnostic(frame, monkeypatch)
    np.testing.assert_allclose(result["actual_total"], frame["pred_nn_total"])
    for by_model in result["metrics"].values():
        assert by_model["NN"]["total"]["mae"].eq(0).all()
    pd.testing.assert_frame_equal(frame, original)


def test_ordinary_totals_are_unchanged(monkeypatch):
    frame = _frame()
    frame["fantasy_points"] = frame["pred_nn_total"]
    result = _run_diagnostic(frame, monkeypatch)
    np.testing.assert_allclose(result["actual_total"], frame["fantasy_points"])


@pytest.mark.parametrize("missing_column", [False, True])
def test_missing_actual_component_stays_unavailable(monkeypatch, missing_column):
    frame = _frame()
    if missing_column:
        frame = frame.drop(columns="receiving_yards")
    else:
        frame.loc[0, "receiving_yards"] = np.nan
    result = _run_diagnostic(frame, monkeypatch)
    if missing_column:
        assert result["actual_total"].isna().all()
    else:
        assert pd.isna(result["actual_total"].iloc[0])
        assert result["actual_total"].iloc[1] == pytest.approx(frame["pred_nn_total"].iloc[1])
