"""Tests for src.shared.error_analysis — stratification, metrics, and plotting."""

import numpy as np
import pandas as pd
import pytest

from src.shared.error_analysis import (
    add_stratification_columns,
    compute_stratum_metrics,
    find_top_error_sources,
    plot_bias_heatmap,
    plot_error_by_stratum,
    plot_td_zero_vs_scored,
    run_stratified_analysis,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]


# ---------------------------------------------------------------------------
# add_stratification_columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAddStratificationColumns:
    def test_adds_all_bucket_columns(self, error_df_factory):
        df = error_df_factory()
        result = add_stratification_columns(df, TARGETS)
        for col in [
            "snap_bucket",
            "opp_tier",
            "week_phase",
            "td_bucket",
            "volatility_q",
            "home_away",
        ]:
            assert col in result.columns

    def test_snap_bucket_percentage_format(self):
        df = pd.DataFrame({"snap_pct": [80.0], "week": [5]})
        result = add_stratification_columns(df, TARGETS)
        assert result["snap_bucket"].iloc[0] == "starter"

    def test_snap_bucket_decimal_format(self):
        df = pd.DataFrame({"snap_pct": [0.8], "week": [5]})
        result = add_stratification_columns(df, TARGETS)
        assert result["snap_bucket"].iloc[0] == "starter"

    def test_td_bucket_values(self):
        df = pd.DataFrame(
            {
                "rushing_tds": [1.0, 0.0],
                "week": [5, 5],
            }
        )
        result = add_stratification_columns(df, TARGETS)
        assert result["td_bucket"].iloc[0] == "has_td"
        assert result["td_bucket"].iloc[1] == "zero_td"

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"week": [5, 10]})
        result = add_stratification_columns(df, TARGETS)
        assert (result["snap_bucket"] == "unknown").all()
        assert (result["opp_tier"] == "unknown").all()

    def test_home_away_from_is_home(self):
        df = pd.DataFrame({"is_home": [1, 0], "week": [1, 2]})
        result = add_stratification_columns(df, TARGETS)
        assert result["home_away"].iloc[0] == "home"
        assert result["home_away"].iloc[1] == "away"


# ---------------------------------------------------------------------------
# compute_stratum_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeStratumMetrics:
    def test_output_columns(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "actual": [1.0, 2.0, 3.0, 4.0],
                "pred": [1.5, 2.5, 3.5, 4.5],
            }
        )
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert set(result.columns) == {"group", "n", "mae", "rmse", "bias"}

    def test_perfect_predictions(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B"],
                "actual": [1.0, 2.0, 3.0],
                "pred": [1.0, 2.0, 3.0],
            }
        )
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert (result["mae"] == 0).all()
        assert (result["rmse"] == 0).all()
        assert (result["bias"] == 0).all()

    def test_positive_bias(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A"],
                "actual": [1.0, 2.0],
                "pred": [3.0, 4.0],
            }
        )
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert result["bias"].iloc[0] == pytest.approx(2.0)

    def test_handles_nan(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "actual": [1.0, np.nan, 3.0],
                "pred": [1.5, 2.5, np.nan],
            }
        )
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        # Only the first row should be counted (both actual and pred non-NaN)
        assert result["n"].iloc[0] == 1


# ---------------------------------------------------------------------------
# run_stratified_analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunStratifiedAnalysis:
    def test_output_structure(self, error_df_factory):
        df = error_df_factory()
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away"]
        result = run_stratified_analysis(df, model_pred_cols, target_cols, strata)
        for stratum in result:
            assert "Ridge" in result[stratum]
            assert "total" in result[stratum]["Ridge"]

    def test_skips_missing_columns(self, error_df_factory):
        df = error_df_factory()
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        result = run_stratified_analysis(df, model_pred_cols, target_cols, ["nonexistent_col"])
        assert "nonexistent_col" not in result

    def test_skips_low_cardinality(self, error_df_factory):
        df = error_df_factory()
        df["constant_col"] = "same"  # single unique value
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        result = run_stratified_analysis(df, model_pred_cols, target_cols, ["constant_col"])
        assert "constant_col" not in result


# ---------------------------------------------------------------------------
# find_top_error_sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindTopErrorSources:
    @pytest.fixture
    def stratified_results(self, error_df_factory):
        df = error_df_factory(200)
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away", "td_bucket"]
        return run_stratified_analysis(df, model_pred_cols, target_cols, strata)

    def test_returns_sorted_by_metric(self, stratified_results):
        sources = find_top_error_sources(
            stratified_results, "Ridge", metric="mae", top_k=5, min_n=1
        )
        # Pre-condition: the 200-row fixture with 3 strata must yield at least
        # 2 buckets, otherwise the sort assertion below is silently skipped.
        assert len(sources) >= 2, "fixture produced too few strata to test sorting"
        maes = [s["mae"] for s in sources]
        assert maes == sorted(maes, reverse=True)

    def test_min_n_filter(self, stratified_results):
        sources = find_top_error_sources(stratified_results, "Ridge", min_n=9999)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# Plotting (non-crash tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlotting:
    @pytest.fixture
    def results_and_df(self, error_df_factory):
        df = error_df_factory(200)
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away"]
        results = run_stratified_analysis(df, model_pred_cols, target_cols, strata)
        return results, df

    def test_plot_error_by_stratum(self, results_and_df, tmp_path):
        results, _ = results_and_df
        save_path = str(tmp_path / "error.png")
        plot_error_by_stratum(results, "Ridge", "week_phase", ["total"], save_path)
        assert (tmp_path / "error.png").exists()

    def test_plot_bias_heatmap(self, results_and_df, tmp_path):
        results, _ = results_and_df
        save_path = str(tmp_path / "bias.png")
        plot_bias_heatmap(results, "Ridge", ["week_phase", "home_away"], ["total"], save_path)
        assert (tmp_path / "bias.png").exists()

    def test_plot_td_zero_vs_scored(self, results_and_df, tmp_path):
        _, df = results_and_df
        save_path = str(tmp_path / "td.png")
        plot_td_zero_vs_scored(df, "pred_total", "rushing_tds", save_path)
        assert (tmp_path / "td.png").exists()
