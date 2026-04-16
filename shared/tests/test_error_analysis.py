"""Tests for shared.error_analysis — stratification, metrics, and plotting."""

import numpy as np
import pandas as pd
import pytest

from shared.error_analysis import (
    add_stratification_columns,
    compute_stratum_metrics,
    run_stratified_analysis,
    find_top_error_sources,
    plot_error_by_stratum,
    plot_bias_heatmap,
    plot_td_zero_vs_scored,
)

TARGETS = ["rushing_floor", "receiving_floor", "td_points"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_df(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "week": np.random.randint(1, 18, size=n),
        "snap_pct": np.random.rand(n) * 100,
        "opp_def_rank_vs_pos": np.random.randint(1, 33, size=n),
        "is_home": np.random.choice([0, 1], size=n),
        "td_points": np.random.choice([0.0, 6.0, 12.0], size=n, p=[0.6, 0.3, 0.1]),
        "rolling_std_fantasy_points_L3": np.random.rand(n) * 5,
        "fantasy_points": np.random.rand(n) * 20,
        "pred_total": np.random.rand(n) * 20,
    })


# ---------------------------------------------------------------------------
# add_stratification_columns
# ---------------------------------------------------------------------------

class TestAddStratificationColumns:
    def test_adds_all_bucket_columns(self):
        df = _make_test_df()
        result = add_stratification_columns(df, TARGETS)
        for col in ["snap_bucket", "opp_tier", "week_phase", "td_bucket",
                     "volatility_q", "home_away"]:
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
        df = pd.DataFrame({
            "td_points": [6.0, 0.0],
            "week": [5, 5],
        })
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

class TestComputeStratumMetrics:
    def test_output_columns(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "actual": [1.0, 2.0, 3.0, 4.0],
            "pred": [1.5, 2.5, 3.5, 4.5],
        })
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert set(result.columns) == {"group", "n", "mae", "rmse", "bias"}

    def test_perfect_predictions(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B"],
            "actual": [1.0, 2.0, 3.0],
            "pred": [1.0, 2.0, 3.0],
        })
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert (result["mae"] == 0).all()
        assert (result["rmse"] == 0).all()
        assert (result["bias"] == 0).all()

    def test_positive_bias(self):
        df = pd.DataFrame({
            "group": ["A", "A"],
            "actual": [1.0, 2.0],
            "pred": [3.0, 4.0],
        })
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        assert result["bias"].iloc[0] == pytest.approx(2.0)

    def test_handles_nan(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "actual": [1.0, np.nan, 3.0],
            "pred": [1.5, 2.5, np.nan],
        })
        result = compute_stratum_metrics(df, "actual", "pred", "group")
        # Only the first row should be counted (both actual and pred non-NaN)
        assert result["n"].iloc[0] == 1


# ---------------------------------------------------------------------------
# run_stratified_analysis
# ---------------------------------------------------------------------------

class TestRunStratifiedAnalysis:
    def test_output_structure(self):
        df = _make_test_df()
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away"]
        result = run_stratified_analysis(df, model_pred_cols, target_cols, strata)
        for stratum in result:
            assert "Ridge" in result[stratum]
            assert "total" in result[stratum]["Ridge"]

    def test_skips_missing_columns(self):
        df = _make_test_df()
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        result = run_stratified_analysis(df, model_pred_cols, target_cols, ["nonexistent_col"])
        assert "nonexistent_col" not in result

    def test_skips_low_cardinality(self):
        df = _make_test_df()
        df["constant_col"] = "same"  # single unique value
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        result = run_stratified_analysis(df, model_pred_cols, target_cols, ["constant_col"])
        assert "constant_col" not in result


# ---------------------------------------------------------------------------
# find_top_error_sources
# ---------------------------------------------------------------------------

class TestFindTopErrorSources:
    def _make_results(self):
        df = _make_test_df(200)
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away", "td_bucket"]
        return run_stratified_analysis(df, model_pred_cols, target_cols, strata)

    def test_returns_sorted_by_metric(self):
        results = self._make_results()
        sources = find_top_error_sources(results, "Ridge", metric="mae", top_k=5, min_n=1)
        if len(sources) >= 2:
            assert sources[0]["mae"] >= sources[1]["mae"]

    def test_min_n_filter(self):
        results = self._make_results()
        sources = find_top_error_sources(results, "Ridge", min_n=9999)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# Plotting (non-crash tests)
# ---------------------------------------------------------------------------

class TestPlotting:
    def _make_results_and_df(self):
        df = _make_test_df(200)
        add_stratification_columns(df, TARGETS)
        model_pred_cols = {"Ridge": {"total": "pred_total"}}
        target_cols = {"total": "fantasy_points"}
        strata = ["week_phase", "home_away"]
        results = run_stratified_analysis(df, model_pred_cols, target_cols, strata)
        return results, df

    def test_plot_error_by_stratum(self, tmp_path):
        results, _ = self._make_results_and_df()
        save_path = str(tmp_path / "error.png")
        plot_error_by_stratum(results, "Ridge", "week_phase", ["total"], save_path)
        assert (tmp_path / "error.png").exists()

    def test_plot_bias_heatmap(self, tmp_path):
        results, _ = self._make_results_and_df()
        save_path = str(tmp_path / "bias.png")
        plot_bias_heatmap(results, "Ridge", ["week_phase", "home_away"], ["total"], save_path)
        assert (tmp_path / "bias.png").exists()

    def test_plot_td_zero_vs_scored(self, tmp_path):
        _, df = self._make_results_and_df()
        save_path = str(tmp_path / "td.png")
        plot_td_zero_vs_scored(df, "pred_total", "td_points", save_path)
        assert (tmp_path / "td.png").exists()
