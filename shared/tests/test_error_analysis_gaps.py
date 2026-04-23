"""Additional coverage for the printing + edge-case branches of
``shared/error_analysis.py``.

The existing ``test_error_analysis.py`` covers stratification + metric
computation. This file fills in the table-printing helpers
(``print_stratified_table``, ``print_top_error_sources``), the
missing-column / missing-model early-returns inside
``run_stratified_analysis`` / ``plot_error_by_stratum`` / ``plot_bias_heatmap``,
and the empty-match branches in ``plot_td_zero_vs_scored``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.error_analysis import (
    find_top_error_sources,
    plot_bias_heatmap,
    plot_error_by_stratum,
    plot_td_zero_vs_scored,
    print_stratified_table,
    print_top_error_sources,
    run_stratified_analysis,
)


# --------------------------------------------------------------------------
# Helper: build a "results" dict in the same shape the stratified analysis
# produces (keyed stratum -> model -> target -> metrics_df).
# --------------------------------------------------------------------------


def _metrics_df(buckets: list[str], maes: list[float]) -> pd.DataFrame:
    """DataFrame that mirrors compute_stratum_metrics output:
    (bucket_col, n, mae, rmse, bias)."""
    return pd.DataFrame(
        {
            "bucket": buckets,
            "n": [25] * len(buckets),
            "mae": maes,
            "rmse": [m * 1.3 for m in maes],
            "bias": [0.5 * i - 1.0 for i in range(len(buckets))],
        }
    )


@pytest.fixture()
def _results():
    """Two-stratum / two-model / two-target results dict."""
    return {
        "snap_bucket": {
            "Ridge": {
                "rushing_yards": _metrics_df(["bench", "starter"], [15.0, 25.0]),
                "rushing_tds": _metrics_df(["bench", "starter"], [0.4, 0.6]),
            },
            "NN": {
                "rushing_yards": _metrics_df(["bench", "starter"], [12.0, 22.0]),
                "rushing_tds": _metrics_df(["bench", "starter"], [0.35, 0.55]),
            },
        },
        "home_away": {
            "Ridge": {
                "rushing_yards": _metrics_df(["home", "away"], [18.0, 20.0]),
            },
        },
    }


# --------------------------------------------------------------------------
# print_stratified_table
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_print_stratified_table_emits_all_strata(_results, capsys):
    print_stratified_table(_results, "Ridge", "rushing_yards")
    out = capsys.readouterr().out
    assert "Error Stratification: Ridge — rushing_yards" in out
    # Buckets from both strata should appear.
    assert "bench" in out
    assert "home" in out


@pytest.mark.unit
def test_print_stratified_table_skips_unknown_model_or_target(_results, capsys):
    """Missing model or target → the stratum silently skips (no crash, no row)."""
    print_stratified_table(_results, "MissingModel", "rushing_yards")
    out = capsys.readouterr().out
    # Header printed but no data rows (no stratum matched MissingModel).
    assert "Error Stratification" in out
    assert "bench" not in out


@pytest.mark.unit
def test_print_stratified_table_skips_target_not_in_model(_results, capsys):
    """Model exists but target doesn't → skip."""
    # NN has rushing_yards + rushing_tds but not passing_yards
    print_stratified_table(_results, "NN", "passing_yards")
    out = capsys.readouterr().out
    assert "passing_yards" in out  # header
    assert "bench" not in out  # no data rows


# --------------------------------------------------------------------------
# print_top_error_sources
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_print_top_error_sources_table(_results, capsys):
    sources = find_top_error_sources(_results, "Ridge", metric="mae", top_k=5, min_n=1)
    print_top_error_sources(sources, "Ridge")
    out = capsys.readouterr().out
    assert "Top Error Sources: Ridge" in out
    assert "rushing_yards" in out or "rushing_tds" in out


# --------------------------------------------------------------------------
# find_top_error_sources — skip-model branch
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_find_top_error_sources_skips_strata_missing_the_model(_results):
    """home_away only has Ridge; querying for NN → only snap_bucket rows returned."""
    out = find_top_error_sources(_results, "NN", metric="mae", top_k=100, min_n=1)
    # Only snap_bucket should contribute rows.
    assert all(r["stratum"] == "snap_bucket" for r in out)


# --------------------------------------------------------------------------
# run_stratified_analysis — skip-column branches
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_run_stratified_analysis_skips_missing_pred_or_actual_cols():
    """Missing pred column → skip; missing actual column → skip; stratum with
    <2 unique values → skip."""
    df = pd.DataFrame(
        {
            "snap_bucket": ["bench", "starter"] * 10,
            "flat_stratum": ["x"] * 20,  # <2 unique → skipped
            "rushing_yards": [10.0] * 20,
            "ridge_rush_yds": [12.0] * 20,
            # missing: nn_rush_yds (pred)
            # missing: rushing_tds (actual)
        }
    )
    target_cols = {"rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds"}
    model_pred_cols = {
        "Ridge": {"rushing_yards": "ridge_rush_yds", "rushing_tds": "ridge_rush_tds"},
        "NN": {"rushing_yards": "nn_rush_yds"},
    }

    out = run_stratified_analysis(
        df,
        target_cols=target_cols,
        model_pred_cols=model_pred_cols,
        strata_cols=["snap_bucket", "flat_stratum", "nope"],
    )
    # snap_bucket present; flat_stratum skipped (single value); nope skipped (absent).
    assert "snap_bucket" in out
    assert "flat_stratum" not in out
    assert "nope" not in out
    # Under Ridge, rushing_tds gets skipped (no ridge_rush_tds column).
    assert "rushing_tds" not in out["snap_bucket"].get("Ridge", {})
    # Under NN, rushing_tds isn't even listed in pred_map → skipped.
    assert "rushing_tds" not in out["snap_bucket"].get("NN", {})


# --------------------------------------------------------------------------
# plot_error_by_stratum
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_error_by_stratum_happy_path(_results, tmp_path):
    save_path = tmp_path / "err.png"
    plot_error_by_stratum(
        _results, "Ridge", "snap_bucket", ["rushing_yards", "rushing_tds"], str(save_path)
    )
    assert save_path.exists()


@pytest.mark.unit
def test_plot_error_by_stratum_noop_on_missing_stratum(_results, tmp_path):
    """Unknown stratum → early return, no file written."""
    save_path = tmp_path / "missing.png"
    plot_error_by_stratum(_results, "Ridge", "missing_stratum", ["rushing_yards"], str(save_path))
    assert not save_path.exists()


@pytest.mark.unit
def test_plot_error_by_stratum_noop_on_missing_model(_results, tmp_path):
    save_path = tmp_path / "noop.png"
    plot_error_by_stratum(
        _results, "NonExistentModel", "snap_bucket", ["rushing_yards"], str(save_path)
    )
    assert not save_path.exists()


@pytest.mark.unit
def test_plot_error_by_stratum_noop_when_no_targets_match(_results, tmp_path):
    """Target names don't intersect the model's recorded targets → data dict
    ends up empty → early return."""
    save_path = tmp_path / "empty.png"
    plot_error_by_stratum(_results, "Ridge", "snap_bucket", ["nonexistent_target"], str(save_path))
    assert not save_path.exists()


# --------------------------------------------------------------------------
# plot_bias_heatmap
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_bias_heatmap_happy_path(_results, tmp_path):
    save_path = tmp_path / "heat.png"
    plot_bias_heatmap(
        _results,
        "Ridge",
        ["snap_bucket", "home_away"],
        ["rushing_yards", "rushing_tds"],
        str(save_path),
    )
    assert save_path.exists()


@pytest.mark.unit
def test_plot_bias_heatmap_skips_missing_strata_and_targets(_results, tmp_path):
    """Strata or targets that don't exist in results → skipped; row_labels may
    end up empty → early return, no file."""
    save_path = tmp_path / "heat.png"
    plot_bias_heatmap(
        _results,
        "Ridge",
        ["nonexistent_stratum"],
        ["rushing_yards"],
        str(save_path),
    )
    assert not save_path.exists()


@pytest.mark.unit
def test_plot_bias_heatmap_skips_missing_model(_results, tmp_path):
    save_path = tmp_path / "heat.png"
    plot_bias_heatmap(
        _results,
        "NonExistentModel",
        ["snap_bucket", "home_away"],
        ["rushing_yards"],
        str(save_path),
    )
    assert not save_path.exists()


# --------------------------------------------------------------------------
# plot_td_zero_vs_scored
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_td_zero_vs_scored_happy_path(tmp_path):
    df = pd.DataFrame(
        {
            "actual": [0, 0, 0, 1, 2, 1, 0, 1, 0, 2],
            "pred": [0.1, 0.2, 0.0, 0.9, 1.8, 1.1, 0.3, 0.95, 0.15, 2.2],
        }
    )
    save_path = tmp_path / "td.png"
    plot_td_zero_vs_scored(df, "pred", "actual", str(save_path), title="Test")
    assert save_path.exists()


@pytest.mark.unit
def test_plot_td_zero_vs_scored_returns_early_when_cols_missing(tmp_path):
    """Missing actual or pred column → early return."""
    df = pd.DataFrame({"actual": [0, 1]})  # no 'pred'
    save_path = tmp_path / "noop.png"
    plot_td_zero_vs_scored(df, "pred", "actual", str(save_path))
    assert not save_path.exists()


@pytest.mark.unit
def test_plot_td_zero_vs_scored_handles_empty_mask(tmp_path):
    """When no 0-TD rows exist, the zero-mask branch continues without error."""
    df = pd.DataFrame({"actual": [1, 2, 3], "pred": [0.9, 2.1, 2.8]})
    save_path = tmp_path / "only_scored.png"
    plot_td_zero_vs_scored(df, "pred", "actual", str(save_path))
    # File still written (the 1+ TD panel has data).
    assert save_path.exists()
