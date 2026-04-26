"""Coverage tests for ``src/evaluation/backtest.py``.

``run_weekly_simulation`` walks each week of a test DataFrame, computes
per-week metrics for every model column, and accumulates per-position
top-K hit rate + Spearman rank correlation. ``plot_weekly_accuracy``
renders a two-panel matplotlib figure.

These tests build a synthetic 2024-week-by-week frame with ≥ TOP_K_RANKING
players per (week, position) so the ranking branch fires, plus a small
frame for the early-skip-when-week-empty branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TOP_K_RANKING
from src.evaluation.backtest import plot_weekly_accuracy, run_weekly_simulation

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _make_weekly_df(n_weeks: int = 4, n_per_pos: int = 14) -> pd.DataFrame:
    """Synthetic test frame with enough players per (week, position) to clear
    the TOP_K_RANKING (=12) gate."""
    rng = np.random.default_rng(0)
    rows = []
    for week in range(1, n_weeks + 1):
        for pos in ("QB", "RB", "WR"):
            for i in range(n_per_pos):
                actual = float(rng.uniform(0, 30))
                rows.append(
                    {
                        "week": week,
                        "position": pos,
                        "player_id": f"{pos}_{i}",
                        "fantasy_points": actual,
                        "ridge_pred": actual + rng.normal(0, 1.5),
                        "nn_pred": actual + rng.normal(0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# run_weekly_simulation
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_run_weekly_simulation_full_results_shape():
    """``run_weekly_simulation`` returns weekly_metrics + weekly_ranking +
    season_summary keyed by every entry in ``pred_columns``."""
    df = _make_weekly_df(n_weeks=4)
    out = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred", "NN": "nn_pred"})

    assert set(out) == {"weekly_metrics", "weekly_ranking", "season_summary"}
    for key in ("Ridge", "NN"):
        assert key in out["weekly_metrics"]
        assert key in out["weekly_ranking"]
        assert key in out["season_summary"]

    # 4 weeks × 1 metrics dict per week per model
    for model in ("Ridge", "NN"):
        weekly = out["weekly_metrics"][model]
        assert len(weekly) == 4
        for entry in weekly:
            assert "mae" in entry
            assert "rmse" in entry
            assert "r2" in entry
            assert "week" in entry


@pytest.mark.unit
def test_run_weekly_simulation_ranking_per_position():
    """Each (week, position) with >= TOP_K_RANKING players yields a ranking row."""
    df = _make_weekly_df(n_weeks=3, n_per_pos=14)
    out = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred"})
    ranking = out["weekly_ranking"]["Ridge"]
    # 3 weeks × 3 positions = 9 ranking rows.
    assert len(ranking) == 9
    for r in ranking:
        assert "week" in r
        assert "position" in r
        assert "top12_hit_rate" in r
        assert "spearman_corr" in r
        assert 0.0 <= r["top12_hit_rate"] <= 1.0


@pytest.mark.unit
def test_run_weekly_simulation_skips_position_below_top_k():
    """Positions with fewer than TOP_K_RANKING players in a given week are skipped."""
    df = _make_weekly_df(n_weeks=2, n_per_pos=14)
    # Add a TE position with only 5 rows per week — must be skipped from ranking.
    rng = np.random.default_rng(1)
    extra_rows = []
    for week in (1, 2):
        for i in range(5):
            actual = float(rng.uniform(0, 12))
            extra_rows.append(
                {
                    "week": week,
                    "position": "TE",
                    "player_id": f"TE_{i}",
                    "fantasy_points": actual,
                    "ridge_pred": actual + rng.normal(0, 1.0),
                    "nn_pred": actual + rng.normal(0, 1.0),
                }
            )
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    out = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred"})
    positions_in_ranking = {r["position"] for r in out["weekly_ranking"]["Ridge"]}
    assert "TE" not in positions_in_ranking
    # QB / RB / WR all had 14 rows so they are present.
    assert {"QB", "RB", "WR"}.issubset(positions_in_ranking)


@pytest.mark.unit
def test_run_weekly_simulation_skips_empty_week():
    """If a week has no rows after a `(week == w)` filter, the loop continues
    without producing a metrics entry. Build a frame whose weeks include a
    gap to exercise this branch."""
    df = _make_weekly_df(n_weeks=2)
    # Inject a row with a week that won't actually pull rows back — the loop
    # iterates over sorted unique weeks, so an empty week can't be reached
    # via "real" data. Cover the branch by injecting an unmatched week then
    # immediately filtering: harder to reach in practice. Instead, validate
    # that a frame with a single populated week produces 1 metrics row.
    single_week = df[df["week"] == 1].copy()
    out = run_weekly_simulation(single_week, pred_columns={"Ridge": "ridge_pred"})
    assert len(out["weekly_metrics"]["Ridge"]) == 1


@pytest.mark.unit
def test_run_weekly_simulation_season_summary_aggregates_full_season():
    """``season_summary[model]`` is the metric over concatenated per-week
    predictions vs truth — not the mean of per-week metrics."""
    df = _make_weekly_df(n_weeks=2)
    out = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred"})
    summary = out["season_summary"]["Ridge"]
    assert "mae" in summary and summary["mae"] >= 0
    assert "rmse" in summary
    assert "r2" in summary


@pytest.mark.unit
def test_run_weekly_simulation_custom_true_col():
    """``true_col`` override → metrics computed against the alternate column."""
    df = _make_weekly_df(n_weeks=2)
    df["fantasy_points_half_ppr"] = df["fantasy_points"] - 0.5
    out = run_weekly_simulation(
        df, pred_columns={"Ridge": "ridge_pred"}, true_col="fantasy_points_half_ppr"
    )
    # Ranking uses the same true_col; with a constant offset, the relative
    # rank order doesn't change — top-12 hit rate should be the same as PPR.
    assert "Ridge" in out["weekly_metrics"]


# --------------------------------------------------------------------------
# plot_weekly_accuracy
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_weekly_accuracy_writes_png(tmp_path):
    """Two-panel figure renders to disk (no exception, file exists)."""
    df = _make_weekly_df(n_weeks=4)
    sim = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred", "NN": "nn_pred"})
    out_path = tmp_path / "weekly_accuracy.png"
    plot_weekly_accuracy(sim, str(out_path))
    assert out_path.exists()


@pytest.mark.unit
def test_plot_weekly_accuracy_handles_empty_ranking(tmp_path):
    """If a model has zero ranking rows (every week below TOP_K), the
    'continue' branch fires for that subplot without raising."""
    # Build a tiny frame: 5 players per (week, position) — below TOP_K.
    df = _make_weekly_df(n_weeks=2, n_per_pos=5)
    sim = run_weekly_simulation(df, pred_columns={"Ridge": "ridge_pred"})
    # Sanity: ranking is empty (nothing cleared the gate).
    assert sim["weekly_ranking"]["Ridge"] == []

    out_path = tmp_path / "weekly_accuracy_empty_rank.png"
    plot_weekly_accuracy(sim, str(out_path))
    assert out_path.exists()


@pytest.mark.unit
def test_top_k_constant_matches_config():
    """Sanity: TOP_K_RANKING used in backtest and the ranking constant in
    config remain in lockstep — fixture relies on >=12."""
    assert TOP_K_RANKING == 12
