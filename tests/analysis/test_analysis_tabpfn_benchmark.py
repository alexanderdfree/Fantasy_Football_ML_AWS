"""Unit + smoke tests for src/analysis/analysis_tabpfn_benchmark.py.

The run path (``run_position`` / ``load_rotowire``) trains pipelines, loads the
optional TabPFN weights, and hits the network — out of scope for unit tests. These
exercise the pure metric + formatting helpers on synthetic frames and import-smoke
the module so a signature break (e.g. ``compute_metrics``) fails the unit shard
rather than only at PR time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_tabpfn_benchmark as mod

pytestmark = pytest.mark.unit


def _synth_test_df(n_players: int = 10, n_weeks: int = 4, seed: int = 0) -> pd.DataFrame:
    # Grid of unique (player_id, season, week) — mirrors a real test_df (one row per
    # player-game-week), so the RotoWire inner-join stays 1:1.
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        for wk in range(1, n_weeks + 1):
            actual = float(rng.gamma(2.0, 5.0))
            rows.append(
                {
                    "player_id": f"P{pid}",
                    "season": 2025,
                    "week": wk,
                    "fantasy_points": actual,
                    "pred_ridge_total": actual + float(rng.normal(0, 3)),
                    "pred_tabpfn_total": actual + float(rng.normal(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def test_position_metrics_structure():
    df = _synth_test_df()
    pm = mod.position_metrics(df, top_n=5)
    assert set(pm) == {"Ridge", "TabPFN"}
    for d in pm.values():
        assert {"mae", "rmse", "r2", "n"} <= set(d["regular"])
        assert {"r2", "rmse", "n"} <= set(d["topN"])
        assert set(d["bands"]) == {"Q1", "Q2", "Q3", "Q4"}
        assert d["regular"]["n"] == len(df)


def test_quartile_bands_survive_ties():
    # Rank-based qcut must yield 4 bands even with heavy ties (K/DST low-variance).
    s = pd.Series([1, 1, 1, 2, 2, 2, 3, 3])
    assert set(mod._quartile_bands(s).unique()) == {"Q1", "Q2", "Q3", "Q4"}


def test_rotowire_matched_metrics_and_alignment():
    df = _synth_test_df()
    expert = df[["player_id", "season", "week"]].copy()
    expert["expert_pred_total"] = df["fantasy_points"].to_numpy() + 1.0  # constant +1 error
    bm = mod.rotowire_matched_metrics(df, expert, top_n=5)
    assert bm["_n_matched"] == len(df)
    assert "RotoWire" in bm and "Ridge" in bm and "TabPFN" in bm
    assert bm["RotoWire"]["regular"]["mae"] == pytest.approx(1.0, abs=1e-6)


def test_rotowire_no_overlap_returns_zero():
    df = _synth_test_df()
    expert = pd.DataFrame(
        {"player_id": ["NOBODY"], "season": [2025], "week": [1], "expert_pred_total": [5.0]}
    )
    bm = mod.rotowire_matched_metrics(df, expert, top_n=5)
    assert bm == {"_n_matched": 0}


def test_format_report_renders_both_sections():
    df = _synth_test_df()
    a = {"WR": mod.position_metrics(df, top_n=5)}
    expert = df[["player_id", "season", "week"]].copy()
    expert["expert_pred_total"] = df["fantasy_points"].to_numpy() + 1.0
    b = {"WR": mod.rotowire_matched_metrics(df, expert, top_n=5)}
    report = mod.format_report(a, b, top_n=5)
    assert "Section A" in report and "Section B" in report
    assert "WR" in report and "RotoWire" in report
