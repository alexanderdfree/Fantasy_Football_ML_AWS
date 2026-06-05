"""Unit tests for the ATTN week-by-week / subgroup accuracy helpers.

Import-smoke (so signature drift fails the unit shard, not PR review) plus pure
aggregation tests on a synthetic test_df — no pipeline run, no torch import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import attn_weekly_accuracy as awa

pytestmark = pytest.mark.unit


def _synthetic() -> pd.DataFrame:
    """Two positions, two seeds, four weeks; ATTN deliberately best on QB."""
    rng = np.random.default_rng(0)
    rows = []
    for pos in ["QB", "RB"]:
        for seed in [42, 123]:
            for wk in [1, 5, 14, 17]:
                for _ in range(8):
                    actual = float(rng.uniform(0, 30))
                    rows.append(
                        {
                            "position": pos,
                            "seed": seed,
                            "player_id": f"{pos}{rng.integers(1000)}",
                            "season": 2024,
                            "week": wk,
                            "fantasy_points": actual,
                            # ATTN closest to actual on QB, Ridge worst.
                            "pred_attn_nn_total": actual + (0.5 if pos == "QB" else 3.0),
                            "pred_ridge_total": actual + 4.0,
                            "pred_nn_total": actual + 2.0,
                            "pred_lgbm_total": actual + 1.5,
                        }
                    )
    return pd.DataFrame(rows)


def test_import_smoke_exposes_cli_and_constants():
    assert hasattr(awa, "main")
    assert awa.ATTN == "Attention NN"
    assert awa.ATTN in awa.MODELS
    assert set(awa.PEERS) == {"Ridge", "NN", "LightGBM"}


def test_models_in_only_returns_present_columns():
    df = _synthetic().drop(columns=["pred_lgbm_total"])
    models = awa._models_in(df)
    assert "LightGBM" not in models
    assert "Attention NN" in models


def test_metric_by_averages_over_seeds():
    df = _synthetic()
    agg = awa._metric_by(df, ["position"], "mae")
    # One row per (position, model).
    assert len(agg) == df["position"].nunique() * len(awa._models_in(df))
    qb_attn = agg[(agg["position"] == "QB") & (agg["model"] == awa.ATTN)]
    # ATTN is a constant +0.5 from actual on QB -> MAE ~= 0.5, std ~= 0 over seeds.
    assert qb_attn["mean"].iloc[0] == pytest.approx(0.5, abs=1e-9)
    assert qb_attn["std"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_attn_vs_best_peer_flags_win_and_loss():
    df = _synthetic()
    agg = awa._metric_by(df, ["position"], "mae")
    cmp = awa._attn_vs_best_peer(agg, "position")
    # QB: ATTN (0.5) beats best peer (LightGBM 1.5) -> win, negative delta.
    assert bool(cmp.loc["QB", "attn_wins"]) is True
    assert cmp.loc["QB", "attn_minus_peer"] < 0
    # RB: ATTN (3.0) loses to best peer (LightGBM 1.5) -> no win.
    assert bool(cmp.loc["RB", "attn_wins"]) is False
    assert cmp.loc["RB", "best_peer"] == "LightGBM"


def test_fmt_handles_nan():
    assert awa._fmt(float("nan")) == "nan"
    assert awa._fmt(1.23456) == "1.235"
