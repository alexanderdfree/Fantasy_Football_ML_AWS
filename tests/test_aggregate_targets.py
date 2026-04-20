"""Parity tests: predictions_to_fantasy_points on true stats == compute_fantasy_points."""

import numpy as np
import pandas as pd
import pytest

from shared.aggregate_targets import predictions_to_fantasy_points
from src.config import SCORING_PPR
from src.data.loader import compute_fantasy_points


def _sample_df():
    return pd.DataFrame(
        {
            "passing_yards": [300.0, 0.0, 250.0],
            "passing_tds": [2, 0, 1],
            "interceptions": [1, 0, 0],
            "rushing_yards": [20.0, 80.0, 10.0],
            "rushing_tds": [0, 1, 0],
            "receiving_yards": [0.0, 40.0, 75.0],
            "receiving_tds": [0, 0, 1],
            "receptions": [0, 3, 6],
            "sack_fumbles_lost": [1, 0, 0],
            "rushing_fumbles_lost": [0, 1, 0],
            "receiving_fumbles_lost": [0, 0, 1],
        }
    )


def _fumbles_lost(df, pos):
    if pos == "QB":
        return df["sack_fumbles_lost"] + df["rushing_fumbles_lost"]
    return df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"]


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_aggregator_parity_with_compute_fantasy_points(pos):
    df = _sample_df()

    # Build a preds_dict whose values ARE the true raw stats — aggregating
    # should reproduce fantasy_points exactly under PPR.
    preds = {
        "passing_yards": df["passing_yards"].values,
        "rushing_yards": df["rushing_yards"].values,
        "receiving_yards": df["receiving_yards"].values,
        "passing_tds": df["passing_tds"].values,
        "rushing_tds": df["rushing_tds"].values,
        "receiving_tds": df["receiving_tds"].values,
        "receptions": df["receptions"].values,
        "interceptions": df["interceptions"].values,
        "fumbles_lost": _fumbles_lost(df, pos).values,
    }

    actual = predictions_to_fantasy_points(pos, preds, "ppr")

    # Build a DF that mirrors the position's target subset (drop stats the position
    # doesn't predict, so compute_fantasy_points excludes them too — the per-position
    # aggregator also excludes them via POSITION_TARGET_MAP).
    truth_df = df.copy()
    truth_df["fumbles_lost"] = _fumbles_lost(truth_df, pos)
    if pos == "QB":
        truth_df["receptions"] = 0
        truth_df["receiving_yards"] = 0
        truth_df["receiving_tds"] = 0
        # QB fumbles_lost = sack + rushing only; zero receiving so compute_fantasy_points agrees.
        truth_df["receiving_fumbles_lost"] = 0
    if pos in ("WR", "TE"):
        truth_df["passing_yards"] = 0
        truth_df["passing_tds"] = 0
        truth_df["interceptions"] = 0
        truth_df["rushing_yards"] = 0
        truth_df["rushing_tds"] = 0
        # WR/TE fumbles_lost = rushing + receiving only; zero sack.
        truth_df["sack_fumbles_lost"] = 0
    if pos == "RB":
        truth_df["passing_yards"] = 0
        truth_df["passing_tds"] = 0
        truth_df["interceptions"] = 0
        # RB fumbles_lost = rushing + receiving only; zero sack.
        truth_df["sack_fumbles_lost"] = 0

    expected = compute_fantasy_points(truth_df, SCORING_PPR).values
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def test_aggregator_ignores_total_key():
    preds = {
        "receiving_yards": np.array([10.0]),
        "receptions": np.array([1.0]),
        "receiving_tds": np.array([0.0]),
        "fumbles_lost": np.array([0.0]),
        "total": np.array([999.0]),  # should be ignored
    }
    out = predictions_to_fantasy_points("WR", preds, "ppr")
    assert np.isclose(out[0], 1.0 + 1.0 + 0.0 + 0.0)  # 10×0.1 + 1×1.0


def test_aggregator_unknown_position():
    with pytest.raises(ValueError):
        predictions_to_fantasy_points("ZZ", {"x": np.zeros(1)}, "ppr")
