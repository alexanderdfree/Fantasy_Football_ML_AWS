"""Parity tests: predictions_to_fantasy_points on true stats == compute_fantasy_points."""

import numpy as np
import pandas as pd
import pytest
import torch

from DST.dst_targets import compute_dst_targets
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


# ---------------------------------------------------------------------------
# DST branch: linear raw-stat coefficients + tier-mapped PA/YA bonuses.
# ---------------------------------------------------------------------------


def _dst_sample_preds_numpy():
    """Sample DST raw-stat preds as numpy arrays — 3 team-weeks."""
    return {
        "def_sacks": np.array([3.0, 0.0, 5.0]),
        "def_ints": np.array([1.0, 0.0, 2.0]),
        "def_fumble_rec": np.array([1.0, 0.0, 1.0]),
        "def_fumbles_forced": np.array([2.0, 1.0, 3.0]),
        "def_safeties": np.array([0.0, 0.0, 1.0]),
        "def_tds": np.array([1.0, 0.0, 2.0]),
        "def_blocked_kicks": np.array([0.0, 0.0, 1.0]),
        "special_teams_tds": np.array([1.0, 0.0, 0.0]),
        "points_allowed": np.array([10.0, 35.0, 0.0]),
        "yards_allowed": np.array([220.0, 420.0, 80.0]),
    }


def test_dst_aggregator_matches_compute_dst_targets_numpy():
    """predictions_to_fantasy_points('DST', ...) on true stats must match compute_dst_targets fantasy_points."""
    preds = _dst_sample_preds_numpy()
    df = pd.DataFrame(preds)
    expected = compute_dst_targets(df)["fantasy_points"].values
    actual = predictions_to_fantasy_points("DST", preds, "ppr")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def test_dst_aggregator_scalar_values():
    """Hand-computed: all 3 rows."""
    preds = _dst_sample_preds_numpy()
    actual = predictions_to_fantasy_points("DST", preds, "ppr")
    # Row 0: 3 + 2 + 2 + 2 + 0 + 6 + 0 + 6 = 21; PA(10)=+4, YA(220)=+2 => 27
    # Row 1: 0 + 0 + 0 + 1 + 0 + 0 + 0 + 0 = 1; PA(35)=-4, YA(420)=-3 => -6
    # Row 2: 5 + 4 + 2 + 3 + 2 + 12 + 2 + 0 = 30; PA(0)=+10, YA(80)=+5 => 45
    np.testing.assert_allclose(actual, [27.0, -6.0, 45.0], atol=1e-9)


def test_dst_aggregator_works_on_torch_tensors():
    """The NN forward pass calls aggregate_fn on torch tensors — must not break."""
    preds_np = _dst_sample_preds_numpy()
    preds_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in preds_np.items()}
    out = predictions_to_fantasy_points("DST", preds_torch, "ppr")
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(
        out.detach().numpy(),
        predictions_to_fantasy_points("DST", preds_np, "ppr"),
        rtol=0,
        atol=1e-5,
    )


def test_dst_aggregator_tier_boundaries_vectorized():
    """Sweep PA/YA across every tier edge — must match the scalar helpers."""
    from DST.dst_targets import _pts_allowed_to_bonus, _yds_allowed_to_bonus

    pa_values = np.array([0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 40, 55], dtype=np.float64)
    ya_values = np.array(
        [0, 99, 100, 199, 200, 299, 300, 349, 350, 399, 400, 449, 450, 600], dtype=np.float64
    )
    # Same length; pair up for a sweep.
    n = min(len(pa_values), len(ya_values))
    preds = {k: np.zeros(n, dtype=np.float64) for k in _dst_sample_preds_numpy()}
    preds["points_allowed"] = pa_values[:n]
    preds["yards_allowed"] = ya_values[:n]
    actual = predictions_to_fantasy_points("DST", preds, "ppr")
    expected = np.array(
        [
            _pts_allowed_to_bonus(p) + _yds_allowed_to_bonus(y)
            for p, y in zip(pa_values, ya_values, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
