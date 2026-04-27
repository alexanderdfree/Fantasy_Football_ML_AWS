"""Branch-coverage tests for ``shared/aggregate_targets.py``.

``tests/test_aggregate_targets.py`` already covers the happy-path parity
between the aggregator and ``compute_fantasy_points``. These tests fill
the remaining branches: the torch Tensor path through ``_tier_bonuses``,
DST aggregation with torch inputs, and the two ValueError branches in
``predictions_to_fantasy_points``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.shared.aggregate_targets import (
    _dst_predictions_to_fantasy_points,
    _tier_bonuses,
    aggregate_fn_for,
    predictions_to_fantasy_points,
)


@pytest.mark.unit
def test_tier_bonuses_numpy_and_torch_agree():
    """torch + numpy paths through _tier_bonuses must produce the same bonuses.

    DST PA tiers: [0, 1) → +5, [1, 7) → +4, [7, 14) → +3, [14, 21) → +1,
    [21, 28) → 0, [28, 35) → -1, [35, 999] → -4. We pick inputs that land in
    several different tiers (including boundary values).
    """
    from src.shared.aggregate_targets import _PA_BONUSES, _PA_BOUNDARIES

    values_np = np.array([0.0, 6.9, 14.0, 27.999, 35.0], dtype=np.float64)
    values_t = torch.tensor(values_np.tolist(), dtype=torch.float32)

    out_np = _tier_bonuses(values_np, _PA_BOUNDARIES, _PA_BONUSES)
    out_t = _tier_bonuses(values_t, _PA_BOUNDARIES, _PA_BONUSES)

    np.testing.assert_allclose(np.asarray(out_t), out_np)


@pytest.mark.unit
def test_dst_aggregation_with_torch_inputs():
    """DST predictions as torch tensors return a torch tensor of the same shape."""
    n = 4
    preds = {
        "def_sacks": torch.zeros(n),
        "def_ints": torch.zeros(n),
        "def_fumble_rec": torch.zeros(n),
        "def_fumbles_forced": torch.zeros(n),
        "def_safeties": torch.zeros(n),
        "def_tds": torch.zeros(n),
        "special_teams_tds": torch.zeros(n),
        "def_blocked_kicks": torch.zeros(n),
        "points_allowed": torch.tensor([0.0, 20.0, 35.0, 6.0], dtype=torch.float32),
        "yards_allowed": torch.tensor([50.0, 350.0, 500.0, 250.0], dtype=torch.float32),
    }
    out = _dst_predictions_to_fantasy_points(preds)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (n,)
    # Positive PA bonus for 0 points allowed, negative for 35+
    assert out[0].item() > out[2].item()


@pytest.mark.unit
def test_predictions_to_fantasy_points_raises_unknown_position():
    """Unknown position (not QB/RB/WR/TE/DST) → ValueError."""
    with pytest.raises(ValueError, match="No target map for position"):
        predictions_to_fantasy_points("UNKNOWN", {"anything": np.zeros(3)}, "ppr")


@pytest.mark.unit
def test_predictions_to_fantasy_points_raises_unknown_scoring():
    """Unknown scoring format → ValueError."""
    with pytest.raises(ValueError, match="Unknown scoring format"):
        predictions_to_fantasy_points("QB", {"passing_yards": np.zeros(3)}, "2qb")


@pytest.mark.unit
def test_predictions_to_fantasy_points_raises_on_no_recognized_targets():
    """preds_dict without ANY position target → ValueError."""
    with pytest.raises(ValueError, match="no recognized targets"):
        predictions_to_fantasy_points("QB", {"bogus_stat": np.zeros(3)}, "ppr")


@pytest.mark.unit
def test_aggregate_fn_for_returns_callable_bound_to_position():
    """``aggregate_fn_for('QB')`` → callable that needs only preds_dict."""
    fn = aggregate_fn_for("QB", scoring_format="half_ppr")
    preds = {
        "passing_yards": np.array([300.0]),
        "rushing_yards": np.array([0.0]),
        "passing_tds": np.array([2.0]),
        "rushing_tds": np.array([0.0]),
        "interceptions": np.array([0.0]),
        "fumbles_lost": np.array([0.0]),
    }
    out = fn(preds)
    assert out.shape == (1,)
    # 300*0.04 + 2*4 = 12 + 8 = 20 (half_ppr doesn't affect non-reception positions)
    np.testing.assert_allclose(out, [20.0])


@pytest.mark.unit
def test_aggregate_ignores_extra_total_key():
    """A stray 'total' key in preds_dict is silently ignored."""
    preds = {
        "passing_yards": np.array([100.0]),
        "passing_tds": np.array([1.0]),
        "rushing_yards": np.array([0.0]),
        "rushing_tds": np.array([0.0]),
        "interceptions": np.array([0.0]),
        "fumbles_lost": np.array([0.0]),
        "total": np.array([999.0]),  # should be ignored
    }
    out = predictions_to_fantasy_points("QB", preds, "ppr")
    # 100*0.04 + 1*4 = 8
    np.testing.assert_allclose(out, [8.0])
