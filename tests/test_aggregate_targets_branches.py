"""Branch-coverage tests for ``src/shared/aggregate_targets.py``.

``tests/test_aggregate_targets.py`` already covers the happy-path parity
between the aggregator and ``compute_fantasy_points``. These tests fill
the remaining branches: the torch Tensor path through ``_tier_bonuses``,
DST aggregation with torch inputs, and the two ValueError branches in
``predictions_to_fantasy_points``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
def test_numpy_scoring_does_not_require_torch():
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise AssertionError("NumPy scoring imported Torch")
        sys.meta_path.insert(0, NoTorch())
        import numpy as np
        from src.shared.aggregate_targets import (
            DST_TARGETS, K_TARGETS, POSITION_TARGET_MAP, predictions_to_fantasy_points,
        )
        targets = {**POSITION_TARGET_MAP, "K": K_TARGETS, "DST": DST_TARGETS}
        for position, names in targets.items():
            values = {name: np.zeros(2) for name in names}
            for scoring in ("ppr", "half_ppr", "standard"):
                result = predictions_to_fantasy_points(position, values, scoring)
                assert result.shape == (2,) and np.isfinite(result).all()
        assert "torch" not in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


from src.shared.registry import get_config


@pytest.mark.unit
@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_tensor_scoring_matches_numpy_with_leading_member_dimension(position):
    targets = get_config(position)["targets"]
    values = np.array([[0.0, 7.0, 14.0], [1.0, 28.0, 35.0]])
    arrays = {t: values for t in targets}
    tensors = {t: torch.tensor(values, requires_grad=True) for t in targets}
    actual = predictions_to_fantasy_points(position, tensors)
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(
        actual.detach().numpy(), predictions_to_fantasy_points(position, arrays)
    )
    actual.sum().backward()
    assert any(t.grad is not None for t in tensors.values())


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
def test_tier_bonuses_ya_numpy_and_torch_agree():
    """torch + numpy parity for the YA tier path.

    Yards-allowed tiers span a different boundary set than PA: pick inputs
    that land in several different tiers (including boundary values) so the
    bucketize/digitize edge-inclusion convention is exercised on both sides.
    """
    from src.shared.aggregate_targets import _YA_BONUSES, _YA_BOUNDARIES

    values_np = np.array([0.0, 99.0, 100.0, 350.0, 450.0, 600.0], dtype=np.float64)
    values_t = torch.tensor(values_np.tolist(), dtype=torch.float32)

    out_np = _tier_bonuses(values_np, _YA_BOUNDARIES, _YA_BONUSES)
    out_t = _tier_bonuses(values_t, _YA_BOUNDARIES, _YA_BONUSES)

    np.testing.assert_allclose(np.asarray(out_t), out_np)


@pytest.mark.unit
def test_aggregation_with_torch_inputs():
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
