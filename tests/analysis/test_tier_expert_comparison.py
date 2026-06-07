"""Unit tests for the tier-sliced model-vs-expert comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import tier_expert_comparison as tec
from src.analysis.analysis_expert_comparison import _EXPERT_PRED_COL, ExpertSource

pytestmark = pytest.mark.unit


def test_mae_bias_basic_and_empty():
    actual = np.array([10.0, 20.0])
    pred = np.array([8.0, 25.0])
    mae, bias, n = tec._mae_bias(actual, pred)
    assert mae == pytest.approx((2.0 + 5.0) / 2)
    assert bias == pytest.approx((-2.0 + 5.0) / 2)
    assert n == 2
    mae, bias, n = tec._mae_bias(np.array([]), np.array([]))
    assert np.isnan(mae) and np.isnan(bias) and n == 0


def _fake_expert():
    def _project(raw, pos, scoring_format):
        return raw.rename(columns={"proj": _EXPERT_PRED_COL})

    return ExpertSource(
        name="fake",
        label="Fake",
        load=lambda seasons: pd.DataFrame(),
        project=_project,
    )


def test_compare_position_slices_by_tier_and_includes_all_models(capsys):
    # Two elite + two field WRs (by prior-season FP), one matched week each.
    test_df = pd.DataFrame(
        {
            "player_id": ["E1", "E2", "F1", "F2"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "position": ["WR", "WR", "WR", "WR"],
            "fantasy_points": [25.0, 22.0, 8.0, 6.0],
            "pred_ridge_total": [20.0, 18.0, 7.5, 6.5],
            "pred_attn_nn_total": [21.0, 19.0, 7.0, 6.0],
        }
    )
    prior_fp = pd.Series(
        {("E1", 2024): 24.0, ("E2", 2024): 21.0, ("F1", 2024): 9.0, ("F2", 2024): 7.0}
    )
    expert_raw = pd.DataFrame(
        {
            "player_id": ["E1", "E2", "F1", "F2"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "proj": [23.0, 20.0, 7.0, 6.0],
        }
    )
    expert = _fake_expert()
    tec.compare_position("WR", test_df, prior_fp, [expert], {"fake": expert_raw}, tier_topn=2)
    out = capsys.readouterr().out
    assert "elite_top_drafted  (n=2)" in out
    assert "field  (n=2)" in out
    # All present models plus the expert are reported.
    assert "Ridge" in out and "Attention NN" in out and "Fake" in out
