"""Unit tests for the pure helpers in src.analysis.artifact_eval.

The full ``build_test_df_from_artifacts`` path needs saved model artifacts +
data splits (exercised by the CLI smoke / diagnostics, not here); these tests
cover the drift-free pure helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import artifact_eval as ae

pytestmark = pytest.mark.unit


def test_attach_predictions_writes_total_and_per_target_columns():
    pos_test = pd.DataFrame({"player_id": ["a", "b"], "fantasy_points": [10.0, 20.0]})
    preds = {"yards": np.array([5.0, 7.0]), "tds": np.array([1.0, 2.0])}
    targets = ["yards", "tds"]
    total_fn = lambda p: p["yards"] + 6.0 * p["tds"]  # noqa: E731 - tiny test stub
    ae.attach_predictions(pos_test, "ridge", preds, targets, total_fn)
    assert list(pos_test["pred_ridge_yards"]) == [5.0, 7.0]
    assert list(pos_test["pred_ridge_tds"]) == [1.0, 2.0]
    # total = yards + 6*tds
    assert list(pos_test["pred_ridge_total"]) == [11.0, 19.0]


def test_make_total_fn_uses_target_signs_for_K_like_position():
    reg = {"target_signs": {"fg_points": 1.0, "misses": -1.0}}
    total_fn = ae._make_total_fn("K", ["fg_points", "misses"], reg, "ppr")
    preds = {"fg_points": np.array([9.0, 3.0]), "misses": np.array([1.0, 0.0])}
    # sign-vectored sum: fg_points - misses
    assert list(total_fn(preds)) == [8.0, 3.0]


def test_attn_supported_for_flat_incl_opp_history_but_not_nested():
    assert ae._attn_is_supported({}) is True
    assert ae._attn_is_supported({"attn_history_structure": "flat"}) is True
    # Opponent-history side branch (skill positions) IS supported in v1.
    assert ae._attn_is_supported({"opp_attn_history_stats": ["def_sacks"]}) is True
    # Nested per-kick variant (K) is not yet handled.
    assert ae._attn_is_supported({"attn_history_structure": "nested"}) is False
