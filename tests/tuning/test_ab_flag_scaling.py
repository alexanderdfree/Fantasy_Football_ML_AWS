"""Validate the dormant injury-scaling experiment's activation and cohort evidence."""

import numpy as np
import pandas as pd
import pytest

from src.shared.pipeline import _scale_xs
from src.tuning import ab_flag_scaling as spec

pytestmark = pytest.mark.unit


def config():
    return {
        "get_feature_columns_fn": lambda: ["game_status", "practice_status", "other"],
        "train_base_nn": True,
        "train_attention_nn": True,
        "attn_static_features": ["game_status", "practice_status"],
    }


def test_variants_keep_baseline_and_wire_both_explicit_ranges():
    assert [v.name for v in spec.VARIANTS] == ["baseline", "bounded_r1", "bounded_r4"]
    assert spec.VARIANTS[0].cfg_mutator is None
    for variant, expected in zip(spec.VARIANTS[1:], (1.0, 4.0), strict=True):
        assert variant.cfg_mutator(config())["nn_bounded_flag_range"] == expected
        assert variant.expect_ridge_identical is True


def test_mutator_rejects_a_missing_flag_in_either_scaled_path():
    cfg = config()
    cfg["attn_static_features"] = ["other"]
    with pytest.raises(AssertionError, match="attention static"):
        spec.VARIANTS[1].cfg_mutator(cfg)
    cfg = config()
    cfg["get_feature_columns_fn"] = lambda: ["other"]
    with pytest.raises(AssertionError, match="base NN"):
        spec.VARIANTS[1].cfg_mutator(cfg)


def result():
    return {
        "test_df": pd.DataFrame(
            {
                "fantasy_points": [10.0, 20.0, 30.0, 40.0],
                "pred_ridge_total": [11.0, 21.0, 31.0, 41.0],
                "pred_nn_total": [9.0, 19.0, 28.0, 38.0],
                "game_status": [1.0, 1.0, 0.5, 0.5],
            }
        )
    }


def test_metrics_preserve_ridge_sentinel_and_distinguish_cohort_bias():
    metrics = spec.metric_fn(result(), "WR")
    assert metrics["Ridge"]["mae"] == 1.0
    assert metrics["NN @healthy"]["bias"] == -1.0
    assert metrics["NN @questionable"]["bias"] == -2.0
    assert metrics["healthy coverage"]["n"] == 2
    assert metrics["questionable coverage"]["n"] == 2


def test_missing_or_unparseable_status_never_looks_like_a_complete_cohort():
    missing = result()
    missing["test_df"] = missing["test_df"].drop(columns="game_status")
    with pytest.raises(ValueError, match="cannot measure"):
        spec.metric_fn(missing, "WR")
    invalid = result()
    invalid["test_df"]["game_status"] = [1.0, "unknown", 0.5, 0.5]
    metrics = spec.metric_fn(invalid, "WR")
    assert metrics["game_status coverage"]["unparseable"] == 1
    assert metrics["healthy coverage"]["n"] == 1


@pytest.mark.parametrize("flag_range", [1.0, 4.0])
def test_flag_policy_preserves_standard_scaling_of_other_features(flag_range):
    values = np.array([[0, 0], [0.1, 1], [0.5, 2], [1, 3]], dtype=np.float32)
    baseline_scaler, [baseline] = _scale_xs(values)
    scaler, [scaled] = _scale_xs(
        values,
        cfg={"nn_bounded_flag_range": flag_range},
        feature_cols=["game_status", "ordinary_feature"],
    )
    np.testing.assert_allclose(
        scaled[:, 0], np.array([-1.0, -0.8, 0.0, 1.0]) * flag_range, atol=1e-6
    )
    np.testing.assert_array_equal(scaled[:, 1], baseline[:, 1])
    assert scaler.mean_[1] == baseline_scaler.mean_[1]
    assert scaler.scale_[1] == baseline_scaler.scale_[1]
    assert scaler.var_[1] == baseline_scaler.var_[1]
