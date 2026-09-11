"""Regression contracts for the evaluation boundary and provenance records."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics
from src.evaluation.records import evaluation_data_identity, record_for_result

pytestmark = pytest.mark.unit


def sample():
    return pd.DataFrame(
        {
            "player_id": ["b", "a", "c"],
            "season": 2025,
            "week": 1,
            "fantasy_points": [10.0, 20.0, 30.0],
            "pred_ridge_total": [11.0, 18.0, 31.0],
            "bucket": ["low", "high", "high"],
        }
    )


def test_legacy_metric_exports_preserve_callable_identity():
    from src.analysis import cohort_analysis, significance
    from src.shared import evaluation

    assert evaluation.compute_metrics is metrics.compute_metrics
    for name in (
        "available_models",
        "per_model_metrics",
        "bucket_model_table",
        "bias_corrected_mae",
        "best_model",
    ):
        assert getattr(cohort_analysis, name) is getattr(metrics, name)
    assert significance.pred_columns_from_test_df is metrics.pred_columns_from_test_df
    assert significance.CANONICAL_PRED_COLUMNS is metrics.CANONICAL_PRED_COLUMNS


def test_metrics_retain_error_sign_empty_and_invalid_input_behavior():
    frame = sample()
    result = metrics.per_model_metrics(frame)["Ridge"]
    assert result == {
        "mae": pytest.approx(4 / 3),
        "bias": 0.0,
        "rmse": pytest.approx(np.sqrt(2)),
        "n": 3,
    }
    assert metrics.best_model(frame)[0] == "Ridge"
    empty = metrics.per_model_metrics(frame.iloc[:0])["Ridge"]
    assert empty["n"] == 0 and np.isnan(empty["mae"])
    frame.loc[0, "pred_ridge_total"] = np.nan
    with pytest.raises(ValueError):
        metrics.per_model_metrics(frame)


def test_bucket_and_bias_helpers_keep_selection_and_centering():
    table = metrics.bucket_model_table(sample(), "bucket").set_index("bucket")
    assert table.loc["high", "n"] == 2
    assert table.loc["high", "bias"] == -0.5
    assert table.loc["low", "mae"] == 1
    centered = metrics.bias_corrected_mae(sample(), "fantasy_points", "pred_ridge_total", "bucket")
    assert centered.to_dict() == {"high": 1.5, "low": 0.0}


def test_evaluation_identity_ignores_predictions_and_row_order():
    frame = sample()
    identity = evaluation_data_identity(frame)
    frame["pred_ridge_total"] = 99.0
    assert evaluation_data_identity(frame.iloc[::-1]) == identity
    frame.loc[0, "fantasy_points"] += 1
    assert evaluation_data_identity(frame) != identity


def test_record_identity_distinguishes_equal_native_dst_totals_with_different_raw_truth():
    from src.shared.aggregate_targets import DST_TARGETS, predictions_to_fantasy_points
    from src.shared.comparison_scoring import score_actual_components

    frame = pd.DataFrame(
        {
            "player_id": ["BUF"],
            "season": [2025],
            "week": [1],
            "position": ["DST"],
            **{target: [0.0] for target in DST_TARGETS},
        }
    )
    frame["yards_allowed"] = 349.0
    frame["fantasy_points"] = predictions_to_fantasy_points(
        "DST", {c: frame[c] for c in DST_TARGETS}
    )
    changed = frame.copy()
    changed["points_allowed"] = 21.0
    changed["def_sacks"] = 10.0
    changed["fantasy_points"] = predictions_to_fantasy_points(
        "DST", {c: changed[c] for c in DST_TARGETS}
    )
    assert frame["fantasy_points"].equals(changed["fantasy_points"])
    assert not score_actual_components(frame, "DST").equals(score_actual_components(changed, "DST"))
    first = record_for_result("DST", {"test_df": frame}, actual_columns=DST_TARGETS)
    second = record_for_result("DST", {"test_df": changed}, actual_columns=DST_TARGETS)
    assert first.evaluation_data_id is not None
    assert first.evaluation_data_id != second.evaluation_data_id


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_summary_identity_binds_raw_targets_even_when_model_metrics_are_sparse(position):
    from src.shared.aggregate_targets import DST_TARGETS, K_TARGETS, POSITION_TARGET_MAP
    from src.shared.benchmark_utils import summarize_pipeline_result

    targets = tuple({**POSITION_TARGET_MAP, "K": K_TARGETS, "DST": DST_TARGETS}[position])
    frame = sample().assign(**{target: 0.0 for target in targets})
    result = {
        "test_df": frame,
        "ridge_metrics": {"total": {"mae": 1.0, "r2": 0.0}},
        "nn_metrics": {"total": {"mae": 2.0, "r2": 0.0}},
        "cohorts": {"custom": {}},
    }
    first = summarize_pipeline_result(position, result)["evaluation_record"]["evaluation_data_id"]
    assert first is not None
    frame.loc[0, targets[0]] = 1.0
    second = summarize_pipeline_result(position, result)["evaluation_record"]["evaluation_data_id"]
    assert second != first
    result["test_df"] = frame.drop(columns=targets[0])
    assert (
        summarize_pipeline_result(position, result)["evaluation_record"]["evaluation_data_id"]
        is None
    )


def test_record_requires_declared_truth_and_keeps_predictions_out_of_identity():
    frame = sample().assign(passing_yards=100.0, fantasy_points_half_ppr=10.0)
    assert record_for_result("QB", {"test_df": frame}).evaluation_data_id is None
    result = {"test_df": frame, "ridge_metrics": {"passing_yards": {"mae": 1.0}}}
    first = record_for_result("QB", result).evaluation_data_id
    assert first is not None
    frame["pred_ridge_total"] = 999.0
    frame["engineered_feature"] = 10.0
    result["test_df"] = frame.iloc[::-1]
    assert record_for_result("QB", result).evaluation_data_id == first
    result["test_df"].loc[0, "fantasy_points_half_ppr"] = 11.0
    assert record_for_result("QB", result).evaluation_data_id != first
    assert (
        record_for_result("QB", result, actual_columns=("missing_truth",)).evaluation_data_id
        is None
    )


def test_record_identity_binds_prefill_comparison_truth_and_availability():
    frame = sample().assign(passing_yards=0.0, actual_projected_total=np.nan)
    result = {"test_df": frame}
    unavailable = record_for_result("QB", result, actual_columns=("passing_yards",))
    frame["actual_projected_total"] = 0.0
    available = record_for_result("QB", result, actual_columns=("passing_yards",))
    assert unavailable.evaluation_data_id is not None
    assert unavailable.evaluation_data_id != available.evaluation_data_id


def test_record_keeps_unknown_history_and_execution_regimes_distinct(monkeypatch):
    monkeypatch.setenv("FF_DATASET_ID", "current-job-dataset")
    result = {
        "test_df": sample(),
        "data_id": "fitted-data",
        "model_bundle_ids": {"ridge": "bundle"},
        "cohorts": {
            "reference": {
                "status": "unavailable",
                "n": None,
                "actual_basis": "shared_projected_components_v1",
            }
        },
    }
    eager = record_for_result("QB", result, execution_regime="eager").to_dict()
    stacked = record_for_result(
        "QB", result, execution_regime="stacked", use_environment=True
    ).to_dict()
    assert eager["scoring"] is None and eager["actual_basis"] is None
    assert eager["dataset_id"] is None
    assert stacked["dataset_id"] == "current-job-dataset"
    assert eager["training_data_id"] == "fitted-data"
    assert eager["evaluation_data_id"] != eager["training_data_id"]
    assert eager["execution_regime"] != stacked["execution_regime"]
    assert eager["cohorts"]["reference"]["n"] is None
    assert eager["cohorts"]["reference"]["actual_basis"] == "shared_projected_components_v1"
    assert json.loads(json.dumps(eager))["model_bundle_ids"] == {"ridge": "bundle"}
