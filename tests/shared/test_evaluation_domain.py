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
