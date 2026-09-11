"""Evidence and regime contracts for the complete RMSE selector comparison."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.shared.registry import get_config
from src.tuning import ab_rmse_readiness as spec

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", spec.POSITIONS)
def test_combined_arms_only_change_three_selection_policies(position, monkeypatch):
    monkeypatch.setattr(spec, "_observe_capture", lambda: None)
    monkeypatch.delenv("FF_NN_FIXED_EPOCHS", raising=False)
    monkeypatch.delenv("FF_AB_STACKED", raising=False)
    config = get_config(position)
    keys = ("nn_selection_metric", "ridge_selection_metric", "lgbm_selection_metric")
    for variant, expected in zip(
        spec.VARIANTS,
        (["weighted_mae", "raw_mae", "per_target"], ["fantasy_rmse_ppr"] * 3),
        strict=True,
    ):
        candidate = copy.deepcopy(config)
        before = {k: v for k, v in candidate.items() if k not in keys}
        candidate = variant.cfg_mutator(candidate)
        assert [candidate[k] for k in keys] == expected
        assert {k: v for k, v in candidate.items() if k not in keys} == before


@pytest.mark.parametrize("key,value", [("FF_AB_STACKED", "1"), ("FF_NN_FIXED_EPOCHS", "30")])
def test_rejects_modes_that_bypass_checkpoint_selection(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        spec, "_observe_capture", lambda: pytest.fail("should fail before wrapping")
    )
    with pytest.raises(ValueError):
        spec.baseline({})


def test_k_capture_exception_does_not_disable_other_positions(monkeypatch):
    event = {"family": "nn", "returned_true": True, "graph_present": True, "capturable_loss": True}
    monkeypatch.setattr(spec, "_CAPTURE_EVENTS", [event])
    result = {"execution": {"device": "cuda:0"}}
    assert spec._capture_evidence(result, "K")["required"] is True
    with pytest.raises(ValueError, match="capture did not engage"):
        spec._capture_evidence(result, "WR")


def test_row_metrics_keep_zero_and_exclude_missing():
    rows = pd.DataFrame(
        {"comparison_actual": [0.0, 2.0, np.nan], "comparison_available": [True, True, False]}
    )
    for family in spec.FAMILIES:
        rows[f"comparison_{family}"] = [0.0, 0.0, 0.0]
    for metric in spec._row_metrics(rows, "native").values():
        assert metric == {"n": 2, "unavailable": 1, "mae": 1.0, "rmse": np.sqrt(2), "bias": -1.0}


def test_prepared_hashes_detect_real_inputs_and_ignore_selector_metadata():
    frame = pd.DataFrame({"player_id": ["a", "b"], "x": [1.0, 2.0]})
    prepared = SimpleNamespace(**{k: frame.copy() for k in ("train", "val", "test")})
    for split in ("train", "val", "test"):
        setattr(prepared, f"X_{split}", np.array([[1.0], [2.0]]))
        setattr(prepared, f"y_{split}", {"target": np.array([0.0, 1.0])})
    original = spec._prepared_hashes(prepared)
    prepared.train.attrs["prepared_data_id"] = "selector-dependent"
    assert spec._prepared_hashes(prepared) == original
    prepared.X_train[0, 0] += 1.0
    changed = spec._prepared_hashes(prepared)
    assert changed["X_train"] != original["X_train"]
    assert changed["X_val"] == original["X_val"]


def test_local_evidence_cannot_overwrite_other_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("FF_AB_RUN_ID", raising=False)
    monkeypatch.setenv("FF_RMSE_READINESS_OUTPUT", str(tmp_path))
    write = spec._evidence_sink("WR-baseline-42")
    write("rows.parquet", b"original")
    write("rows.parquet", b"original")
    with pytest.raises(ValueError, match="different bytes"):
        write("rows.parquet", b"changed")


def _populated_cohorts():
    return {
        name: {
            "status": "available",
            "n": 2,
            "models": {
                family: (
                    {"n_weeks": 2, "hit_rate": 0.5, "points_captured": 1.0, "lineup_regret": 1.0}
                    if name == "weekly_actual_top24"
                    else {"n": 2, "mae": 1.0, "rmse": 1.0, "bias": 0.0}
                )
                for family in ("Ridge", "NN", "Attention NN", "LightGBM")
            },
        }
        for name in spec.REQUIRED_COHORTS
    }


def _selector_fixture(aligned=False):
    config = get_config("RB")
    targets = config["targets"]
    metric = "fantasy_rmse_ppr" if aligned else "weighted_mae"
    config.update(
        nn_selection_metric=metric,
        ridge_selection_metric="fantasy_rmse_ppr" if aligned else "raw_mae",
        lgbm_selection_metric="fantasy_rmse_ppr" if aligned else "per_target",
    )
    report = {
        "metric": metric,
        "scoring_format": "ppr" if aligned else None,
        "fixed_epochs": False,
        "epoch": 2,
        "score": 1.0,
        "validation_curve": [2.0, 1.0, 1.5],
        "validation_metrics": {"val_fantasy_rmse_ppr": 1.0},
    }
    values = {
        "history": {"checkpoint_selection": copy.deepcopy(report)},
        "attn_history": {"checkpoint_selection": copy.deepcopy(report)},
    }
    special = set(config.get("two_stage_targets", {})) | set(
        config.get("classification_targets", {})
    )
    if aligned:
        values["ridge_selection"] = {
            "metric": "mean_cv_fantasy_rmse_ppr",
            "scoring_format": "ppr",
            "score": 1.0,
            "alphas": {t: 1.0 for t in targets if t not in special},
            "n_folds": 4,
            "score_history": [2.0, 1.0],
        }
        values["lgbm_selection"] = {
            "metric": "fantasy_rmse_ppr",
            "scoring_format": "ppr",
            "score": 1.0,
            "iterations": {t: 2 for t in targets},
            "n_validation_rows": 2,
            "score_history": [2.0, 1.0],
        }
    models = {
        "ridge": SimpleNamespace(_alphas={t: 1.0 for t in targets}),
        "lgbm": SimpleNamespace(
            selection_metric=config["lgbm_selection_metric"],
            selected_iterations={t: 2 for t in targets} if aligned else {},
            _models={t: SimpleNamespace(best_iteration_=2) for t in targets},
        ),
    }
    return config, values, models


def test_complete_evidence_restores_raw_predictions_from_training_result(tmp_path, monkeypatch):
    import json

    from src.training.context import RunContext, use_context
    from src.training.contracts import PreparedDataset, TrainingResult, resolve_recipe

    config, selector_values, fitted_models = _selector_fixture()
    targets = config["targets"]
    frame = pd.DataFrame({t: [0.0, 1.0] for t in targets})
    frame["player_id"] = ["a", "b"]
    frame["season"] = 2025
    frame["week"] = [1, 2]
    frame["season_type"] = "REG"
    frame["position"] = "RB"
    frame["fantasy_points"] = 999.0
    predictions = {family: {t: np.zeros(2) for t in targets} for family in spec.FAMILIES}
    for family in spec.FAMILIES:
        frame[f"pred_{family}_total"] = 0.0
    truth = {t: frame[t].to_numpy() for t in targets}
    prepared = PreparedDataset(
        *(np.ones((2, 1)) for _ in range(3)),
        truth,
        truth,
        truth,
        frame,
        frame,
        frame,
        ("x",),
        "fake-prepared-id",
    )
    result = TrainingResult(
        {
            "test_df": frame,
            "per_target_preds": predictions,
            **{f"{f}_metrics": {} for f in spec.FAMILIES},
            "execution": {"device": "cpu"},
            "data_id": prepared.data_id,
            "cohorts": _populated_cohorts(),
            **selector_values,
        },
        resolve_recipe("RB", config),
        prepared,
        fitted_models,
        "test-run",
    )
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (data / "splits" / f"{split}.parquet").write_bytes(b"pinned-fixture")
    reference = tmp_path / "reference.parquet"
    reference.write_bytes(b"pinned-reference")
    monkeypatch.setattr(spec, "load_reference", lambda **kwargs: None)
    monkeypatch.setattr(spec, "reference_path", lambda **kwargs: reference)
    monkeypatch.setattr(
        spec,
        "reference_selection",
        lambda *args: (np.array([True, False]), {"status": "available"}),
    )
    monkeypatch.setattr(spec, "_ARM", "baseline")
    monkeypatch.setattr(spec, "_CAPTURE_EVENTS", [])
    monkeypatch.delenv("FF_AB_RUN_ID", raising=False)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setenv("FF_RMSE_READINESS_OUTPUT", str(tmp_path / "evidence"))
    with use_context(RunContext(tmp_path / "output", data, seed=42)):
        metrics = spec.metric_fn(result, "RB")
    assert metrics["readiness"]["native_rows"] == 2
    assert metrics["native:nn"]["n"] == 2
    assert metrics["native:nn"]["mae"] < 999
    directory = tmp_path / "evidence" / "RB-baseline-42"
    manifest = json.loads(next(directory.glob("manifest-*.json")).read_text())
    rows = pd.read_parquet(manifest["evaluations"]["native"]["location"])
    assert all(f"pred_{f}_{t}" in rows for f in spec.FAMILIES for t in targets)
    assert rows["comparison_available"].all()
    assert len(manifest["cohorts"]) == 4
    assert manifest["prepared_inputs_sha256"]["X_train"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_metric",
        "bad_epoch",
        "empty_curve",
        "nonfinite_curve",
        "wrong_alpha",
        "wrong_prefix",
    ],
)
def test_aligned_selection_rejects_incomplete_or_ignored_policies(monkeypatch, mutation):
    from src.training.contracts import TrainingResult

    config, values, models = _selector_fixture(aligned=True)
    if mutation == "missing":
        del values["ridge_selection"]
    elif mutation == "wrong_metric":
        values["history"]["checkpoint_selection"]["metric"] = "weighted_mae"
    elif mutation == "bad_epoch":
        values["history"]["checkpoint_selection"]["epoch"] = 8
    elif mutation == "empty_curve":
        values["lgbm_selection"]["score_history"] = []
    elif mutation == "nonfinite_curve":
        values["attn_history"]["checkpoint_selection"]["validation_curve"] = [float("nan")]
    elif mutation == "wrong_alpha":
        values["ridge_selection"]["alphas"][next(iter(values["ridge_selection"]["alphas"]))] = 2.0
    elif mutation == "wrong_prefix":
        values["lgbm_selection"]["iterations"][
            next(iter(values["lgbm_selection"]["iterations"]))
        ] = 3
    result = TrainingResult(values, config, None, models, "test")
    monkeypatch.setattr(spec, "_ARM", "ppr_rmse")
    with pytest.raises(ValueError):
        spec._selection_evidence(result)


@pytest.mark.parametrize("aligned", [False, True])
def test_selection_accepts_complete_matched_fitted_evidence(monkeypatch, aligned):
    from src.training.contracts import TrainingResult

    config, values, models = _selector_fixture(aligned=aligned)
    monkeypatch.setattr(spec, "_ARM", "ppr_rmse" if aligned else "baseline")
    assert (
        spec._selection_evidence(TrainingResult(values, config, None, models, "test"))["nn"][
            "epoch"
        ]
        == 2
    )


@pytest.mark.parametrize("mutation", ["empty", "missing_model", "zero_model", "nonfinite"])
def test_required_cohort_rejects_unpopulated_evidence(mutation):
    cohorts = _populated_cohorts()
    block = cohorts["weekly_reference_top24"]
    if mutation == "empty":
        block["n"] = 0
    elif mutation == "missing_model":
        del block["models"]["LightGBM"]
    elif mutation == "zero_model":
        block["models"]["NN"]["n"] = 0
    elif mutation == "nonfinite":
        block["models"]["Attention NN"]["rmse"] = None
    with pytest.raises(ValueError):
        spec._cohort_evidence(cohorts)


@pytest.mark.parametrize("key", ["scheduler_type", "attn_scheduler_type"])
def test_legacy_schedule_contract_rejects_plateau_override(key, monkeypatch):
    monkeypatch.setattr(spec, "_observe_capture", lambda: pytest.fail("must reject first"))
    with pytest.raises(ValueError, match="non-plateau"):
        spec.baseline({key: "plateau"})


def test_ranking_cohort_requires_real_ranked_weeks():
    cohorts = _populated_cohorts()
    assert spec._cohort_evidence(cohorts)
    cohorts["weekly_actual_top24"]["models"]["NN"]["n_weeks"] = 0
    with pytest.raises(ValueError, match="weekly_actual_top24/NN"):
        spec._cohort_evidence(cohorts)
