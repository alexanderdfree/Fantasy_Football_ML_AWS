"""Report the modeled components without changing inputs or full fantasy totals."""

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.shared import pipeline as p
from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.comparison_scoring import scoring_components

pytestmark = pytest.mark.unit


def _case(position):
    targets = list(scoring_components(position))
    truth = {target: np.zeros(32) for target in targets}
    truth[targets[0]] = np.tile(np.arange(16, dtype=float), 2)
    frame = pd.DataFrame(truth)
    frame["player_id"] = [f"p{i}" for i in range(16)] * 2
    frame["season"] = 2025
    frame["week"] = np.repeat([1, 2], 16)
    frame["position"] = position
    frame["fantasy_points"] = aggregate_fn_for(position)(truth)
    # An unprojected contribution changes the true top 12 under full scoring.
    frame.loc[[0, 16], "fantasy_points"] += 100
    for name in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"):
        frame[name] = 0.0
    for target, source, factor in (
        ("fg_yard_points", "fg_yards_made", 10),
        ("pat_points", "pat_made", 1),
        ("fg_misses", "fg_missed", 1),
        ("xp_misses", "pat_missed", 1),
    ):
        if target in truth:
            frame[source] = truth[target] * factor
    cfg = {
        "targets": targets,
        "aggregate_fn": aggregate_fn_for(position),
        "compute_targets_fn": importlib.import_module(
            f"src.{position.lower()}.targets"
        ).compute_targets,
        "ridge_alpha_grids": {target: [1.0] for target in targets},
        "nn_batch_size": 16,
        "nn_lr": 0.001,
        "nn_weight_decay": 0.0,
        "nn_patience": 1,
        "attn_static_features": ["a", "b"],
    }
    return frame, truth, cfg


def _fake_training(monkeypatch, tmp_path, position):
    frame, truth, cfg = _case(position)
    train, val = frame.assign(season=2023), frame.assign(season=2024)
    x = np.arange(len(frame) * 2, dtype=np.float32).reshape(-1, 2)
    prepared = (
        x.copy(),
        x.copy(),
        x.copy(),
        copy.deepcopy(truth),
        copy.deepcopy(truth),
        copy.deepcopy(truth),
        train.copy(),
        val.copy(),
        frame.copy(),
        ["a", "b"],
    )
    pristine = copy.deepcopy(prepared)
    seen = {}

    class ExactModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            return copy.deepcopy(truth)

        predict_numpy = predict

        def to(self, device):
            return self

        def state_dict(self):
            return {}

        def save(self, *args, **kwargs):
            pass

        def get_feature_importance(self, *args, **kwargs):
            return {}

        def convergence_report(self):
            return {}

    def nn(*args, **kwargs):
        return (
            ExactModel(),
            None,
            copy.deepcopy(truth),
            p.compute_target_metrics(truth, truth, cfg["targets"]),
            {},
        )

    def cohorts(position, result_frame, **kwargs):
        seen["report"] = result_frame.copy()
        return {}

    for name, value in {
        "_prepare_position_data": lambda *a, **kw: prepared,
        "_prepare_train_val": lambda *a, **kw: (
            prepared[0],
            prepared[1],
            prepared[3],
            prepared[4],
            prepared[6],
            prepared[7],
            prepared[9],
        ),
        "_tune_ridge_alphas_cv": lambda *a, **kw: {},
        "RidgeMultiTarget": ExactModel,
        "_build_lgbm": lambda *a, **kw: ExactModel(),
        "_train_nn": nn,
        "_train_attention_holdout": lambda *a, **kw: (*nn(), ["a", "b"]),
        "_tune_enet_cv": lambda *a, **kw: {},
        "_train_elasticnet": lambda *a, **kw: (ExactModel(), copy.deepcopy(truth), nn()[3]),
        "_train_lightgbm": lambda *a, **kw: (ExactModel(), copy.deepcopy(truth), nn()[3]),
        "_train_tabpfn": lambda *a, **kw: (ExactModel(), copy.deepcopy(truth), nn()[3]),
        "expanding_window_folds": lambda *a, **kw: [(0, train, val)],
        "_scale_xs": lambda *xs: (None, xs),
        "make_dataloaders": lambda *a, **kw: (None, None),
        "build_multihead_net": lambda *a, **kw: ExactModel(),
        "_maybe_compile": lambda model: model,
        "_run_nn_training": lambda **kw: None,
        "build_cohorts": cohorts,
        "plot_training_curves": lambda *a, **kw: None,
        "plot_weekly_accuracy": lambda *a, **kw: None,
        "plot_pred_vs_actual": lambda *a, **kw: None,
        "write_scaler_meta": lambda *a, **kw: None,
    }.items():
        monkeypatch.setattr(p, name, value)
    monkeypatch.setattr(p.torch, "save", lambda *a, **kw: None)
    monkeypatch.setattr(p.joblib, "dump", lambda *a, **kw: None)
    monkeypatch.setattr(p.plt, "subplots", lambda *a, **kw: (None, []))
    for name in ("tight_layout", "savefig", "close"):
        monkeypatch.setattr(p.plt, name, lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(
        frame=frame,
        truth=truth,
        cfg=cfg,
        train=train,
        val=val,
        prepared=prepared,
        pristine=pristine,
        seen=seen,
    )


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
@pytest.mark.parametrize("mode", ["holdout", "cv", "partial_cpu", "partial_nn"])
def test_pipeline_reports_matching_components_without_changing_inputs(
    monkeypatch, tmp_path, position, mode
):
    case = _fake_training(monkeypatch, tmp_path, position)
    before = case.frame.copy(deep=True)
    case.cfg.update(train_base_nn=mode != "partial_cpu", train_ridge=mode != "partial_nn")
    if mode == "cv":
        result = p.run_cv_pipeline(
            position, case.cfg, pd.concat([case.train, case.val]), case.frame
        )
    else:
        result = p.run_pipeline(position, case.cfg, case.train, case.val, case.frame)
    report = case.seen["report"]
    expected = case.cfg["aggregate_fn"](case.truth)
    np.testing.assert_array_equal(report["actual_projected_total"], expected)
    pd.testing.assert_series_equal(report["fantasy_points"], before["fantasy_points"])
    pd.testing.assert_frame_equal(case.frame, before)
    for actual, original in zip(case.prepared[:9], case.pristine[:9], strict=True):
        if isinstance(actual, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, original)
        elif isinstance(actual, dict):
            for key in actual:
                np.testing.assert_array_equal(actual[key], original[key])
        else:
            np.testing.assert_array_equal(actual, original)
    for key in ("ridge_ranking", "nn_ranking"):
        if key in result:
            assert result[key]["season_avg_hit_rate"] == 1.0
    if mode in {"holdout", "cv"}:
        assert result["sim_results"]["season_summary"]["Ridge"]["mae"] == 0.0
        assert result["sim_results"]["season_summary"]["Neural Net"]["mae"] == 0.0
        np.testing.assert_array_equal(result["test_df"]["pred_ridge_total"], expected)
        # The season-average comparator must use the same components as its truth.
        np.testing.assert_array_equal(result["test_df"]["pred_baseline"].iloc[16:], expected[:16])
    assert report.attrs["actual_projected_total_metadata"] == {
        "basis": "configured_target_aggregation_v1",
        "targets": case.cfg["targets"],
        "scoring_format": "ppr",
    }


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_original_missing_component_is_not_made_available_by_target_fill(position):
    source, truth, cfg = _case(position)
    frame = source.copy()
    raw_column = "fg_yards_made" if position == "K" else cfg["targets"][0]
    source.loc[0, raw_column] = np.nan
    # A feature merge can reorder/reset indices. Match by player/season/week.
    source = source.iloc[::-1].reset_index(drop=True)
    report = p._reporting_frame(frame, cfg, truth, source_frame=source)
    assert np.isnan(report.loc[0, "actual_projected_total"])
    assert report.loc[1:, "actual_projected_total"].notna().all()
    assert np.isfinite(truth[cfg["targets"][0]]).all()


def test_derived_fumble_requires_all_original_components():
    source, truth, cfg = _case("RB")
    source.loc[0, "sack_fumbles_lost"] = np.nan
    report = p._reporting_frame(source.copy(), cfg, truth, source_frame=source)
    assert np.isnan(report.loc[0, "actual_projected_total"])
    source = source.drop(columns="receiving_fumbles_lost")
    assert (
        p._reporting_frame(source, cfg, truth, source_frame=source)["actual_projected_total"]
        .isna()
        .all()
    )


def test_reduced_custom_targets_and_unavailable_values():
    frame = pd.DataFrame({"fantasy_points": [999.0, 999.0, 999.0]})
    cfg = {"targets": ["a", "b"], "aggregate_fn": lambda y: y["a"] - 2 * y["b"]}
    truth = {"a": np.array([5.0, np.nan, 3.0]), "b": np.array([1.0, 1.0, np.inf])}
    report = p._reporting_frame(frame, cfg, truth)
    assert report["actual_projected_total"].iloc[0] == 3.0
    assert report["actual_projected_total"].iloc[1:].isna().all()
    assert report.attrs["actual_projected_total_metadata"]["scoring_format"] is None
    assert p._reporting_frame(frame, cfg, {"a": truth["a"]})["actual_projected_total"].isna().all()
    assert (
        p._reporting_frame(frame, {"targets": ["a"]}, truth)["actual_projected_total"].isna().all()
    )


def test_partial_fixture_without_aggregator_remains_supported(monkeypatch, tmp_path):
    case = _fake_training(monkeypatch, tmp_path, "RB")
    case.cfg.pop("aggregate_fn")
    case.cfg["train_ridge"] = False
    result = p.run_pipeline("RB", case.cfg, case.train, case.val, case.frame)
    assert result["nn_metrics"]["total"]["mae"] == 0.0
    assert "nn_ranking" not in result
    assert case.seen["report"].empty


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
@pytest.mark.parametrize("with_aggregator", [True, False])
def test_tuned_lgbm_reports_both_models_on_matching_truth(monkeypatch, position, with_aggregator):
    from src.tuning import tune_lgbm

    frame, truth, cfg = _case(position)
    before = frame.copy(deep=True)
    x = np.arange(64, dtype=np.float32).reshape(-1, 2)
    monkeypatch.setattr(tune_lgbm.pd, "read_parquet", lambda *a, **kw: frame)
    monkeypatch.setattr(
        tune_lgbm,
        "_prepare_position_data",
        lambda *a, **kw: (x, x, x, truth, truth, truth, frame, frame, frame, ["a", "b"]),
    )

    class ExactLGBM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            return copy.deepcopy(truth)

    monkeypatch.setattr(tune_lgbm, "LightGBMMultiTarget", ExactLGBM)
    if not with_aggregator:
        cfg.pop("aggregate_fn")
    result = tune_lgbm._run_comparison(position, cfg, {}, seeds=(42,))
    for label in ("old", "new"):
        assert result["per_seed"][0][f"{label}_ranking"]["hit_rate"] == 1.0
        assert result["per_seed"][0][f"{label}_metrics"]["total"]["mae"] == 0.0
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("mode", ["holdout", "cv", "partial_cpu", "partial_nn"])
def test_optional_model_rankings_use_projected_truth(monkeypatch, tmp_path, mode):
    case = _fake_training(monkeypatch, tmp_path, "RB")
    case.cfg.update(
        train_base_nn=mode != "partial_cpu",
        train_ridge=mode != "partial_nn",
        train_attention_nn=True,
        train_lightgbm=True,
        train_elasticnet=True,
        train_tabpfn=True,
    )
    if mode == "cv":
        result = p.run_cv_pipeline("RB", case.cfg, pd.concat([case.train, case.val]), case.frame)
    else:
        result = p.run_pipeline("RB", case.cfg, case.train, case.val, case.frame)
    keys = ["elasticnet_ranking", "lgbm_ranking", "attn_nn_ranking"]
    if mode != "cv":
        keys.append("tabpfn_ranking")
    for key in keys:
        assert result[key]["season_avg_hit_rate"] == 1.0
    if "sim_results" in result:
        for name, metrics in result["sim_results"]["season_summary"].items():
            if name != "Season Avg":
                assert metrics["mae"] == 0.0


@pytest.mark.parametrize("all_missing", [False, True])
def test_split_cohorts_preserve_and_merge_original_availability(monkeypatch, tmp_path, all_missing):
    from src.shared.evaluation_cohorts import build_cohorts, merge_cohorts

    case = _fake_training(monkeypatch, tmp_path, "RB")
    monkeypatch.setattr(
        p, "build_cohorts", lambda *a, **kw: build_cohorts(*a, **kw, reference=pd.DataFrame())
    )
    case.frame.loc[case.frame.index if all_missing else [0], "rushing_tds"] = np.nan
    results = []
    for cpu in (True, False):
        case.cfg.update(train_ridge=cpu, train_base_nn=not cpu)
        results.append(p.run_pipeline("RB", case.cfg, case.train, case.val, case.frame))
    merged = merge_cohorts(*(result["cohorts"] for result in results))
    for cohort in merged.values():
        assert cohort["evaluation_rows_total"] == 32
        assert cohort["actual_rows_unavailable"] == (32 if all_missing else 1)
        if all_missing:
            assert cohort["status"] == "unavailable"
    if not all_missing:
        assert merged["seasonal_actual_top24"]["n"] == 31
        assert merged["seasonal_actual_top24"]["models"]["Ridge"]["mae"] == 0.0
        assert merged["seasonal_actual_top24"]["models"]["NN"]["mae"] == 0.0


@pytest.mark.parametrize("mode", ["holdout", "cv"])
def test_backtest_excludes_unavailable_truth_and_keeps_output_rows(monkeypatch, tmp_path, mode):
    case = _fake_training(monkeypatch, tmp_path, "RB")
    case.frame.loc[0, "rushing_tds"] = np.nan
    if mode == "cv":
        result = p.run_cv_pipeline("RB", case.cfg, pd.concat([case.train, case.val]), case.frame)
    else:
        result = p.run_pipeline("RB", case.cfg, case.train, case.val, case.frame)
    assert len(result["test_df"]) == 32
    assert np.isnan(result["test_df"].loc[0, "actual_projected_total"])
    assert result["test_df"].loc[0, "fantasy_points"] == 100
    assert result["sim_results"]["season_summary"]["Ridge"]["mae"] == 0.0


def test_cohort_coverage_counts_only_regular_season_rows(monkeypatch):
    from src.shared.evaluation_cohorts import build_cohorts

    frame, truth, cfg = _case("RB")
    report = p._reporting_frame(frame, cfg, truth)
    report["season_type"] = "REG"
    report.loc[31, "season_type"] = "POST"
    report.loc[30, "week"] = 20
    report.loc[0, "actual_projected_total"] = np.nan
    monkeypatch.setattr(
        p, "build_cohorts", lambda *a, **kw: build_cohorts(*a, **kw, reference=pd.DataFrame())
    )
    for cohort in p._reporting_cohorts("RB", report, prior_frames=()).values():
        assert cohort["evaluation_rows_total"] == 30
        assert cohort["actual_rows_unavailable"] == 1
