"""Report the modeled components without changing inputs or full fantasy totals."""

import copy
import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.shared import pipeline as p
from src.shared.aggregate_targets import DST_TARGETS, K_TARGETS, POSITION_TARGET_MAP
from src.shared.comparison_scoring import score_actual_components, scoring_components
from src.shared.comparison_truth import ACTUAL_BASIS, ACTUAL_METADATA
from src.shared.position_pipeline import build_pipeline_config
from src.training.context import RunContext
from src.training.contracts import PreparedDataset

pytestmark = pytest.mark.unit

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
RAW_SOURCES = {
    "fumbles_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "fg_yard_points": ("fg_yards_made",),
    "pat_points": ("pat_made",),
    "fg_misses": ("fg_missed",),
    "xp_misses": ("pat_missed",),
}


def _targets(position):
    return list({**POSITION_TARGET_MAP, "K": K_TARGETS, "DST": DST_TARGETS}[position])


def _lead_source(position):
    """The raw column behind the leading configured head, and its scale."""
    lead = _targets(position)[0]
    return RAW_SOURCES.get(lead, (lead,))[0], (10.0 if lead == "fg_yard_points" else 1.0)


def _raw_frame(position):
    """Two synthetic weeks of 16 players with every raw column the builders read."""
    columns = {
        *(name for names in POSITION_TARGET_MAP.values() for name in names),
        *DST_TARGETS,
        *(source for sources in RAW_SOURCES.values() for source in sources),
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fantasy_points",
    }
    frame = pd.DataFrame({column: np.zeros(32) for column in sorted(columns)})
    # The leading configured head separates the true top 12 in every week.
    source, factor = _lead_source(position)
    frame[source] = np.tile(np.arange(16, dtype=float), 2) * factor
    frame["player_id"] = [f"p{i}" for i in range(16)] * 2
    frame["season"] = 2025
    frame["week"] = np.repeat([1, 2], 16)
    frame["position"] = position
    frame["season_type"] = "REG"
    frame["snap_pct"] = 1.0
    return frame


def _config(position):
    config_mod = importlib.import_module(f"src.{position.lower()}.config")
    cfg = build_pipeline_config(position, config_mod.POSITION_CONFIG)
    cfg.update(
        train_ridge=True,
        train_base_nn=True,
        train_attention_nn=False,
        train_elasticnet=False,
        train_lightgbm=False,
        train_tabpfn=False,
    )
    if position in ("K", "DST"):
        cfg.setdefault("compute_adjustment_fn", None)
    return cfg


def _case(position, *, certified=True, missing_rows=()):
    """A frame, its observed targets and a production-shaped configuration.

    ``certified`` runs the canonical target builder, which attaches the
    pre-fill comparison truth exactly as ``_prepare_position_data`` does; the
    uncertified variant leaves the raw frame for the configured-aggregation
    fallback. ``missing_rows`` lose the leading raw observation beforehand.
    """
    frame = _raw_frame(position)
    cfg = _config(position)
    source, _ = _lead_source(position)
    frame.loc[list(missing_rows), source] = np.nan
    if certified:
        frame = cfg["compute_targets_fn"](frame)
    else:
        for target in cfg["targets"]:
            if target in RAW_SOURCES:
                frame[target] = sum(frame[column] for column in RAW_SOURCES[target])
        if position == "K":
            frame["fg_yard_points"] = frame["fg_yards_made"] * 0.1
        frame["fantasy_points"] = cfg["aggregate_fn"](
            {target: frame[target].fillna(0).to_numpy() for target in cfg["targets"]}
        )
    truth = {target: frame[target].to_numpy(dtype=float) for target in cfg["targets"]}
    # An unprojected contribution changes the true top 12 under full scoring.
    frame.loc[[0, 16], "fantasy_points"] += 100
    return frame, truth, cfg


def _expected_truth(frame, truth, cfg, position, certified):
    if certified:
        return score_actual_components(frame, position).to_numpy()
    return cfg["aggregate_fn"](truth)


class ExactModel:
    """Predicts the observed targets exactly, for every trainer contract."""

    truth: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        return copy.deepcopy(self.truth)

    predict_numpy = predict

    def to(self, device):
        return self

    def convergence_report(self):
        return {}


def _fake_training(monkeypatch, tmp_path, position, *, certified=True, missing_rows=()):
    frame, truth, cfg = _case(position, certified=certified, missing_rows=missing_rows)
    train, val = frame.assign(season=2023), frame.assign(season=2024)
    x = np.arange(len(frame) * 2, dtype=np.float32).reshape(-1, 2)
    prepared = PreparedDataset(
        x.copy(),
        x.copy(),
        x.copy(),
        copy.deepcopy(truth),
        copy.deepcopy(truth),
        copy.deepcopy(truth),
        train.copy(),
        val.copy(),
        frame.copy(),
        ("a", "b"),
        "synthetic",
    )
    pristine = copy.deepcopy(prepared)
    seen = {}

    class Exact(ExactModel):
        pass

    Exact.truth = truth

    def nn(*args, **kwargs):
        return (
            Exact(),
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
            prepared.X_train,
            prepared.X_val,
            prepared.y_train,
            prepared.y_val,
            prepared.train,
            prepared.val,
            list(prepared.feature_columns),
        ),
        "_tune_ridge_alphas_cv": lambda *a, **kw: {},
        "RidgeMultiTarget": Exact,
        "_build_lgbm": lambda *a, **kw: Exact(),
        "_train_nn": nn,
        "_train_attention_holdout": lambda *a, **kw: (*nn(), ["a", "b"]),
        "_tune_enet_cv": lambda *a, **kw: {},
        "_train_elasticnet": lambda *a, **kw: (Exact(), copy.deepcopy(truth), nn()[3]),
        "_train_lightgbm": lambda *a, **kw: (Exact(), copy.deepcopy(truth), nn()[3]),
        "_train_tabpfn": lambda *a, **kw: (Exact(), copy.deepcopy(truth), nn()[3]),
        "expanding_window_folds": lambda *a, **kw: [(0, train, val)],
        "_scale_xs": lambda *xs, **kw: (None, xs),
        "make_dataloaders": lambda *a, **kw: (None, None),
        "build_multihead_net": lambda *a, **kw: Exact(),
        "_maybe_compile": lambda model: model,
        "_run_nn_training": lambda **kw: None,
        "build_cohorts": cohorts,
    }.items():
        monkeypatch.setattr(p, name, value)
    # Artifacts and figures are execution effects; the report is what is under test.
    context = replace(
        RunContext.defaults(seed=42),
        output_root=tmp_path,
        artifact_sink=lambda effect: {},
        report_sink=lambda effect: None,
    )
    return SimpleNamespace(
        frame=frame,
        truth=truth,
        cfg=cfg,
        train=train,
        val=val,
        prepared=prepared,
        pristine=pristine,
        seen=seen,
        context=context,
    )


def _run(case, position, mode):
    if mode == "cv":
        return p.run_cv_pipeline(
            position,
            case.cfg,
            pd.concat([case.train, case.val]),
            case.frame,
            context=case.context,
        )
    return p.run_pipeline(
        position, case.cfg, case.train, case.val, case.frame, context=case.context
    )


@pytest.mark.parametrize("certified", [True, False])
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("mode", ["holdout", "cv", "partial_cpu", "partial_nn"])
def test_pipeline_reports_matching_components_without_changing_inputs(
    monkeypatch, tmp_path, position, mode, certified
):
    case = _fake_training(monkeypatch, tmp_path, position, certified=certified)
    before = case.frame.copy(deep=True)
    case.cfg.update(train_base_nn=mode != "partial_cpu", train_ridge=mode != "partial_nn")
    result = _run(case, position, mode)
    report = case.seen["report"]
    expected = _expected_truth(before, case.truth, case.cfg, position, certified)
    np.testing.assert_allclose(report["actual_projected_total"], expected)
    pd.testing.assert_series_equal(report["fantasy_points"], before["fantasy_points"])
    pd.testing.assert_frame_equal(case.frame, before)
    for actual, original in zip(case.prepared, case.pristine, strict=True):
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
        # Pipeline totals retain ordinary fantasy scoring for every position.
        np.testing.assert_allclose(
            result["test_df"]["pred_ridge_total"], case.cfg["aggregate_fn"](case.truth)
        )
        # The season-average comparator must use the same components as its truth.
        np.testing.assert_allclose(result["test_df"]["pred_baseline"].iloc[16:], expected[:16])
    assert report.attrs[ACTUAL_METADATA] == {
        "basis": ACTUAL_BASIS,
        "targets": list(scoring_components(position)) if certified else case.cfg["targets"],
        "scoring_format": "ppr",
    }


@pytest.mark.parametrize("position", POSITIONS)
def test_original_missing_component_is_not_made_available_by_target_fill(position):
    frame, truth, cfg = _case(position, missing_rows=[0])
    # Feature construction may reorder or renumber rows; the mask travels with the row.
    frame = frame.iloc[::-1]
    frame.index = range(len(frame))
    report = p._reporting_frame(frame, cfg, truth, position=position)
    assert np.isnan(report.loc[31, "actual_projected_total"])
    assert report.loc[:30, "actual_projected_total"].notna().all()
    assert np.isfinite(truth[cfg["targets"][0]]).all()


def test_derived_fumble_requires_all_original_components():
    frame = _raw_frame("RB")
    cfg = _config("RB")
    frame.loc[0, "sack_fumbles_lost"] = np.nan
    frame = cfg["compute_targets_fn"](frame)
    truth = {target: frame[target].to_numpy(dtype=float) for target in cfg["targets"]}
    report = p._reporting_frame(frame, cfg, truth, position="RB")
    assert np.isnan(report.loc[0, "actual_projected_total"])
    assert report.loc[1:, "actual_projected_total"].notna().all()
    assert truth["fumbles_lost"][0] == 0.0


def test_reduced_custom_targets_and_unavailable_values():
    frame = pd.DataFrame({"fantasy_points": [999.0, 999.0, 999.0]})
    cfg = {"targets": ["a", "b"], "aggregate_fn": lambda y: y["a"] - 2 * y["b"]}
    truth = {"a": np.array([5.0, np.nan, 3.0]), "b": np.array([1.0, 1.0, np.inf])}
    report = p._reporting_frame(frame, cfg, truth, position="RB")
    assert report["actual_projected_total"].iloc[0] == 3.0
    assert report["actual_projected_total"].iloc[1:].isna().all()
    assert report.attrs[ACTUAL_METADATA]["scoring_format"] is None
    partial = p._reporting_frame(frame, cfg, {"a": truth["a"]}, position="RB")
    assert partial["actual_projected_total"].isna().all()
    no_agg = p._reporting_frame(frame, {"targets": ["a"]}, truth, position="RB")
    assert no_agg["actual_projected_total"].isna().all()


def test_reduced_canonical_targets_keep_the_configured_basis():
    """A configured subset of heads is scored on that subset, masked by raw availability."""
    frame, truth, cfg = _case("RB", missing_rows=[0])
    reduced = {**cfg, "targets": cfg["targets"][:2]}
    report = p._reporting_frame(frame, reduced, truth, position="RB")
    expected = cfg["aggregate_fn"]({target: truth[target] for target in reduced["targets"]})
    assert np.isnan(report.loc[0, "actual_projected_total"])
    np.testing.assert_allclose(report.loc[1:, "actual_projected_total"], expected[1:])
    assert report.attrs[ACTUAL_METADATA]["targets"] == reduced["targets"]


def test_certified_dst_truth_rescores_model_totals_without_points_allowed():
    frame, truth, cfg = _case("DST")
    report = p._reporting_frame(frame, cfg, truth, position="DST")
    report["pred_ridge_total"] = cfg["aggregate_fn"](truth)
    for target in cfg["targets"]:
        report[f"pred_ridge_{target}"] = truth[target]
    scored = p._reporting_scored(report, "DST")
    np.testing.assert_allclose(scored["pred_ridge_total"], scored["actual_projected_total"])
    assert not np.allclose(report["pred_ridge_total"], report["actual_projected_total"])
    assert p._reporting_ranking(report, "DST", "pred_ridge_total")["season_avg_hit_rate"] == 1.0


def test_partial_fixture_without_aggregator_remains_supported(monkeypatch, tmp_path):
    case = _fake_training(monkeypatch, tmp_path, "RB")
    case.cfg.pop("aggregate_fn")
    case.cfg["train_ridge"] = False
    result = _run(case, "RB", "partial_nn")
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

    class ExactLGBM(ExactModel):
        pass

    ExactLGBM.truth = truth
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
    result = _run(case, "RB", mode)
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
def test_split_cohorts_preserve_and_combine_original_availability(
    monkeypatch, tmp_path, all_missing
):
    from src.shared.evaluation_cohorts import build_cohorts, merge_cohorts

    missing = range(32) if all_missing else [0]
    case = _fake_training(monkeypatch, tmp_path, "RB", missing_rows=missing)
    monkeypatch.setattr(
        p, "build_cohorts", lambda *a, **kw: build_cohorts(*a, **kw, reference=pd.DataFrame())
    )
    results = []
    for cpu in (True, False):
        case.cfg.update(train_ridge=cpu, train_base_nn=not cpu)
        results.append(_run(case, "RB", "partial_cpu" if cpu else "partial_nn"))
    combined = merge_cohorts(*(result["cohorts"] for result in results))
    for cohort in combined.values():
        assert cohort["evaluation_rows_total"] == 32
        assert cohort["actual_rows_unavailable"] == (32 if all_missing else 1)
        if all_missing:
            assert cohort["status"] == "unavailable"
    if not all_missing:
        assert combined["seasonal_actual_top24"]["n"] == 31
        assert combined["seasonal_actual_top24"]["models"]["Ridge"]["mae"] == 0.0
        assert combined["seasonal_actual_top24"]["models"]["NN"]["mae"] == 0.0


@pytest.mark.parametrize("mode", ["holdout", "cv"])
def test_backtest_excludes_unavailable_truth_and_keeps_output_rows(monkeypatch, tmp_path, mode):
    case = _fake_training(monkeypatch, tmp_path, "RB", missing_rows=[0])
    result = _run(case, "RB", mode)
    assert len(result["test_df"]) == 32
    assert np.isnan(result["test_df"].loc[0, "actual_projected_total"])
    assert result["test_df"].loc[0, "fantasy_points"] == 100
    assert result["sim_results"]["season_summary"]["Ridge"]["mae"] == 0.0


def test_cohort_coverage_counts_only_regular_season_rows(monkeypatch):
    from src.shared.evaluation_cohorts import build_cohorts

    frame, truth, cfg = _case("RB")
    report = p._reporting_frame(frame, cfg, truth, position="RB")
    report.loc[31, "season_type"] = "POST"
    report.loc[30, "week"] = 20
    report.loc[0, "actual_projected_total"] = np.nan
    monkeypatch.setattr(
        p, "build_cohorts", lambda *a, **kw: build_cohorts(*a, **kw, reference=pd.DataFrame())
    )
    for cohort in p._reporting_cohorts("RB", report, prior_frames=()).values():
        assert cohort["evaluation_rows_total"] == 30
        assert cohort["actual_rows_unavailable"] == 1
