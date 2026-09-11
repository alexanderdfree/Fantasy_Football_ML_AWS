"""Adversarial checks for identical truth, populations, and source eligibility."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import tier_expert_comparison, topn_expert_gap
from src.analysis.analysis_expert_comparison import ExpertSource, _compare_one_position
from src.analysis.comparison_frames import common_forecast_frames
from src.serving.comparison import comparison_tables
from src.serving.expert_sources import project_nflcom_to_fantasy
from src.shared.expert_eligibility import eligible_forecast_rows

pytestmark = pytest.mark.unit


def _rows(n=30):
    score = np.arange(n, 0, -1, dtype=float)
    return pd.DataFrame(
        {
            "player_id": [f"p{i:02}" for i in range(n)],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "fantasy_points": score,
            "receiving_yards": score * 10,
            "receptions": 0.0,
            "receiving_tds": 0.0,
            "fumbles_lost": 0.0,
            "pred_attn_nn_total": score,
            "pred_ridge_total": score,
        }
    )


def _expert(name="example"):
    return ExpertSource(name=name, label=name, load=lambda _: None, project=lambda raw, *_: raw)


def _projections(frame):
    return frame[["player_id", "season", "week"]].assign(
        expert_pred_total=frame.receiving_yards / 10
    )


def _report(frame, forecasts, experts=None):
    experts = experts or [_expert()]
    return topn_expert_gap.build_position_report(
        "WR",
        frame,
        expert_raws={src.name: forecasts for src in experts},
        experts=experts,
        scoring_format="ppr",
        n_boot=20,
        seed=42,
    )


def _global(rows, top_n=12):
    return [
        r
        for r in rows
        if r["metric_group"] == "cohort_error"
        and r["slice_family"] == "global"
        and r["top_n"] == top_n
    ]


def test_topn_ignores_unprojected_actuals_and_playoffs():
    frame = _rows()
    expected = _report(frame, _projections(frame))
    changed = frame.assign(fantasy_points=frame.fantasy_points + 1000, rushing_yards=10000.0)
    playoff = changed.iloc[[0]].assign(week=19, season_type="POST", receiving_yards=99999.0)
    actual = _report(pd.concat([changed, playoff]), _projections(frame))
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert all(r["mae"] == 0 for r in _global(actual[0]))


def test_missing_expert_forecast_excludes_hard_row_for_every_model_without_promotion():
    frame = _rows()
    frame.loc[0, ["pred_attn_nn_total", "pred_ridge_total"]] = 10000.0
    forecasts = _projections(frame).iloc[1:]
    rows, _, _, coverage = _report(frame, forecasts)
    assert all(r["n_rows"] == 11 and r["mae"] == 0 for r in _global(rows))
    assert all(c["n_common"] == 29 and c["actual_n"] == 30 for c in coverage)
    assert not any(r.get("slice_family") == "projected_rank_bucket" for r in rows)


def test_models_with_different_rankings_use_identical_error_slices():
    frame = _rows()
    frame["pred_ridge_total"] = frame.pred_ridge_total.iloc[::-1].to_numpy()
    rows, _, _, _ = _report(frame, _projections(frame))
    groups = {}
    for r in rows:
        if r["metric_group"] == "cohort_error":
            groups.setdefault((r["top_n"], r["slice_family"], r["slice_name"]), []).append(
                r["n_rows"]
            )
    assert all(len(set(counts)) == 1 for counts in groups.values())
    assert any(family == "seasonal_actual_top24" for _, family, _ in groups)
    assert not any(family == "elite_top24" for _, family, _ in groups)


def test_shared_frames_keep_zero_predictions_and_exclude_nan_and_inf():
    first = _rows(3).assign(pred_total=[0.0, 1.0, 2.0])
    second = first.assign(pred_total=[0.0, np.inf, np.nan])
    a, b = common_forecast_frames([first, second])
    assert list(a.player_id) == list(b.player_id) == ["p00"]


def test_disjoint_sources_make_all_metrics_unavailable():
    frame = _rows(3).assign(pred_total=[0.0, np.nan, np.nan])
    other = frame.assign(pred_total=[np.nan, 1.0, 2.0])
    assert all(f.empty for f in common_forecast_frames([frame, other]))


def test_missing_component_cannot_fall_back_to_full_fantasy_actuals():
    frame = _rows().drop(columns="fumbles_lost")
    rows, _, _, coverage = _report(frame, _projections(frame))
    assert all(r["n_rows"] == 0 for r in _global(rows))
    assert all(c["actual_n"] == 0 for c in coverage)


def test_tiers_use_shared_actuals_and_identical_sources(capsys):
    frame = _rows(4)
    frame.loc[0, ["pred_attn_nn_total", "pred_ridge_total"]] = 10000.0
    frame["fantasy_points"] += 1000
    prior = pd.Series({(f"p{i:02}", 2025): 100 - i for i in range(4)})
    tier_expert_comparison.compare_position(
        "WR",
        frame,
        prior,
        [_expert()],
        {"example": _projections(frame).iloc[1:]},
        tier_topn=2,
    )
    output = capsys.readouterr().out
    assert "elite_top_drafted  (n=1)" in output
    assert "3 common / 4 covered / 4 actual rows" in output
    assert "1000.000" not in output
    assert output.count("0.000") == 12


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_nflcom_offense_excludes_backfilled_and_unknown_seasons(position):
    frame = pd.DataFrame({"season": [2023, 2024, 2025, None]})
    assert eligible_forecast_rows(frame, "nflcom", position).tolist() == [False, True, True, False]
    assert eligible_forecast_rows(frame, "espn", position).all()


def test_serving_projector_and_metric_boundary_reject_old_cached_nflcom_totals():
    raw = _rows(2).assign(season=[2023, 2024])
    projected = project_nflcom_to_fantasy(raw, "WR")
    assert projected.season.tolist() == [2024]
    cached = raw.rename(
        columns={
            c: f"actual_{c}"
            for c in ["receiving_yards", "receptions", "receiving_tds", "fumbles_lost"]
        }
    )
    cached["nflcom_pred_ppr"] = 9999.0
    cached["nflcom_comparison_pred_ppr"] = 9999.0
    cached["ridge_pred_ppr"] = cached.fantasy_points
    tables, coverage, _, _ = comparison_tables(cached, reference=pd.DataFrame())
    assert tables["all"]["WR"]["nflcom"]["n"] == 1
    assert coverage["all"]["WR"]["n"] == 1


def test_offline_preprojected_totals_cannot_bypass_backfill_rule():
    old = _rows(3).assign(season=2023)
    result = _compare_one_position("WR", old, _projections(old), "nflcom", [2023], "ppr", 20, 42)
    assert result["skipped"]
    current = old.assign(season=2024)
    result = _compare_one_position(
        "WR", current, _projections(current), "nflcom", [2024], "ppr", 20, 42
    )
    assert result["expert"]["mae"] == 0
    rows, _, _, coverage = _report(old, _projections(old), [_expert("nflcom")])
    assert not any(r["source"] == "nflcom" for r in rows)
    assert next(c for c in coverage if c["source"] == "nflcom")["skipped"]


def test_dst_expert_comparison_preserves_normal_forecasts_and_cache_rounding(app_module):
    from src.analysis.analysis_expert_comparison import _build_experts
    from src.serving.core import _apply_expert_predictions
    from src.shared.aggregate_targets import DST_TARGETS

    raw = pd.DataFrame({name: [0.0, 0.0] for name in DST_TARGETS})
    raw = raw.assign(
        player_id=["BUF", "KC"],
        position="DST",
        season=2025,
        week=1,
        def_sacks=3.0,
        yards_allowed=[349.9999, 350.0],
        points_allowed=[0.0, 44.0],
    )
    sources = _build_experts(None, None)
    for source in sources:
        if source.name in {"sleeper", "espn"}:
            assert source.project(raw, "DST", "ppr").expert_pred_total.tolist() == [3.0, 2.0]
    results = raw[["player_id", "position", "season", "week"]].copy()
    results["fantasy_points"] = 0.0
    _apply_expert_predictions(
        results,
        nflcom_loader=lambda **_: pd.DataFrame(),
        rotowire_loader=lambda _: raw,
        espn_loader=lambda _: raw,
    )
    for source in ["rotowire", "espn"]:
        assert results[f"{source}_pred_ppr"].tolist() == [13.0, -2.0]
        assert results[f"{source}_pred_comparison"].tolist() == [3.0, 2.0]
    for column in DST_TARGETS:
        results[f"actual_{column}"] = raw[column]
    results["ridge_pred_comparison"] = [3.0, 2.0]
    results["ridge_pred_ppr"] = [9999.0, -9999.0]
    tables, _, _, _ = comparison_tables(results, reference=pd.DataFrame())
    assert all(tables["all"]["DST"][source]["mae"] == 0 for source in ["ridge", "rotowire", "espn"])
    stale = results.drop(columns=[c for c in results if c.endswith("_pred_comparison")])
    tables, coverage, _, _ = comparison_tables(stale, reference=pd.DataFrame())
    assert coverage["all"]["DST"]["status"] == "unavailable"
    assert all(value is None for value in tables["all"]["DST"].values())


def test_disjoint_experts_do_not_turn_missing_comparison_into_seasonal_losses():
    frame = _rows(2)
    experts = [_expert("a"), _expert("b")]
    metrics, season_misses, weekly_misses, coverage = topn_expert_gap.build_position_report(
        "WR",
        frame,
        expert_raws={"a": _projections(frame).iloc[:1], "b": _projections(frame).iloc[1:]},
        experts=experts,
        scoring_format="ppr",
        n_boot=20,
        seed=42,
    )
    assert all(c["status"] == "unavailable" and c["n_common"] == 0 for c in coverage)
    assert all(r["n_rows"] == 0 for r in _global(metrics))
    assert not any(r["metric_group"] in {"season_selection", "weekly_selection"} for r in metrics)
    assert not season_misses and not weekly_misses
    summary = topn_expert_gap.render_summary(
        pd.DataFrame(metrics),
        pd.DataFrame(coverage),
        {"generated_at": "test", "eval_seasons": [2025], "scoring_format": "ppr"},
    )
    assert "unavailable" in summary and "n_common" in summary
    assert "no_common_finite_forecasts" in summary


def test_local_dst_snapshot_rescores_components_or_reports_unavailable(tmp_path):
    from src.shared.aggregate_targets import DST_TARGETS, predictions_to_fantasy_points

    raw = pd.DataFrame({target: [0.0, 0.0] for target in DST_TARGETS}).assign(
        player_id=["BUF", "KC"],
        position="DST",
        season=2025,
        week=1,
        def_sacks=[0.0, 1.0],
        yards_allowed=350.0,
    )
    raw["projected_points"] = predictions_to_fantasy_points(
        "DST",
        {target: raw[target].to_numpy() for target in DST_TARGETS},
    )
    native = raw.projected_points.copy()
    source = topn_expert_gap.local_expert_source(
        topn_expert_gap.LocalExpertSpec("snapshot", tmp_path / "snapshot.csv")
    )
    projected = source.project(raw, "DST", "ppr")
    # Negative and zero comparison forecasts remain forecasts. The PA bonus
    # present in the native totals cannot enter the restricted comparison.
    assert projected.expert_pred_total.tolist() == [-1.0, 0.0]
    pd.testing.assert_series_equal(raw.projected_points, native)
    totals_only = raw[["player_id", "position", "season", "week", "projected_points"]]
    _, result, status = topn_expert_gap.expert_source_frame(source, totals_only, "DST", raw, "ppr")
    assert result is None and status["skipped"]
    assert "nine shared raw components" in status["reason"]
