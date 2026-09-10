"""Real reporting adapters preserve full labels and match expert scoring."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_tabpfn_benchmark as tab
from src.analysis import attn_weekly_accuracy as weekly
from src.analysis import tier_expert_comparison as tier
from src.analysis.analysis_expert_comparison import ExpertSource
from src.analysis.comparison_data import PROJECTED_ACTUAL, PROJECTED_METADATA, comparison_actuals
from src.shared.comparison_scoring import scoring_components

pytestmark = pytest.mark.unit


def _frame():
    frame = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "receiving_yards": [50.0, 20.0, 70.0, 0.0],
            "receptions": [5.0, 2.0, 4.0, 0.0],
            "receiving_tds": 0.0,
            "fumbles_lost": 0.0,
            "fantasy_points": [10.6, 5.1, 11.7, 0.9],
        }
    )
    for model in ("ridge", "nn", "attn_nn", "lgbm", "tabpfn"):
        frame[f"pred_{model}_total"] = [10.0, 4.0, 11.0, 0.0]
        for target in scoring_components("WR"):
            frame[f"pred_{model}_{target}"] = frame[target]
    return frame


def _runner(monkeypatch, frame):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"test_df": frame.copy()}

    monkeypatch.setitem(sys.modules, "src.wr.run_pipeline", SimpleNamespace(CONFIG={}, run=run))
    return calls


def _expert(frame):
    def project(raw, pos, scoring):
        return raw[["player_id", "season", "week"]].assign(
            expert_pred_total=comparison_actuals(raw, pos, scoring)
        )

    return ExpertSource("fixture", "Fixture", lambda _: frame, project)


def test_tabpfn_matched_report_and_versioned_cache_preserve_full_labels(tmp_path, monkeypatch):
    frame = _frame()
    calls = _runner(monkeypatch, frame)
    frame.to_parquet(tmp_path / "tdf_WR_seed42.parquet", index=False)
    captured = tab.run_position("WR", seed=42, cache_dir=str(tmp_path))
    assert len(calls) == 1, "legacy caches do not establish comparison truth"
    assert captured.fantasy_points.tolist() == frame.fantasy_points.tolist()
    assert captured[PROJECTED_ACTUAL].tolist() == [10.0, 4.0, 11.0, 0.0]
    resumed = tab.run_position("WR", seed=42, cache_dir=str(tmp_path))
    assert len(calls) == 1
    expert = _expert(frame).project(frame, "WR", "ppr")
    report = tab.rotowire_matched_metrics(resumed, expert)
    assert report["_n_matched"] == 4
    assert report["Ridge"]["regular"]["mae"] == 0.0
    assert report["RotoWire"]["regular"]["mae"] == 0.0
    assert tab.position_metrics(resumed)["Ridge"]["regular"]["mae"] == pytest.approx(0.825)


@pytest.mark.parametrize(
    "scoring,expected",
    [
        ("ppr", [10.0, 4.0, 11.0, 0.0]),
        ("half_ppr", [7.5, 3.0, 9.0, 0.0]),
        ("standard", [5.0, 2.0, 7.0, 0.0]),
    ],
)
def test_weekly_expert_copy_rescores_all_sources_without_mutating_full_report(
    monkeypatch, tmp_path, scoring, expected
):
    from src.analysis import analysis_expert_comparison as aec
    from src.analysis import sleeper_loader
    from src.data import nflcom_loader

    frame = _frame()
    before = frame.copy()
    calls = _runner(monkeypatch, frame)
    frame.to_parquet(tmp_path / "wr_seed42.parquet", index=False)
    captured = weekly.collect(["WR"], [42], str(tmp_path))
    assert len(calls) == 1
    resumed = weekly.collect(["WR"], [42], str(tmp_path))
    assert len(calls) == 1
    assert resumed[PROJECTED_ACTUAL].tolist() == [10.0, 4.0, 11.0, 0.0]
    source = _expert(frame)
    monkeypatch.setattr(nflcom_loader, "load_nflcom_with_gsis_id", lambda **_: frame)
    monkeypatch.setattr(sleeper_loader, "load_sleeper_with_gsis_id", lambda **_: frame)
    monkeypatch.setattr(aec, "_project_nflcom_expert", source.project)
    monkeypatch.setattr(aec, "_project_sleeper_to_ppr", source.project)
    report = weekly.attach_experts(resumed, scoring)
    assert report.fantasy_points.tolist() == expected
    assert report.pred_ridge_total.tolist() == expected
    assert report.pred_sleeper_total.tolist() == expected
    assert captured.fantasy_points.tolist() == before.fantasy_points.tolist()
    pd.testing.assert_frame_equal(frame, before)


def test_tier_comparison_uses_shared_truth_and_requires_complete_components(capsys):
    frame = _frame()
    source = _expert(frame)
    prior = pd.Series({(pid, 2025): i for i, pid in enumerate(frame.player_id)})
    tier.compare_position("WR", frame, prior, [source], {"fixture": frame}, tier_topn=2)
    report = capsys.readouterr().out
    assert "elite_top_drafted  (n=2)" in report and "field  (n=2)" in report
    lines = [
        line.split()
        for line in report.splitlines()
        if line.strip().startswith(("Ridge", "Fixture"))
    ]
    assert len(lines) == 4
    assert all(float(line[-2]) == 0.0 for line in lines)
    incomplete = frame.drop(columns="receptions")
    tier.compare_position("WR", incomplete, prior, [source], {"fixture": frame}, tier_topn=2)
    assert "no matched player-weeks" in capsys.readouterr().out


def test_pipeline_truth_requires_matching_metadata_and_never_full_score_fallback():
    frame = _frame()
    frame[PROJECTED_ACTUAL] = -999.0
    assert comparison_actuals(frame, "WR").tolist() == [10.0, 4.0, 11.0, 0.0]
    frame.attrs[PROJECTED_METADATA] = {
        "basis": "configured_target_aggregation_v1",
        "targets": list(scoring_components("WR")),
        "scoring_format": "half_ppr",
    }
    assert comparison_actuals(frame, "WR").tolist() == [10.0, 4.0, 11.0, 0.0]
    frame.attrs[PROJECTED_METADATA]["scoring_format"] = "ppr"
    frame[PROJECTED_ACTUAL] = [10.0, 4.0, np.nan, 0.0]
    assert pd.isna(comparison_actuals(frame, "WR", "standard").iloc[2])
    slim = frame.drop(columns=list(scoring_components("WR")))
    assert comparison_actuals(slim, "WR").iloc[:2].tolist() == [10.0, 4.0]
    assert pd.isna(comparison_actuals(slim, "WR").iloc[2])
    slim.attrs.clear()
    assert comparison_actuals(slim, "WR").isna().all()
    expert = _expert(frame).project(frame, "WR", "ppr")
    assert (
        tab.rotowire_matched_metrics(slim, expert)["_unavailable"]
        == "shared_actual_components_missing"
    )


def test_actual_pipeline_reporting_metadata_is_consumable():
    from src.shared.pipeline import _reporting_frame
    from src.shared.registry import get_config

    frame = _frame()
    for column in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"):
        frame[column] = 0.0
    cfg = get_config("WR")
    targets = {name: frame[name].to_numpy() for name in cfg["targets"]}
    reported = _reporting_frame(frame, cfg, targets, source_frame=frame)
    slim = reported.drop(columns=list(scoring_components("WR")))
    assert comparison_actuals(slim, "WR").tolist() == [10.0, 4.0, 11.0, 0.0]
    assert reported.fantasy_points.tolist() == frame.fantasy_points.tolist()
    source = frame.copy()
    source.loc[0, "receptions"] = np.nan
    missing = _reporting_frame(frame, cfg, targets, source_frame=source)
    assert pd.isna(comparison_actuals(missing, "WR", "standard").iloc[0])
    assert comparison_actuals(missing, "WR", "standard").iloc[1:].tolist() == [2.0, 7.0, 0.0]
