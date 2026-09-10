"""Causal baselines for the kicker signal-floor diagnostic."""

import json
import sys

import pandas as pd
import pytest

from src.analysis import analysis_k_signal_floor as floor

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("window", [None, 8])
def test_kicker_baseline_never_carries_another_players_future(window):
    frame = pd.DataFrame(
        {
            "player_id": ["A", "A", "B"],
            "season": [2023, 2025, 2024],
            "week": [1, 1, 1],
            "fantasy_points": [10.0, 1000.0, 2.0],
        },
        index=[7, 13, 2],
    )
    actual = floor._grouped_rolling_mean(
        frame, "player_id", ["player_id", "season", "week"], window, fill=5.0
    )
    assert actual.tolist() == [5.0, 10.0, 5.0]
    changed = frame.copy()
    changed.loc[13, "fantasy_points"] = 1000000.0
    after = floor._grouped_rolling_mean(
        changed, "player_id", ["player_id", "season", "week"], window, fill=5.0
    )
    pd.testing.assert_series_equal(actual, after)


@pytest.mark.parametrize("external", [False, True])
def test_cli_accepts_benchmark_path_outside_project(monkeypatch, tmp_path, external):
    project = tmp_path / "project"
    project.mkdir()
    benchmark = (tmp_path if external else project) / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "position": "K",
                        "ridge_mae": 3.0,
                        "nn_mae": 3.0,
                        "attn_nn_mae": 3.0,
                        "lgbm_mae": 3.0,
                    }
                ]
            }
        )
    )
    frame = pd.DataFrame(
        {
            "player_id": ["A"] * 3,
            "recent_team": ["KC"] * 3,
            "season": [2023, 2024, 2025],
            "week": [1] * 3,
            "fantasy_points": [2.0, 3.0, 4.0],
            "feature": [1.0] * 3,
        }
    )
    monkeypatch.setattr(floor, "PROJECT_ROOT", project)
    monkeypatch.setattr(floor, "OUT_DIR", project / "output")
    monkeypatch.setattr(floor, "OUT_JSON", project / "output/result.json")
    monkeypatch.setattr(floor, "OUT_PNG", project / "output/result.png")
    monkeypatch.setattr(floor, "ALL_FEATURES", ["feature"])
    monkeypatch.setattr(floor, "load_data", lambda: frame)
    monkeypatch.setattr(floor, "compute_targets", lambda data: data)
    monkeypatch.setattr(floor, "compute_features", lambda data: None)
    monkeypatch.setattr(
        floor,
        "season_split",
        lambda data: tuple(data[data["season"].eq(s)] for s in (2023, 2024, 2025)),
    )
    monkeypatch.setattr(
        floor,
        "_refit_ridge",
        lambda tr, va, te, cols: (va["fantasy_points"].to_numpy(), te["fantasy_points"].to_numpy()),
    )
    monkeypatch.setattr(floor, "_plot_segments", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["floor", "--quick", "--benchmark-path", str(benchmark)])
    assert floor.main() == 0
    result = json.loads(floor.OUT_JSON.read_text())
    assert result["models_from_benchmark"]["source"] == (
        "../benchmark.json" if external else "benchmark.json"
    )
