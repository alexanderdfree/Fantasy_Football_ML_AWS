"""Analysis forecasts and truth use the same requested scoring components."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import artifact_eval, topn_expert_gap
from src.shared.comparison_scoring import scoring_components

pytestmark = pytest.mark.unit

_CASES = {
    "QB": ({"passing_yards": 250, "passing_tds": 2, "rushing_yards": 20}, [20, 20, 20]),
    "RB": ({"rushing_yards": 50, "receiving_yards": 30, "receptions": 4}, [12, 10, 8]),
    "WR": ({"receiving_yards": 100, "receptions": 10}, [20, 15, 10]),
    "TE": ({"receiving_yards": 40, "receptions": 3}, [7, 5.5, 4]),
    "K": ({"fg_yard_points": 6, "pat_points": 2, "fg_misses": 1}, [7, 7, 7]),
    "DST": ({"def_sacks": 2, "def_ints": 1, "points_allowed": 21, "yards_allowed": 350}, [3, 3, 3]),
}
_FORMATS = ("ppr", "half_ppr", "standard")
_FULL_PPR = {"QB": 21.5, "RB": 18.0, "WR": 31.0, "TE": 8.0, "K": 7.0, "DST": 3.0}
_UNPROJECTED = {
    "QB": {"receptions": 1.0, "receiving_yards": 5.0},
    "RB": {"passing_yards": 50.0, "passing_tds": 1.0},
    "WR": {"rushing_yards": 50.0, "rushing_tds": 1.0},
    "TE": {"rushing_yards": 10.0},
}


def _frame(position):
    values, _ = _CASES[position]
    raw = {name: float(values.get(name, 0)) for name in scoring_components(position)}
    row = {
        "player_id": "player",
        "position": position,
        "season": 2025,
        "week": 1,
        "feature": 1.0,
        "fantasy_points": _FULL_PPR[position],
        **_UNPROJECTED.get(position, {}),
        **raw,
    }
    for model, _, total in topn_expert_gap.MODEL_SOURCES:
        row[total] = _CASES[position][1][0]
        row.update({f"pred_{model}_{name}": value for name, value in raw.items()})
    return pd.DataFrame([row])


@pytest.mark.parametrize("position", list(_CASES))
@pytest.mark.parametrize("scoring_format", _FORMATS)
def test_fresh_predictions_rescore_both_truth_and_forecasts(monkeypatch, position, scoring_format):
    source = _frame(position)
    monkeypatch.setitem(
        sys.modules,
        f"src.{position.lower()}.run_pipeline",
        SimpleNamespace(run=lambda: {"test_df": source}),
    )
    actual = topn_expert_gap._fresh_model_predictions(position, [2025], scoring_format)
    expected = _CASES[position][1][_FORMATS.index(scoring_format)]
    assert actual["fantasy_points"].iloc[0] == expected
    for _, _, total in topn_expert_gap.MODEL_SOURCES:
        assert actual[total].iloc[0] == expected
    assert source["fantasy_points"].iloc[0] == _FULL_PPR[position]


@pytest.mark.parametrize("position", list(_CASES))
@pytest.mark.parametrize("scoring_format", _FORMATS)
def test_artifact_truth_uses_the_same_components_as_its_predictions(
    monkeypatch, tmp_path, position, scoring_format
):
    source = _frame(position)
    targets = list(scoring_components(position))
    reg = {
        "targets": targets,
        "filter_fn": lambda frame: frame,
        "compute_targets_fn": lambda frame: frame,
        "get_feature_columns_fn": lambda: ["feature"],
        "min_games_per_season": 0,
        "model_dir": str(tmp_path),
        "nn_file": "absent.pt",
        "nn_kwargs": {},
    }
    monkeypatch.setattr(artifact_eval, "INFERENCE_REGISTRY", {position: reg})
    monkeypatch.setattr(
        artifact_eval, "build_position_features", lambda tr, va, te, *args, **kw: (tr, va, te)
    )

    class Ridge:
        def __init__(self, **kwargs):
            pass

        def load(self, directory):
            pass

        def predict(self, X):
            return {target: source[target].to_numpy() for target in targets}

    monkeypatch.setattr(artifact_eval, "RidgeMultiTarget", Ridge)
    actual = artifact_eval.build_test_df_from_artifacts(
        position, source, source, source, scoring_format=scoring_format, model_dir=str(tmp_path)
    )
    expected = _CASES[position][1][_FORMATS.index(scoring_format)]
    assert actual["pred_ridge_total"].iloc[0] == expected
    assert actual["fantasy_points"].iloc[0] == expected


def test_missing_components_are_unavailable_instead_of_full_total_fallback(monkeypatch):
    frame = _frame("WR").drop(columns=["receptions", "pred_ridge_receptions"])
    monkeypatch.setitem(
        sys.modules, "src.wr.run_pipeline", SimpleNamespace(run=lambda: {"test_df": frame})
    )
    actual = topn_expert_gap._fresh_model_predictions("WR", [2025], "standard")
    assert np.isnan(actual["fantasy_points"].iloc[0])
    assert np.isnan(actual["pred_ridge_total"].iloc[0])
    metrics, _, _, coverage = topn_expert_gap.build_position_report(
        "WR", frame, expert_raws={}, experts=[], scoring_format="standard", n_boot=10, seed=0
    )
    assert metrics and all(row["n_rows"] == 0 for row in metrics)
    assert coverage and all(row["skipped"] for row in coverage)
