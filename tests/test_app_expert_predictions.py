"""Unit tests for serving expert-projection enrichment."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.serving.core as core
from src.shared.aggregate_targets import DST_TARGETS, predictions_to_fantasy_points

pytestmark = pytest.mark.unit


def test_serving_core_does_not_import_analysis_package():
    """Production Docker excludes src/analysis, so serving imports must not rely on it."""
    tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == "src.analysis" or name.startswith("src.analysis.") for name in imported)


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["RB1", "K1", "DST1"],
            "player_display_name": ["RB One", "K One", "DST One"],
            "position": ["RB", "K", "DST"],
            "recent_team": ["KC", "KC", "KC"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
        }
    )


def _rb_projection(receptions: float) -> dict[str, np.ndarray]:
    return {
        "rushing_tds": np.array([1.0]),
        "receiving_tds": np.array([0.0]),
        "rushing_yards": np.array([50.0]),
        "receiving_yards": np.array([30.0]),
        "receptions": np.array([receptions]),
        "fumbles_lost": np.array([0.0]),
    }


def _nflcom_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "RB1",
                "season": 2025,
                "week": 1,
                "position": "RB",
                "nflcom_projected_pts": 99.0,
                "rushing_tds": 1.0,
                "receiving_tds": 0.0,
                "rushing_yards": 50.0,
                "receiving_yards": 30.0,
                "receptions": 4.0,
                "fumbles_lost": 0.0,
            },
            {
                "player_id": "K1",
                "season": 2025,
                "week": 1,
                "position": "K",
                "nflcom_projected_pts": 8.5,
            },
        ]
    )


def _rotowire_raw() -> pd.DataFrame:
    dst_row = {
        "player_id": "DST1",
        "season": 2025,
        "week": 1,
        "position": "DST",
    }
    for target in DST_TARGETS:
        dst_row[target] = 0.0
    dst_row["def_sacks"] = 2.0
    dst_row["def_ints"] = 1.0
    dst_row["points_allowed"] = 10.0
    dst_row["yards_allowed"] = 250.0

    return pd.DataFrame(
        [
            {
                "player_id": "RB1",
                "season": 2025,
                "week": 1,
                "position": "RB",
                "rushing_tds": 1.0,
                "receiving_tds": 0.0,
                "rushing_yards": 50.0,
                "receiving_yards": 30.0,
                "receptions": 2.0,
                "fumbles_lost": 0.0,
            },
            dst_row,
        ]
    )


def test_apply_expert_predictions_joins_sources_and_formats():
    results = _results_frame()

    core._apply_expert_predictions(
        results,
        nflcom_loader=lambda seasons: _nflcom_raw(),
        rotowire_loader=lambda seasons: _rotowire_raw(),
    )

    for source in ("nflcom", "rotowire"):
        for scoring in ("ppr", "half_ppr", "standard"):
            assert f"{source}_pred_{scoring}" in results.columns
        assert f"{source}_pred" in results.columns

    rb = results.set_index("player_id").loc["RB1"]
    nfl_ppr = predictions_to_fantasy_points("RB", _rb_projection(4.0), "ppr")[0]
    nfl_standard = predictions_to_fantasy_points("RB", _rb_projection(4.0), "standard")[0]
    rw_ppr = predictions_to_fantasy_points("RB", _rb_projection(2.0), "ppr")[0]
    rw_standard = predictions_to_fantasy_points("RB", _rb_projection(2.0), "standard")[0]
    assert rb["nflcom_pred_ppr"] == pytest.approx(round(nfl_ppr, 2))
    assert rb["nflcom_pred_standard"] == pytest.approx(round(nfl_standard, 2))
    assert rb["rotowire_pred_ppr"] == pytest.approx(round(rw_ppr, 2))
    assert rb["rotowire_pred_standard"] == pytest.approx(round(rw_standard, 2))
    assert rb["nflcom_pred"] == pytest.approx(rb["nflcom_pred_ppr"])
    assert rb["rotowire_pred"] == pytest.approx(rb["rotowire_pred_ppr"])

    k = results.set_index("player_id").loc["K1"]
    assert k["nflcom_pred_ppr"] == pytest.approx(8.5)
    assert k["nflcom_pred_half_ppr"] == pytest.approx(8.5)
    assert np.isnan(k["rotowire_pred_ppr"])

    dst = results.set_index("player_id").loc["DST1"]
    assert np.isnan(dst["nflcom_pred_ppr"])
    assert dst["rotowire_pred_ppr"] == pytest.approx(dst["rotowire_pred_standard"])
    assert np.isfinite(dst["rotowire_pred_ppr"])


def test_apply_expert_predictions_loader_failure_leaves_stable_null_columns():
    results = _results_frame()

    def _boom(*args, **kwargs):
        raise RuntimeError("source down")

    core._apply_expert_predictions(results, nflcom_loader=_boom, rotowire_loader=_boom)

    for source in ("nflcom", "rotowire"):
        for scoring in ("ppr", "half_ppr", "standard"):
            col = f"{source}_pred_{scoring}"
            assert col in results.columns
            assert results[col].isna().all()
        alias = f"{source}_pred"
        assert alias in results.columns
        assert results[alias].isna().all()
