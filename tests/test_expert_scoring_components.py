"""Forecast components must survive ingestion, serving, and offline grading."""

from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from src.analysis.analysis_expert_comparison import _project_sleeper_to_ppr
from src.analysis.analysis_nflcom_baseline import _project_nflcom_to_ppr
from src.data import nflcom_loader
from src.data.nflcom_loader import _normalize_one_position
from src.scripts.build_evaluation_reference import build_reference
from src.serving.core import _project_rotowire_to_fantasy
from src.serving.espn_projections import project_espn_to_fantasy
from src.serving.expert_sources import project_nflcom_to_fantasy, score_offensive_projections
from src.shared.evaluation_cohorts import REFERENCE_VERSION, reference_selection

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
@pytest.mark.parametrize("fmt,rec_weight", [("ppr", 1), ("half_ppr", 0.5), ("standard", 0)])
@pytest.mark.parametrize(
    "project,column",
    [
        (project_nflcom_to_fantasy, "nflcom_pred_total"),
        (_project_rotowire_to_fantasy, "rotowire_pred_total"),
        (project_espn_to_fantasy, "espn_pred_total"),
        (_project_nflcom_to_ppr, "nflcom_pred_total"),
        (_project_sleeper_to_ppr, "expert_pred_total"),  # RotoWire and FFToday
    ],
)
def test_all_offensive_components_survive_every_projector(
    position, fmt, rec_weight, project, column
):
    frame = pd.DataFrame(
        [
            {
                "player_id": "P1",
                "season": 2025,
                "week": 3,
                "position": position,
                "passing_yards": 20,
                "passing_tds": 1,
                "interceptions": 0.2,
                "rushing_yards": 13.71,
                "rushing_tds": 0.18,
                "receiving_yards": 65.44,
                "receiving_tds": 0.34,
                "receptions": 4.67,
                "fumbles_lost": 0.04,
                "nflcom_projected_pts": 999,
                "two_point_conversions": 99,  # not part of the app's scoring rules
            }
        ],
        index=[42],
    )
    # Passing 4.4 + rushing 2.451 + receiving 8.584 - fumbles 0.08 + receptions.
    assert project(frame, position, fmt).iloc[0][column] == pytest.approx(
        15.355 + 4.67 * rec_weight
    )


def deebo_raw():
    # NFL-Data/2025/3/projected/WR_projected.csv, fetched 2026-09-10.
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 3,
                "position": "WR",
                "PlayerId": "2562721",
                "PlayerName": "Deebo Samuel Sr.",
                "Team": "WAS",
                "PlayerOpponent": "LV",
                "ReceivingRec": 4.67,
                "ReceivingYDS": 65.44,
                "ReceivingTD": 0.34,
                "RushingYDS": 13.71,
                "RushingTD": 0.18,
                "Fum": 0.08,
                "PlayerWeekProjectedPts": 13.21,
            }
        ]
    )


def test_deebo_week3_ingestion_and_scoring_preserve_2451_rushing_points():
    frame = _normalize_one_position(deebo_raw(), "WR").assign(player_id="00-0035719")
    assert frame.iloc[0].rushing_yards == 13.71
    assert frame.iloc[0].rushing_tds == 0.18
    assert frame.iloc[0].fumbles_lost == 0.04
    assert project_nflcom_to_fantasy(frame, "WR").iloc[0].nflcom_pred_total == pytest.approx(15.625)
    assert _project_nflcom_to_ppr(frame, "WR", "ppr").iloc[0].nflcom_pred_total == pytest.approx(
        15.625
    )


def test_nflcom_truncated_raw_and_joined_caches_are_not_reused(tmp_path, monkeypatch):
    stale = pd.DataFrame({"player_id": ["stale"]})
    stale.to_parquet(tmp_path / "nflcom_projections_v1_2025_2025_w3-3.parquet")
    stale.to_parquet(tmp_path / "nflcom_projections_joined_v1_2025_2025_mr90_w3-3.parquet")
    roster = pd.DataFrame(
        [
            {
                "player_id": "00-0035719",
                "player_name": "Deebo Samuel Sr.",
                "position": "WR",
                "team": "WAS",
                "season": 2025,
            }
        ]
    )
    monkeypatch.setattr(nflcom_loader.nfl_source, "rosters", lambda seasons: roster)

    def reader(url):
        if url.endswith("/WR_projected.csv"):
            return deebo_raw()
        raise HTTPError(url, 404, "not found", None, None)

    frame = nflcom_loader.load_nflcom_with_gsis_id(
        [2025], weeks=[3], cache_dir=str(tmp_path), reader=reader
    )
    assert frame.iloc[0].player_id == "00-0035719"
    assert frame.iloc[0].rushing_yards == 13.71
    # The newly written joined cache must retain the restored stats on a warm read too.
    warm = nflcom_loader.load_nflcom_with_gsis_id(
        [2025],
        weeks=[3],
        cache_dir=str(tmp_path),
        reader=lambda url: pytest.fail("warm cache missed"),
    )
    assert project_nflcom_to_fantasy(warm, "WR").iloc[0].nflcom_pred_total == pytest.approx(15.625)


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_nflcom_retains_offense_stats_outside_model_heads(position):
    raw = deebo_raw().assign(position=position, PassingYDS=20, PassingTD=1, PassingInt=0.2)
    frame = _normalize_one_position(raw, position)
    assert frame.iloc[0].passing_yards == 20
    assert frame.iloc[0].rushing_yards == 13.71
    assert frame.iloc[0].receptions == 4.67


def test_sparse_and_missing_stats_keep_rows_and_negative_points():
    frame = pd.DataFrame(
        {"receptions": ["2", None], "rushing_yards": [np.nan, -10], "fumbles_lost": [0.5, 1]},
        index=[9, 4],
    )
    np.testing.assert_allclose(score_offensive_projections(frame, "ppr"), [1, -3])


def test_reference_ranks_full_forecasts_and_rejects_truncated_recipe():
    frame = pd.DataFrame(
        [
            {
                "player_id": "rush",
                "season": 2025,
                "week": 3,
                "position": "WR",
                "receiving_yards": 10,
                "rushing_yards": 100,
            },
            {
                "player_id": "receive",
                "season": 2025,
                "week": 3,
                "position": "WR",
                "receiving_yards": 60,
                "rushing_yards": 0,
            },
        ]
    ).assign(receiving_tds=0, receptions=0, fumbles_lost=0, nflcom_projected_pts=0)
    reference = build_reference(
        [2025], nflcom_loader=lambda seasons: frame, rotowire_loader=lambda seasons: frame
    )
    assert reference.iloc[0].player_id == "rush"
    assert reference.iloc[0].reference_pred == 11
    mask, meta = reference_selection("WR", frame, reference, 1)
    assert mask.tolist() == [True, False]
    assert meta["status"] == "available"
    assert REFERENCE_VERSION != "nflcom_rotowire_mean_v1"
    old = reference.assign(reference_version="nflcom_rotowire_mean_v1")
    mask, meta = reference_selection("WR", frame, old, 1)
    assert not mask.any()
    assert meta["status"] == "unavailable"
