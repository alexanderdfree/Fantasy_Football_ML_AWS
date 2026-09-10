"""Reject mislabeled future seasons and invalidate the abandoned QBR source."""

import pandas as pd
import pytest

from src.data import external_sources as ext
from src.data import nfl_source as source

pytestmark = pytest.mark.unit


def frames():
    qbr = pd.DataFrame(
        {
            "season": [2025, 2026, 2026, 2026],
            "game_week": [1, 1, 1, 2],
            "game_id": [101, 101, 201, 202],
            "season_type": ["Regular"] * 4,
            "qbr_total": [75.0, 75.0, 60.0, 88.0],
        }
    )
    schedules = pd.DataFrame(
        {
            "season": [2025, 2026, 2026],
            "week": [1, 1, 2],
            "espn": [101, 201, 202],
            "game_type": ["REG"] * 3,
            "home_score": [24, 20, None],
            "away_score": [17, 10, None],
            "gameday": ["2025-09-01", "2026-09-01", "2099-09-01"],
        }
    )
    return qbr, schedules


def test_rejects_prior_season_relabel_and_unplayed_rows():
    qbr, schedules = frames()
    out = source._validated_qbr_games(qbr, schedules)
    assert list(zip(out.season, out.game_id, strict=True)) == [(2025, 101), (2026, 201)]


def test_rejects_wrong_week_even_for_valid_game_id():
    qbr, schedules = frames()
    qbr.loc[0, "game_week"] = 2
    assert 101 not in source._validated_qbr_games(qbr, schedules).game_id.tolist()


def test_reads_maintained_release_and_exact_requested_seasons(monkeypatch):
    qbr, schedules = frames()
    urls = []

    def read(url):
        urls.append(url)
        return qbr

    monkeypatch.setattr(source.pd, "read_parquet", read)
    monkeypatch.setattr(source, "schedules", lambda seasons: schedules)
    out = source.qbr_weekly([2025])
    assert len(out) == 1
    assert "/espn_data/qbr_week_level.parquet" in urls[0]


def test_old_schema_valid_cache_cannot_hide_new_years(tmp_path, monkeypatch):
    cols = ["player_id", "season", "week", *ext.QBR_FEATURE_COLUMNS]
    pd.DataFrame(columns=cols).to_parquet(tmp_path / "qbr_weekly_2025_2025.parquet")
    raw = pd.DataFrame(
        {
            "season": [2025],
            "season_type": ["Regular"],
            "game_week": [1],
            "player_id": [7],
            "qbr_total": [70.0],
            "pts_added": [5.0],
        }
    )
    monkeypatch.setattr(source, "qbr_weekly", lambda seasons: raw)
    monkeypatch.setattr(
        source, "player_ids", lambda: pd.DataFrame({"espn_id": [7], "gsis_id": ["A"]})
    )
    out = ext.load_qbr_weekly([2025], cache_dir=str(tmp_path))
    assert out.qbr_total.tolist() == [70.0]
    assert (tmp_path / "qbr_weekly_v2_2025_2025.parquet").exists()
