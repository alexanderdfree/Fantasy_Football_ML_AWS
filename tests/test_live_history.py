"""Live-season ingestion, career context, and unknown practice imputation."""

import pandas as pd
import pytest

from src.serving import core
from src.serving import upcoming_week as live

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def missing_live_snaps(monkeypatch):
    def unavailable(seasons):
        raise ConnectionError("snap_counts_2026.parquet: 404")

    monkeypatch.setattr(live.nfl_source, "snap_counts", unavailable)


def test_current_season_is_fresh_and_team_tokens_advance(monkeypatch, tmp_path):
    old = pd.DataFrame(
        {"player_id": ["old"], "season": [2025], "week": [1], "recent_team": ["SEA"]}
    )
    current = pd.DataFrame(
        {
            "player_id": ["A", "B", "future"],
            "season": [2026] * 3,
            "week": [1, 1, 2],
            "recent_team": ["SEA", "NE", "SEA"],
        }
    )
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [13],
            "away_score": [10],
        }
    )
    calls = []

    def load(seasons, cache_dir=None):
        calls.append((seasons, cache_dir))
        return old.copy() if len(seasons) > 1 else current.copy()

    def teams(seasons, cache_dir=None):
        frame = old if len(seasons) > 1 else current
        return frame.rename(columns={"recent_team": "team"}).drop(columns="player_id")

    monkeypatch.setattr(live, "load_raw_data", load)
    monkeypatch.setattr(live, "load_team_week_stats", teams)
    monkeypatch.setattr(live, "preprocess", lambda df: df)
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "_history_cache", None)
    monkeypatch.setattr(live, "_history_seasons", None)
    one = live._load_history(2026, 1, schedule)
    two = live._load_history(2026, 1, schedule)
    assert set(one.player_id) == {"old", "A", "B"}
    pd.testing.assert_frame_equal(one, two)
    assert calls[0][0][-1] == 2025
    assert calls[1][0] == calls[2][0] == [2026]
    assert calls[1][1] != calls[2][1]  # no stale current-season file reuse
    assert one.attrs["live_history_sources"]["snap_counts"] == "unavailable"
    assert one.attrs["live_history_sources"]["player_rows"] == 2
    assert one.attrs["live_history_sources"]["ff_opportunity"] == "unavailable"
    assert one.attrs["live_history_sources"]["qbr_observed_rows"] == 0
    team_cache = pd.read_parquet(tmp_path / "team_stats_2012_2025.parquet")
    assert set(team_cache.loc[team_cache.season.eq(2026), "team"]) == {"SEA", "NE"}
    assert team_cache.week.max() == 1


def test_current_season_missing_completed_team_fails(monkeypatch, tmp_path):
    old = pd.DataFrame(
        {"player_id": ["old"], "season": [2025], "week": [1], "recent_team": ["SEA"]}
    )
    monkeypatch.setattr(live, "_history_cache", old)
    monkeypatch.setattr(live, "_history_seasons", tuple(range(live.SEASONS[0], 2026)))
    monkeypatch.setattr(live, "preprocess", lambda df: df)
    monkeypatch.setattr(live, "load_raw_data", lambda *a, **k: old.iloc[:0])
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [13],
            "away_score": [10],
        }
    )
    with pytest.raises(RuntimeError, match="not available yet"):
        live._load_history(2026, 1, schedule)


def test_inference_keeps_heldout_year_without_repeating_fit_years(monkeypatch):
    frame = pd.DataFrame(
        {
            "player_id": ["A"] * 3,
            "position": ["QB"] * 3,
            "recent_team": ["TEN"] * 3,
            "season": [2024, 2025, 2026],
            "week": [1, 1, 1],
            "_is_upcoming": [False, False, True],
        }
    )
    train = pd.DataFrame({"season": [2023]})
    val = pd.DataFrame({"season": [2024]})
    monkeypatch.setattr(live.core, "_ensure_base_data", lambda: None)
    monkeypatch.setattr(live.app_pkg, "_cache", {"splits": {"QB": (train, val, frame)}})
    seen = []
    monkeypatch.setattr(
        live.core, "_apply_position_models", lambda tr, va, te, p, r: seen.extend(te.season)
    )
    roster = pd.DataFrame({"player_id": ["A"], "espn_name": ["A"], "espn_id": ["1"]})
    slate = pd.DataFrame({"recent_team": ["TEN"], "spread_line": [1.0], "total_line": [40.0]})
    out = live.run_upcoming_inference(frame, roster, slate, 2026, 1)
    assert seen == [2025, 2026]
    assert out.season.tolist() == [2026]


def test_unknown_practice_uses_filtered_training_mean(monkeypatch):
    reg = {
        "targets": [],
        "model_dir": "unused",
        "filter_fn": lambda df: df.copy(),
        "compute_targets_fn": lambda df: df,
        "min_games_per_season": 2,
        "get_feature_columns_fn": lambda: ["practice_status"],
    }
    monkeypatch.setattr(core, "POSITION_REGISTRY", {"QB": reg})
    train = pd.DataFrame(
        {
            "player_id": ["A", "A", "B"],
            "season": [2023] * 3,
            "week": [1, 2, 1],
            "practice_status": [2.0, 1.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "player_id": ["X", "Y"],
            "season": [2026, 2026],
            "week": [1, 1],
            "practice_status": [float("nan"), 1.0],
            "_practice_status_missing": [True, False],
        }
    )

    class StopAfterPreparation(Exception):
        pass

    def capture(tr, va, te, *args, **kwargs):
        assert te.practice_status.tolist() == [1.5, 1.0]
        raise StopAfterPreparation

    monkeypatch.setattr(core, "build_position_features", capture)
    with pytest.raises(StopAfterPreparation):
        core._apply_position_models(train, train.iloc[:0], test, "QB", pd.DataFrame())
