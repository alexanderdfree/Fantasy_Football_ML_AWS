"""Played zero-stat games must survive loading, preprocessing, and history."""

import numpy as np
import pandas as pd
import pytest

from src.data.participation import restore_offensive_appearances

pytestmark = pytest.mark.unit


def _inputs():
    weekly = pd.DataFrame(
        {
            "player_id": ["TE", "QB", "TE"],
            "season": [2023] * 3,
            "week": [1, 2, 3],
            "season_type": ["REG"] * 3,
            "position": ["TE", "QB", "TE"],
            "recent_team": ["LV"] * 3,
            "opponent_team": ["KC"] * 3,
            "player_name": ["T.End", "Q.Back", "T.End"],
            "receptions": [3.0, 0.0, 2.0],
            "targets": [4.0, 0.0, 3.0],
            "passing_yards": [0.0, 100.0, 0.0],
            "completions": [0.0, 10.0, 0.0],
            "pacr": [np.nan, 1.1, np.nan],
        }
    )
    snaps = pd.DataFrame(
        {
            "gsis_id": ["TE"] * 3,
            "season": [2023] * 3,
            "week": [1, 2, 3],
            "game_type": ["REG"] * 3,
            "team": ["OAK"] * 3,
            "opponent": ["KC"] * 3,
            "offense_snaps": [60, 66, 62],
            "offense_pct": [0.9, 1.0, 0.95],
            "position": ["TE"] * 3,
            "player": ["Tight End"] * 3,
        }
    )
    rosters = pd.DataFrame(
        {"player_id": ["TE", "QB"], "season": [2023, 2023], "position": ["TE", "QB"]}
    )
    return weekly, snaps, rosters


def test_restore_played_game_preserves_existing_stats_and_missing_rates():
    weekly, snaps, rosters = _inputs()
    original = weekly.copy(deep=True)
    out = restore_offensive_appearances(weekly, pd.concat([snaps, snaps]), rosters)
    pd.testing.assert_frame_equal(weekly, original)
    pd.testing.assert_frame_equal(out.iloc[:3].reset_index(drop=True), original)
    added = out.iloc[3]
    assert len(out) == 4
    assert (added.player_id, added.week, added.recent_team, added.opponent_team) == (
        "TE",
        2,
        "LV",
        "KC",
    )
    assert added.player_name == "Tight End"
    assert added.receptions == added.targets == added.completions == 0
    assert pd.isna(added.pacr)
    assert not out.duplicated(["player_id", "season", "week"]).any()
    pd.testing.assert_frame_equal(restore_offensive_appearances(out, snaps, rosters), out)


@pytest.mark.parametrize(
    "case",
    [
        "no_offense",
        "unknown_id",
        "unmodeled_position",
        "postseason",
        "missing_game",
        "missing_season",
    ],
)
def test_restoration_needs_positive_participation_and_covered_game(case):
    weekly, snaps, rosters = _inputs()
    snaps = snaps.loc[snaps.week.eq(2)].copy()
    if case == "no_offense":
        snaps["offense_snaps"] = 0
    elif case == "unknown_id":
        snaps["gsis_id"] = None
    elif case == "unmodeled_position":
        rosters.loc[rosters.player_id.eq("TE"), "position"] = "FB"
    elif case == "postseason":
        snaps["game_type"] = "POST"
    elif case == "missing_game":
        snaps["week"] = 4
    else:
        snaps["season"] = 2024
        rosters["season"] = 2024
    pd.testing.assert_frame_equal(restore_offensive_appearances(weekly, snaps, rosters), weekly)


def test_loader_restores_game_before_preprocessing_and_history(tmp_path, monkeypatch):
    import src.data.loader as loader
    from src.data.preprocessing import preprocess
    from src.features.engineer import _build_contextual_features, build_game_history_arrays
    from tests.test_data_loader import _mock_all_nfl_helpers

    _mock_all_nfl_helpers(monkeypatch)
    weekly, snaps, rosters = _inputs()
    snaps["pfr_player_id"] = "pfrTE"
    monkeypatch.setattr(loader.nfl_source, "weekly_data", lambda seasons: weekly.copy())
    monkeypatch.setattr(loader.nfl_source, "rosters", lambda seasons: rosters.copy())
    monkeypatch.setattr(
        loader.nfl_source, "snap_counts", lambda seasons: snaps.drop(columns="gsis_id")
    )
    monkeypatch.setattr(
        loader.nfl_source,
        "player_ids",
        lambda: pd.DataFrame({"pfr_id": ["pfrTE"], "gsis_id": ["TE"]}),
    )
    raw = loader.load_raw_data([2023], cache_dir=str(tmp_path))
    clean = _build_contextual_features(preprocess(raw))
    te = clean.loc[clean.position.eq("TE")].sort_values("week").reset_index(drop=True)
    assert te.week.tolist() == [1, 2, 3]
    assert te.loc[1, "snap_pct"] == 1.0
    assert te.loc[1, "fantasy_points"] == 0
    assert te.loc[2, "is_returning_from_absence"] == 0
    assert te.loc[2, "days_rest"] == 7
    history, mask = build_game_history_arrays(te, history_stats=["receptions"])
    assert mask[2, :2].all()
    assert history[2, :2, 0].tolist() == [0, 3]
