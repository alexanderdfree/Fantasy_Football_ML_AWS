"""Native K/DST frames must reach the shared offline-diagnostic loaders."""

import pandas as pd
import pytest

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.shared import position_data

pytestmark = pytest.mark.unit


def _frames(position, value=1.0):
    return tuple(
        pd.DataFrame(
            {
                "player_id": [position],
                "position": [position],
                "season": [season],
                "week": [1],
                "fantasy_points": [value],
                "native_value": [value],
            }
        )
        for season in (TRAIN_SEASONS[-1], VAL_SEASONS[0], TEST_SEASONS[0])
    )


@pytest.mark.parametrize("position", ["K", "DST"])
def test_native_loader_derives_real_targets_before_splitting(monkeypatch, position):
    import importlib

    data = importlib.import_module(f"src.{position.lower()}.data")
    features = importlib.import_module(f"src.{position.lower()}.features")
    raw = pd.concat(_frames(position), ignore_index=True)
    games = getattr(data, "MIN_GAMES", 1)
    raw = pd.concat([raw.assign(week=week) for week in range(1, games + 1)], ignore_index=True)
    if position == "K":
        raw = raw.assign(fg_yards_made=70.0, pat_made=2.0, fg_missed=1.0, pat_missed=0.0)
        expected_points = 8.0
    else:
        raw = raw.assign(
            points_allowed=21.0,
            yards_allowed=350.0,
            def_sacks=2.0,
            def_ints=1.0,
            def_fumble_rec=0.0,
            def_fumbles_forced=0.0,
            def_safeties=0.0,
            def_tds=0.0,
            def_blocked_kicks=0.0,
            special_teams_tds=0.0,
        )
        expected_points = 3.0
    monkeypatch.setattr(data, "load_data" if position == "K" else "build_data", lambda: raw)

    def compute_features(frame):
        assert len(frame) == 3 * games  # features see full history before splitting
        assert frame["fantasy_points"].eq(expected_points).all()
        frame["feature_ready"] = True

    monkeypatch.setattr(features, "compute_features", compute_features)
    frames = position_data.load_position_frames(position)
    for frame, season in zip(
        frames, (TRAIN_SEASONS[-1], VAL_SEASONS[0], TEST_SEASONS[0]), strict=True
    ):
        assert frame["season"].tolist() == [season] * games
        assert frame["feature_ready"].all()
        assert frame["fantasy_points"].eq(expected_points).all()


def test_native_ablation_preserves_full_kick_history_and_config(monkeypatch):
    from src.k import data as kicker_data
    from src.shared import registry

    full = pd.concat(_frames("K"), ignore_index=True)
    splits = _frames("K")
    kicks = pd.DataFrame({"kick_distance": [30.0, 40.0]})
    cfg = {"attn_kick_stats": ["kick_distance"], "attn_max_games": 2, "attn_max_kicks_per_game": 3}
    monkeypatch.setattr(position_data, "_native_frame", lambda pos: full)
    monkeypatch.setattr(position_data, "_native_splits", lambda pos, frame: splits)
    monkeypatch.setattr(registry, "get_config", lambda pos: cfg)

    def load_kicks(frame):
        assert frame is full  # before the K training-row filter, not concatenated splits
        return kicks

    monkeypatch.setattr(kicker_data, "load_kicks", load_kicks)
    actual_splits, actual_cfg = position_data.prepare_native_ablation("K")
    builder = actual_cfg["attn_history_builder_fn"]
    assert actual_splits is splits
    assert builder.keywords["kicks_df"] is kicks
    assert builder.keywords["max_games"] == 2
    assert builder.keywords["max_kicks_per_game"] == 3
    assert "attn_history_builder_fn" not in cfg
