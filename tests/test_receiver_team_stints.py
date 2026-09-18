"""Team-relative WR/TE shares restart at each stint; personal volume continues."""

import importlib

import numpy as np
import pandas as pd
import pytest

from tests.wr.conftest import make_wr_player_games

pytestmark = pytest.mark.unit
SHARES = ["opportunity_index_L3", "redzone_target_share_L3"]


@pytest.fixture(params=["wr", "te"])
def feature_module(request):
    return importlib.import_module(f"src.{request.param}.features")


def _games(*, same_team=False):
    player = make_wr_player_games(n_weeks=8, player_id="receiver", recent_team="OLD", season=2024)
    if not same_team:
        player["recent_team"] = ["OLD"] * 3 + ["NEW"] * 2 + ["OLD"] * 3
    player["targets"] = [9, 9, 9, 1, 1, 2, 2, 2]
    player["receiving_tds"] = 0
    player["redzone_target_share"] = [0.8, 0.6, 0.4, 0.02, 0.04, 0.4, 0.2, 0.1]
    player["redzone_targets"] = np.arange(1, 9, dtype=float)
    teammate = player.copy()
    teammate["player_id"] = "teammate"
    teammate["targets"] = [1, 1, 1, 99, 99, 8, 8, 8]
    frame = pd.concat([player, teammate], ignore_index=True)
    frame.index = pd.Index(np.arange(len(frame)) * 11 - 45, name="source_row")
    return frame


def _player(frame):
    return frame[frame["player_id"].eq("receiver")].sort_values(["season", "week"])


def test_share_windows_restart_on_each_team_stint(feature_module):
    frame = _games()
    feature_module._compute_features(frame)
    player = _player(frame)
    np.testing.assert_allclose(
        player["opportunity_index_L3"],
        [np.nan, 0.9, 0.9, np.nan, 0.01, np.nan, 0.2, 0.2],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        player["redzone_target_share_L3"],
        [np.nan, 0.8, 0.7, np.nan, 0.02, np.nan, 0.4, 0.3],
        equal_nan=True,
    )
    # Raw personal volume remains useful after a trade and uses prior games only.
    np.testing.assert_allclose(
        player["redzone_targets_L3"], [np.nan, 1, 1.5, 2, 3, 4, 5, 6], equal_nan=True
    )
    # Per-game attention inputs keep the observed current-team fraction.
    np.testing.assert_allclose(
        player["game_opportunity_index"], [0.9, 0.9, 0.9, 0.01, 0.01, 0.2, 0.2, 0.2]
    )
    assert set(SHARES) <= set(feature_module.get_feature_columns())


def test_same_team_share_windows_retain_existing_values(feature_module):
    frame = _games(same_team=True)
    feature_module._compute_features(frame)
    player = _player(frame)
    np.testing.assert_allclose(
        player["opportunity_index_L3"],
        [np.nan, 0.9, 0.9, 0.9, 1.81 / 3, 0.92 / 3, 0.22 / 3, 0.41 / 3],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        player["redzone_target_share_L3"],
        [np.nan, 0.8, 0.7, 0.6, 1.02 / 3, 0.46 / 3, 0.46 / 3, 0.64 / 3],
        equal_nan=True,
    )


def test_trade_share_values_follow_rows_after_permutation(feature_module):
    ordered = _games()
    shuffled = ordered.sample(frac=1, random_state=31)
    expected_index = shuffled.sort_values(["player_id", "season", "week"]).index
    feature_module._compute_features(ordered)
    feature_module._compute_features(shuffled)
    pd.testing.assert_frame_equal(ordered, shuffled)
    pd.testing.assert_index_equal(shuffled.index, expected_index)
    # A first appearance for NEW is unknown even with arbitrary source labels.
    new_stint = shuffled[shuffled["player_id"].eq("receiver") & shuffled["week"].eq(4)]
    assert new_stint[SHARES].isna().all().all()


def test_share_windows_reset_again_at_season_boundary(feature_module):
    frame = _games()
    next_year = frame[frame["week"].le(2)].copy()
    next_year["season"] += 1
    next_year.index += 1000
    frame = pd.concat([frame, next_year]).sample(frac=1, random_state=19)
    feature_module._compute_features(frame)
    opener = _player(frame).groupby("season").head(1)
    assert opener[SHARES].isna().all().all()
    later = _player(frame).query("season == 2025 and week == 2")
    assert later["opportunity_index_L3"].iloc[0] == pytest.approx(0.9)
    assert later["redzone_target_share_L3"].iloc[0] == pytest.approx(0.8)


def test_current_game_changes_do_not_rewrite_prior_share_windows(feature_module):
    baseline = _games()
    changed = baseline.copy()
    last_game = changed["player_id"].eq("receiver") & changed["week"].eq(8)
    changed.loc[last_game, ["targets", "redzone_target_share"]] = [999, 0.99]
    feature_module._compute_features(baseline)
    feature_module._compute_features(changed)
    pd.testing.assert_frame_equal(baseline[SHARES], changed[SHARES])
