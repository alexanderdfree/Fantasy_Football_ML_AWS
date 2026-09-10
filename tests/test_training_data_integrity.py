"""Real failure shapes from the training-data completeness/semantics audit."""

import numpy as np
import pandas as pd
import pytest

from src.data.identity import bridge_snap_counts, valid_player_ids
from src.data.loader import _normalize_espn_depth

pytestmark = pytest.mark.unit


def test_complete_data_builder_cli_imports_without_fetching():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "src.data.build", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "unpinned producer" in result.stdout


def test_bridge_uses_unique_roster_id_then_team_season_name_without_fanout():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["morgan", "blount", "same", "none"],
            "player": ["David Morgan", "LeGarrette Blount", "Chris Williams", "Unknown"],
            "season": [2016] * 4,
            "team": ["MIN", "NE", "LA", "LA"],
        }
    )
    roster = pd.DataFrame(
        {
            "pfr_id": ["morgan", None, None, None],
            "player_id": ["g1", "g2", "g3", "g4"],
            "full_name": [
                "David Morgan II",
                "LeGarrette Blount",
                "Chris Williams",
                "Chris Williams",
            ],
            "season": [2016] * 4,
            "team": ["MIN", "NE", "LA", "LA"],
        }
    )
    result = bridge_snap_counts(snaps, pd.DataFrame(), roster, pd.DataFrame())
    assert len(result) == len(snaps)
    assert result.gsis_id.iloc[:2].tolist() == ["g1", "g2"]
    assert result.gsis_id.iloc[2:].isna().all()  # ambiguous name remains unresolved


def test_bridge_never_overrides_primary_or_matches_a_name_across_team_or_season():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["known", "missing", "other"],
            "player": ["Player", "Player", "Player"],
            "season": [2024, 2023, 2024],
            "team": ["KC", "KC", "BUF"],
        }
    )
    ids = pd.DataFrame({"pfr_id": ["known"], "gsis_id": ["authoritative"]})
    roster = pd.DataFrame(
        {"player_id": ["other_id"], "full_name": ["Player"], "season": [2024], "team": ["KC"]}
    )
    result = bridge_snap_counts(snaps, ids, roster, pd.DataFrame())
    assert result.gsis_id.iloc[0] == "authoritative"
    assert result.gsis_id.iloc[1:].isna().all()


def test_partial_metadata_cannot_resolve_an_ambiguous_roster_name():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["missing"],
            "player": ["Chris Williams"],
            "season": [2025],
            "team": ["KC"],
        }
    )
    roster = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "full_name": ["Chris Williams"] * 2,
            "season": [2025] * 2,
            "team": ["KC"] * 2,
        }
    )
    metadata = pd.DataFrame(
        {
            "gsis_id": ["a"],
            "display_name": ["Chris Williams"],
            "last_name": ["Williams"],
        }
    )
    result = bridge_snap_counts(snaps, pd.DataFrame(), roster, pd.DataFrame(), metadata)
    assert result.gsis_id.isna().all()


def test_missing_identity_strings_are_not_join_keys():
    values = pd.Series([None, np.nan, pd.NA, "", " None ", "nan", "<NA>", "00-0032430"])
    assert valid_player_ids(values).tolist() == [False] * 7 + [True]


def test_documented_nfl_aliases_are_scoped_to_rostered_identity():
    snaps = pd.DataFrame(
        {
            "pfr_player_id": ["missing1", "missing2", "missing3"],
            "player": ["Rodney Williams", "John Samuel Shenker", "Nathan Carter"],
            "season": [2025] * 3,
            "team": ["PIT", "LV", "ATL"],
        }
    )
    roster = pd.DataFrame(
        {
            "player_id": ["rod", "john", "00-0040547"],
            "season": [2025] * 3,
            "team": ["PIT", "LV", "ATL"],
            "full_name": ["Rod Williams", "John Shenker", "Nate Carter"],
        }
    )
    metadata = pd.DataFrame(
        {
            "gsis_id": ["rod", "john", "00-0040547"],
            "first_name": ["Rodney", "John", "Nate"],
            "last_name": ["Williams", "Shenker", "Carter"],
            "football_name": ["Rod", "John Samuel", "Nate"],
        }
    )
    result = bridge_snap_counts(snaps, pd.DataFrame(), roster, pd.DataFrame(), metadata)
    assert result.gsis_id.tolist() == ["rod", "john", "00-0040547"]


def test_pinned_release_rejects_source_refetch_and_optional_source_fallback(tmp_path, monkeypatch):
    from src.data.external_sources import load_ff_opportunity
    from src.data.loader import load_team_week_stats
    from src.data.release import DataReleaseError

    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    with pytest.raises(DataReleaseError):
        load_team_week_stats([2025], cache_dir=str(tmp_path))
    with pytest.raises(DataReleaseError):
        load_ff_opportunity([2025], cache_dir=str(tmp_path))


def test_kicker_backfill_is_reused_offline_within_a_pinned_release(tmp_path, monkeypatch):
    from src.data.release import DataReleaseError
    from src.k import data

    frame = pd.DataFrame(
        {
            "season": [2025],
            "season_type": ["REG"],
            "week": [1],
            "posteam": ["KC"],
            "kicker_player_id": ["k"],
            "field_goal_attempt": [1],
            "field_goal_result": ["made"],
            "kick_distance": [42],
            "fg_prob": [0.8],
            "qtr": [1],
            "wind": [0],
            "temp": [65],
            "roof": ["dome"],
            "surface": ["grass"],
        }
    )
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    monkeypatch.setattr(data.nfl_source, "pbp_data", lambda *args: frame)
    first = data._load_backfill_pbp(2025)
    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    monkeypatch.setattr(
        data.nfl_source, "pbp_data", lambda *args: pytest.fail("unexpected source fetch")
    )
    pd.testing.assert_frame_equal(first, data._load_backfill_pbp(2025))
    with pytest.raises(DataReleaseError):
        data._load_backfill_pbp(2024)


def test_depth_preserves_three_receiver_starters_and_unknown_starter_slot():
    espn = pd.DataFrame(
        {
            "dt": ["2025-09-01T00:00:00Z"] * 6,
            "team": ["CIN"] * 6,
            "pos_grp": ["3WR 1TE"] * 6,
            "gsis_id": [None, "higgins", "iosivas", "backup", "burrow", "qb2"],
            "pos_abb": ["WR"] * 4 + ["QB"] * 2,
            "pos_slot": [1, 2, 8, 1, 9, 9],
            "pos_rank": [1, 2, 3, 4, 1, 2],
        }
    )
    schedules = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2025-09-07"],
            "home_team": ["CIN"],
            "away_team": ["CLE"],
        }
    )
    result = _normalize_espn_depth(espn, schedules, 2025).set_index("gsis_id")
    assert result.depth_team.to_dict() == {
        "higgins": "1",
        "iosivas": "1",
        "backup": "2",
        "burrow": "1",
        "qb2": "2",
    }


def test_kicker_shared_filter_matches_its_declared_split_threshold():
    from src.k.config import POSITION_CONFIG
    from src.k.data import MIN_GAMES
    from src.shared.registry import get_config

    assert MIN_GAMES == POSITION_CONFIG.min_games == get_config("K")["min_games_per_season"] == 4
