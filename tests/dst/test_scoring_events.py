"""D/ST event-source regressions, cache integrity and complete-game coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import dst_scoring
from src.dst import data as dst_data
from tests.dst.test_data_build import (
    _make_schedules,
    _make_scoring_events,
    _make_team_stats,
    _make_weekly,
)

pytestmark = pytest.mark.unit


def _plays(*overrides, season=2024):
    base = dict.fromkeys(dst_scoring.PBP_COLUMNS, 0)
    base.update(
        season=season,
        season_type="REG",
        week=1,
        game_id=f"{season}_01_BUF_KC",
        home_team="KC",
        away_team="BUF",
        posteam="BUF",
        defteam="KC",
        td_team=None,
        play_type="run",
    )
    return pd.DataFrame([{**base, "play_id": i, **update} for i, update in enumerate(overrides)])


def test_complete_td_categories_exclude_offensive_fumbles_and_do_not_double_count_returns():
    plays = _plays(
        # A pick-six and a scoop-and-score are both defensive touchdowns.
        {"touchdown": 1, "td_team": "KC"},
        {"touchdown": 1, "td_team": "KC"},
        # An offensive recovery in the end zone (e.g. WAS, 2012 W4) is not D/ST.
        {"touchdown": 1, "td_team": "BUF"},
        {"touchdown": 1, "td_team": "KC", "punt_attempt": 1},
        # Kickoff receiving team can be posteam; same-team comparison alone fails.
        {"touchdown": 1, "td_team": "BUF", "kickoff_attempt": 1},
        # Kicking-team recovery scores count as special teams too.
        {"touchdown": 1, "td_team": "KC", "kickoff_attempt": 1},
        {"touchdown": 1, "td_team": "KC", "field_goal_attempt": 1},
        {"punt_attempt": 1, "punt_blocked": 1},
        {"touchdown": 1, "td_team": "KC", "two_point_attempt": 1},
        {"touchdown": 1, "td_team": "KC", "extra_point_attempt": 1},
        {"touchdown": 1, "td_team": "KC", "play_type": "no_play", "punt_blocked": 1},
        {"touchdown": 1, "td_team": "KC", "season_type": "POST"},
    )
    # Repeated provider rows must not count an event twice.
    plays = pd.concat([plays, plays.iloc[[0]]], ignore_index=True)
    actual = dst_scoring.aggregate_dst_scoring_events(plays).set_index("team")
    assert actual.loc["KC", list(dst_scoring.SCORING_COLUMNS)].tolist() == [2, 3, 1]
    assert actual.loc["BUF", list(dst_scoring.SCORING_COLUMNS)].tolist() == [0, 1, 0]


def test_zero_event_games_are_present_and_historical_team_codes_are_normalized():
    plays = _plays({"home_team": "OAK", "defteam": "OAK"})
    actual = dst_scoring.aggregate_dst_scoring_events(plays)
    assert set(actual.team) == {"BUF", "LV"}
    assert actual[list(dst_scoring.SCORING_COLUMNS)].eq(0).all().all()


def test_missing_source_column_is_not_treated_as_zero():
    with pytest.raises(ValueError, match="punt_blocked"):
        dst_scoring.aggregate_dst_scoring_events(_plays({}).drop(columns="punt_blocked"))


def test_cache_reuses_complete_data_and_preserves_noncontiguous_seasons(tmp_path, monkeypatch):
    calls = []

    def fetch(seasons, columns):
        calls.append(seasons)
        assert columns == dst_scoring.PBP_COLUMNS
        return _plays({"punt_blocked": 1}, season=seasons[0])

    monkeypatch.setattr(dst_scoring.nfl_source, "pbp_data", fetch)
    expected = dst_scoring.load_dst_scoring_events([2022, 2024], tmp_path)
    assert sorted(calls) == [[2022], [2024]]
    assert set(expected.season) == {2022, 2024}
    monkeypatch.setattr(
        dst_scoring.nfl_source, "pbp_data", lambda *a: pytest.fail("cache hit fetched PBP")
    )
    actual = dst_scoring.load_dst_scoring_events([2024, 2022], tmp_path, allow_fetch=False)
    pd.testing.assert_frame_equal(expected, actual)
    with pytest.raises(FileNotFoundError, match="training data release"):
        dst_scoring.load_dst_scoring_events([2022, 2023, 2024], tmp_path, allow_fetch=False)


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_season_never_publishes_partial_cache(tmp_path, monkeypatch, failure):
    def fetch(seasons, _):
        if seasons == [2024]:
            if failure == "exception":
                raise RuntimeError("source unavailable")
            return _plays({}).iloc[:0]
        return _plays({}, season=2023)

    monkeypatch.setattr(dst_scoring.nfl_source, "pbp_data", fetch)
    with pytest.raises((ValueError, RuntimeError)):
        dst_scoring.load_dst_scoring_events([2023, 2024], tmp_path)
    assert not list(tmp_path.glob("*.parquet"))


def test_stale_schema_is_rebuilt_and_cache_only_mode_cannot_fetch(tmp_path, monkeypatch):
    path = tmp_path / "dst_scoring_pbp_v1_2024_2024.parquet"
    pd.DataFrame({"season": [2024]}).to_parquet(path)
    monkeypatch.setattr(dst_scoring.nfl_source, "pbp_data", lambda *_: _plays({}))
    with pytest.raises(FileNotFoundError):
        dst_scoring.load_dst_scoring_events([2024], tmp_path, allow_fetch=False)
    actual = dst_scoring.load_dst_scoring_events([2024], tmp_path)
    assert len(actual) == 2


def test_build_data_uses_complete_td_sources_and_adds_punt_blocks_without_changing_other_targets(
    monkeypatch,
):
    monkeypatch.setattr(dst_data.nfl_source, "teams", lambda: pd.DataFrame())
    weekly, teams, schedules = _make_weekly(), _make_team_stats(), _make_schedules()
    events = _make_scoring_events()
    weekly["special_teams_tds"] = 90  # obsolete source must not double-count
    teams["def_tds"] = 90
    teams["fg_blocked"], teams["pat_blocked"] = 2, 1
    events["def_tds"], events["special_teams_tds"], events["def_punt_blocks"] = 2, 1, 3
    actual = dst_data.build_data(
        weekly=weekly, team_stats=teams, schedules=schedules, scoring_events=events
    )
    assert actual.def_tds.eq(2).all()
    assert actual.special_teams_tds.eq(1).all()
    assert actual.def_blocked_kicks.eq(6).all()
    # Preserve the owner-excluded overall points-allowed scoring contract.
    assert actual.loc[actual.is_home.eq(1), "points_allowed"].eq(17).all()
    assert actual.loc[actual.is_home.eq(0), "points_allowed"].eq(24).all()
    expected_recoveries = teams.set_index(["team", "season", "week"]).fumble_recovery_opp
    pd.testing.assert_series_equal(
        actual.set_index(["team", "season", "week"]).def_fumble_rec.sort_index(),
        expected_recoveries.sort_index(),
        check_names=False,
    )


def test_injected_inputs_never_fetch_history_and_missing_completed_events_fail(monkeypatch):
    monkeypatch.setattr(dst_data, "load_dst_scoring_events", lambda *a, **kw: pytest.fail("fetch"))
    kwargs = dict(weekly=_make_weekly(), team_stats=_make_team_stats(), schedules=_make_schedules())
    with pytest.raises(ValueError, match="require matching scoring_events"):
        dst_data.build_data(**kwargs)
    events = _make_scoring_events().iloc[1:]
    with pytest.raises(ValueError, match="missing completed team-weeks"):
        dst_data.build_data(**kwargs, scoring_events=events)
