"""Pregame replay contracts using the actual K and DST data/feature builders."""

import numpy as np
import pandas as pd
import pytest

from src.dst import data as dst_data
from src.dst import features as dst_features
from src.dst import targets as dst_targets
from src.dst.config import POSITION_CONFIG as DST_CONFIG
from src.k.config import POSITION_CONFIG as K_CONFIG
from src.serving import espn_live
from src.serving import upcoming_special_teams as special
from tests.dst.test_data_build import _make_schedules, _make_team_stats, _make_weekly
from tests.k.conftest import _build_games

pytestmark = pytest.mark.unit


def schedule():
    frame = _make_schedules()
    frame["temp"] = 60.0
    frame["wind"] = 10.0
    frame["game_id"] = frame["week"].astype(str) + frame["home_team"]
    return frame


def test_dst_replay_matches_all_38_features_without_target_week_outcomes(monkeypatch):
    monkeypatch.setattr(dst_data.nfl_source, "teams", lambda: pd.DataFrame())
    weekly, teams, games = _make_weekly(), _make_team_stats(), schedule()
    expected = dst_targets.compute_targets(
        dst_data.build_data(weekly=weekly, schedules=games, team_stats=teams)
    )
    dst_features.compute_features(expected)
    replay = special.build_defense_frame(weekly, teams, games, 2024, 3)
    cols = list(DST_CONFIG.all_features)
    lhs = expected[expected.week.eq(3)].set_index("team")[cols].sort_index()
    rhs = replay[replay.week.eq(3)].set_index("team")[cols].sort_index()
    pd.testing.assert_frame_equal(lhs, rhs)
    assert replay.loc[replay.week.eq(3), DST_CONFIG.targets].isna().all().all()
    # Positive control: a changed PREVIOUS game's QB signal must reach week 3.
    changed = weekly.copy()
    changed.loc[changed.week.eq(2) & changed.position.eq("QB"), "passing_epa"] += 100
    new = special.build_defense_frame(changed, teams, games, 2024, 3)
    assert not new.loc[new.week.eq(3), "opp_qb_epa_L5"].equals(
        replay.loc[replay.week.eq(3), "opp_qb_epa_L5"]
    )
    # A target-week outcome edit must change neither features nor prior tokens.
    weekly.loc[weekly.week.eq(3), "passing_epa"] = 1e9
    teams.loc[teams.week.eq(3), "def_tds"] = 999
    games.loc[games.week.eq(3), "home_score"] = 999
    poisoned = special.build_defense_frame(weekly, teams, games, 2024, 3)
    pd.testing.assert_frame_equal(replay, poisoned)


def test_kicker_live_history_updates_rollups_and_preserves_cross_season_context():
    history = _build_games(player_id="K1", n_weeks=3, season=2023)
    history = history.assign(
        position="K", recent_team="BUF", fg_yards_made=70, fg_missed=1, pat_missed=0
    )
    current = history.iloc[:2].copy()
    current["season"] = 2024
    current["fg_att"] = [1.0, 5.0]
    games = pd.concat([schedule().assign(season=2023), schedule()], ignore_index=True)
    roster = pd.DataFrame([{"player_id": "K1", "position": "K", "recent_team": "BUF"}])
    replay = special.build_kicker_frame(history, current, roster, games, 2024, 3)
    row = replay[replay.season.eq(2024) & replay.week.eq(3)].iloc[0]
    assert set(K_CONFIG.all_features) <= set(replay.columns)
    assert row.fg_attempts_L3 == pytest.approx((3 + 1 + 5) / 3)
    assert row.implied_team_total == pytest.approx((44.5 - 3.5) / 2)
    assert row.game_wind == 10
    assert row.game_temp == 60
    assert row[K_CONFIG.targets].isna().all()
    assert replay[replay.season.eq(2024) & replay.week.eq(2)].iloc[0].team_points_scored == 24
    # Source correction replaces the live season's rows instead of duplicating
    # an old snapshot, and future rows cannot leak into the rollup.
    current.loc[current.week.eq(2), "fg_att"] = 11
    future = current.assign(season=2025, fg_att=9999)
    new = special.build_kicker_frame(
        pd.concat([history, current, future]), current, roster, games, 2024, 3
    )
    assert len(new) == len(replay)
    assert new[new.season.eq(2024) & new.week.eq(3)].iloc[0].fg_attempts_L3 == 5


def test_matchup_join_uses_canonical_id_and_fresh_context():
    games = schedule().iloc[:1].copy()
    espn = games.copy()
    espn["game_id"] = "espn-different-id"
    espn["spread_line"] = 8.5
    espn["home_score"] = np.nan
    result = special.merge_live_schedule(games, espn)
    assert result.iloc[0].game_id == games.iloc[0].game_id
    assert result.iloc[0].spread_line == 8.5
    assert result.iloc[0].home_rest == 7
    assert pd.isna(result.iloc[0].home_score)
    espn["home_team"] = "MISSING"
    with pytest.raises(RuntimeError, match="missing upcoming"):
        special.merge_live_schedule(games, espn)


def test_opener_does_not_require_unpublished_outcomes(monkeypatch):
    monkeypatch.setattr(special.nfl_source, "schedules", lambda _: schedule())
    monkeypatch.setattr(special.nfl_source, "weekly_data", lambda _: pytest.fail("no games needed"))
    result = special.fetch_live_inputs(2024, 1, _make_weekly(), _make_team_stats())
    assert result.weekly.empty and result.team_stats.empty and result.pbp.empty


def test_missing_completed_history_fails_instead_of_zero_filling(monkeypatch):
    games = schedule()
    monkeypatch.setattr(special.nfl_source, "schedules", lambda _: games)
    weekly = _make_weekly().assign(season_type="REG")
    teams = _make_team_stats().assign(season_type="REG")
    pbp = weekly.rename(columns={"recent_team": "posteam"})
    monkeypatch.setattr(
        special.nfl_source, "weekly_data", lambda _: weekly[weekly.recent_team.ne("BUF")]
    )
    monkeypatch.setattr(special.nfl_source, "team_week_stats_release", lambda _: teams)
    monkeypatch.setattr(special.nfl_source, "pbp_data", lambda *_: pbp)
    with pytest.raises(RuntimeError, match="weekly missing completed team-weeks"):
        special.fetch_live_inputs(2024, 3, weekly, teams)


def test_live_roster_includes_pk_but_not_punter_or_inactive():
    player = {"id": "10", "displayName": "Kicker", "position": {"abbreviation": "PK"}}
    data = {
        "athletes": [
            {
                "position": "specialTeam",
                "items": [player, {**player, "id": "11", "position": {"abbreviation": "P"}}],
            },
            {"position": "injuredReserveOrOut", "items": [{**player, "id": "12"}]},
        ]
    }
    assert espn_live._parse_roster_players(data, "BUF") == []
    assert espn_live._parse_roster_players(data, "BUF", include_kickers=True) == [
        {
            "espn_id": "10",
            "espn_name": "Kicker",
            "position": "K",
            "recent_team": "BUF",
        }
    ]


def test_prepare_reuses_verified_schedule_for_both_special_positions(monkeypatch, tmp_path):
    games = schedule()
    verified = games.assign(temp=52.0, wind=17.0, roof="outdoors", surface="grass")
    espn = games[games.week.eq(1)].assign(game_id="espn-id", venue=[{"id": "new"}] * 2)
    weekly, teams = _make_weekly(), _make_team_stats()
    weekly["season_type"] = "REG"
    suffix = f"{special.SEASONS[0]}_{special.SEASONS[-1]}"
    weekly.to_parquet(tmp_path / f"weekly_{suffix}.parquet")
    games.to_parquet(tmp_path / f"schedules_{suffix}.parquet")
    monkeypatch.setattr(special, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(special, "load_team_week_stats", lambda *args, **kwargs: teams)
    live = special.LiveInputs(games, weekly.iloc[:0], teams.iloc[:0], pd.DataFrame())
    monkeypatch.setattr(special, "fetch_live_inputs", lambda *args: live)
    monkeypatch.setattr(special.k_data, "load_data", lambda: weekly.iloc[:0])
    monkeypatch.setattr(
        special.forecast_weather,
        "enrich_forecasts",
        lambda *args: pytest.fail("forecast fetched twice"),
    )
    seen = []

    def capture(schedules):
        seen.append(schedules)
        return weekly.iloc[:0]

    monkeypatch.setattr(special, "build_kicker_frame", lambda h, c, r, s, *a: capture(s))
    monkeypatch.setattr(special, "build_defense_frame", lambda w, t, s, *a: capture(s))
    roster = pd.DataFrame({"player_id": ["K1"], "position": ["K"], "recent_team": ["BUF"]})
    statuses = [{"game_id": "g", "weather": "forecast"}]
    result = special.prepare_special_teams(
        2024,
        1,
        roster,
        espn,
        pd.DataFrame(columns=["season", "week"]),
        schedule_context=verified,
        weather_status=statuses,
    )
    assert len(seen) == 2
    for frame in seen:
        assert set(frame.week) == {1}  # no later schedule rows become synthetic games
        assert frame.temp.eq(52).all() and frame.wind.eq(17).all()
        assert frame.roof.eq("outdoors").all() and frame.surface.eq("grass").all()
        assert "venue" not in frame  # nested ESPN metadata must not break input hashing
    assert result.source_status["weather"] == statuses
