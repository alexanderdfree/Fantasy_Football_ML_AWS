"""Historical QB role proxies and upcoming pregame population/role reconstruction."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import _build_inheritance_features
from src.features.roster_availability import weekly_roster_status
from src.serving import upcoming_week
from src.serving.upcoming_week import _availability_rosters

pytestmark = pytest.mark.unit
FEATURES = ["is_top_available", "inherited_opportunity"]
KEYS = ["player_id", "season", "week"]


def _qb_history():
    # Mirrors DAL's 2025 opener: active Joe Milton had the higher prior-season
    # opportunity average but no opener stat line. His previous team is immaterial
    # to the current roster, which establishes his DAL membership before kickoff.
    return pd.DataFrame(
        [
            ("dak", "DAL", 2024, 8, 14.0),
            ("milton", "NE", 2024, 18, 22.0),
            ("dak", "DAL", 2025, 1, 30.0),
            ("dak", "DAL", 2025, 2, 10.0),
        ],
        columns=["player_id", "recent_team", "season", "week", "total_fantasy_points_exp"],
    ).assign(position="QB")


def _rosters(status="ACT"):
    return pd.DataFrame(
        [(p, "DAL", 2025, w, s) for w in (1, 2) for p, s in (("dak", "ACT"), ("milton", status))],
        columns=["player_id", "team", "season", "week", "status"],
    ).assign(position="QB", game_type="REG")


def _qb_upcoming(week=1):
    history = _qb_history()
    history = history[history.season.lt(2025) | history.week.lt(week)].assign(_is_upcoming=False)
    upcoming = (
        _rosters()
        .query("week == @week")
        .rename(columns={"team": "recent_team"})[
            ["player_id", "recent_team", "season", "week", "position"]
        ]
    )
    upcoming = upcoming.assign(total_fantasy_points_exp=np.nan, _is_upcoming=True)
    return pd.concat([history, upcoming], ignore_index=True)


def _live_rows(frame):
    return frame[frame._is_upcoming].set_index("player_id")


def test_historical_qb_participation_reconstructs_role_without_fabricated_games():
    history = _qb_history()
    for roster in (_rosters(), None):
        result = _build_inheritance_features(history, None, roster)
        assert len(result) == len(history)
        assert result[KEYS].equals(history[KEYS])
        dak = result[result.player_id.eq("dak") & result.season.eq(2025)]
        assert dak.is_top_available.eq(1).all()
        assert dak.inherited_opportunity.eq(0).all()


def test_upcoming_qb_depth_disambiguates_active_backup_with_high_prior_role():
    frame = _qb_upcoming()
    result = _build_inheritance_features(
        frame, None, _rosters(), qb_depth_chart_ranks={"dak": 1, "milton": 2}
    )
    assert result[KEYS].equals(frame[KEYS])
    live = _live_rows(result)
    assert live.loc["dak", "is_top_available"] == 1.0
    assert live.loc["milton", "is_top_available"] == 0.0
    assert live.inherited_opportunity.eq(0).all()
    assert live.total_fantasy_points_exp.isna().all()


@pytest.mark.parametrize("depth", [None, {}, {"dak": np.nan, "milton": -1}, {"dak": "invalid"}])
def test_upcoming_depth_unavailable_preserves_prior_role_fallback(depth, caplog):
    result = _build_inheritance_features(
        _qb_upcoming(), None, _rosters(), qb_depth_chart_ranks=depth
    )
    live = _live_rows(result)
    assert live.loc["dak", "is_top_available"] == 0.0
    assert live.loc["milton", "is_top_available"] == 1.0
    assert "pregame QB depth unavailable" in caplog.text


def test_upcoming_equal_depth_uses_prior_role_to_break_tie():
    result = _build_inheritance_features(
        _qb_upcoming(), None, _rosters(), qb_depth_chart_ranks={"dak": 1, "milton": 1}
    )
    assert _live_rows(result).loc["milton", "is_top_available"] == 1.0


@pytest.mark.parametrize("week", [1, 2])
def test_upcoming_role_ignores_current_and_future_performance(week):
    frame = _qb_upcoming(week)
    ranks = {"dak": 1, "milton": 2}
    baseline = _build_inheritance_features(frame, None, _rosters(), qb_depth_chart_ranks=ranks)
    changed = frame.copy()
    changed.loc[changed._is_upcoming, "total_fantasy_points_exp"] = [0, 99999]
    future = _qb_history().iloc[[1]].assign(season=2026, total_fantasy_points_exp=99999)
    actual = _build_inheritance_features(
        pd.concat([changed, future], ignore_index=True),
        None,
        _rosters(),
        qb_depth_chart_ranks=ranks,
    )
    pd.testing.assert_frame_equal(baseline[FEATURES], actual.iloc[: len(frame)][FEATURES])


@pytest.mark.parametrize("status", ["RES", "INA", "DEV", "SUS", "CUT"])
def test_inactive_nonparticipant_does_not_compete(status):
    result = _build_inheritance_features(_qb_upcoming(), None, _rosters(status))
    row = result[result.player_id.eq("dak") & result.season.eq(2025) & result.week.eq(1)].iloc[0]
    assert row.is_top_available == 1.0
    assert row.inherited_opportunity == (22.0 if status in {"RES", "INA"} else 0.0)


@pytest.mark.parametrize("report_status", ["Out", "Doubtful"])
def test_reported_out_player_is_excluded_even_with_a_projection_row(report_status):
    injuries = pd.DataFrame(
        [
            dict(
                gsis_id="milton",
                position="QB",
                team="DAL",
                season=2025,
                week=1,
                report_status=report_status,
            )
        ]
    )
    result = _build_inheritance_features(
        _qb_upcoming(),
        injuries,
        _rosters(),
        qb_depth_chart_ranks={"milton": 1, "dak": 2},
    )
    week = result[result.season.eq(2025) & result.week.eq(1)].set_index("player_id")
    assert week.loc["dak", "is_top_available"] == 1.0
    assert week.loc["dak", "inherited_opportunity"] == 22.0
    assert week.loc["milton", "is_top_available"] == 0.0


def test_historical_participation_overrides_stale_absence_report_in_proxy():
    history = _qb_history()
    milton = history.iloc[[1]].assign(recent_team="DAL", season=2025, week=1)
    injuries = pd.DataFrame(
        [
            dict(
                gsis_id="milton",
                position="QB",
                team="DAL",
                season=2025,
                week=1,
                report_status="Out",
            )
        ]
    )
    result = _build_inheritance_features(
        pd.concat([history, milton], ignore_index=True), injuries, _rosters("INA")
    )
    week = result[result.season.eq(2025) & result.week.eq(1)].set_index("player_id")
    assert week.loc["milton", "is_top_available"] == 1.0
    assert week.loc["dak", "is_top_available"] == 0.0
    assert week.inherited_opportunity.eq(0).all()


def test_inherited_role_value_stays_historical_when_depth_selects_backup():
    injuries = pd.DataFrame(
        [dict(gsis_id="dak", position="QB", team="DAL", season=2025, week=2, report_status="Out")]
    )
    result = _build_inheritance_features(
        _qb_upcoming(2), injuries, _rosters(), qb_depth_chart_ranks={"dak": 1, "milton": 2}
    )
    live = _live_rows(result)
    assert live.loc["milton", "is_top_available"] == 1.0
    assert live.loc["milton", "inherited_opportunity"] == 30.0
    assert live.loc["dak", "is_top_available"] == 0.0


@pytest.mark.parametrize(
    "position,role", [("RB", "snap_pct_raw"), ("WR", "targets"), ("TE", "targets")]
)
def test_other_positions_keep_roster_ranking_independent_of_qb_depth(position, role):
    history = (
        _qb_history().rename(columns={"total_fantasy_points_exp": role}).assign(position=position)
    )
    roster = _rosters().assign(position=position)
    baseline = _build_inheritance_features(history, None, roster)
    result = _build_inheritance_features(
        history, None, roster, qb_depth_chart_ranks={"dak": 1, "milton": 2}
    )
    pd.testing.assert_frame_equal(baseline, result)
    row = result[result.player_id.eq("dak") & result.season.eq(2025) & result.week.eq(1)].iloc[0]
    assert row.is_top_available == 0.0


def test_future_stats_rosters_and_team_changes_cannot_change_earlier_features():
    history = _qb_history()
    baseline = _build_inheritance_features(history, None, _rosters())
    future = history.iloc[[1]].assign(
        recent_team="DAL", season=2025, week=3, total_fantasy_points_exp=9999.0
    )
    later_roster = _rosters().assign(week=3, status="RES")
    changed = _build_inheritance_features(
        pd.concat([history, future], ignore_index=True),
        None,
        pd.concat([_rosters(), later_roster], ignore_index=True),
    )
    pd.testing.assert_frame_equal(baseline[FEATURES], changed.iloc[: len(history)][FEATURES])


def test_missing_population_is_neutral_and_warns(caplog):
    result = _build_inheritance_features(_qb_upcoming(), None)
    assert (_live_rows(result)[FEATURES] == 0.0).all().all()
    assert "no participant fallback" in caplog.text
    # Coverage in another week cannot license reconstructing this week's roster.
    result = _build_inheritance_features(_qb_upcoming(), None, _rosters().assign(week=3))
    assert (_live_rows(result)[FEATURES] == 0.0).all().all()


def test_live_roster_population_is_usable_with_or_without_weekly_act_rows():
    history = _qb_upcoming()
    weekly = _rosters()
    live_roster = weekly[weekly.week.eq(1)].rename(columns={"team": "recent_team"})[
        ["player_id", "position", "recent_team"]
    ]
    reference = _build_inheritance_features(history, None, weekly)
    for archived in (weekly, None):
        population = _availability_rosters(2025, 1, live_roster, archived)
        live = _build_inheritance_features(history, None, population)
        rows = history.season.eq(2025) & history.week.eq(1)
        pd.testing.assert_frame_equal(reference.loc[rows, FEATURES], live.loc[rows, FEATURES])


@pytest.mark.parametrize("season", [2013, 2025])
def test_live_assembler_supplies_roster_population_to_feature_builder(monkeypatch, season):
    history = _qb_history().query("season == 2024").assign(season=season - 1)
    roster = (
        _rosters()
        .query("week == 1")
        .rename(columns={"team": "recent_team"})[["player_id", "position", "recent_team"]]
    )
    slate = pd.DataFrame({"recent_team": ["DAL"], "opponent_team": ["PHI"], "is_home": [0]})
    monkeypatch.setattr(upcoming_week, "_load_history", lambda *args: history)
    monkeypatch.setattr(upcoming_week, "_augment_schedules_cache", lambda *args: None)
    # Exercise the actual assembly and inheritance implementation, keeping unrelated
    # feature engineering outside this small population regression fixture.
    monkeypatch.setattr(upcoming_week, "build_features", _build_inheritance_features)
    result = upcoming_week.build_upcoming_week_frame(
        season,
        1,
        slate,
        roster,
        schedules=pd.DataFrame(),
        depth_chart_ranks={"dak": 1, "milton": 2},
    )
    live = result[result._is_upcoming].set_index("player_id")
    assert live.loc["dak", "is_top_available"] == 1.0
    assert live.loc["milton", "is_top_available"] == 0.0
    assert len(result[~result._is_upcoming]) == len(history)


def test_live_population_replaces_stale_act_and_preserves_inactives():
    weekly = _rosters("INA")
    stale = weekly.iloc[[0]].assign(player_id="stale")
    live_roster = pd.DataFrame(
        {"player_id": ["dak", "milton"], "position": ["QB", "QB"], "recent_team": ["DAL", "DAL"]}
    )
    population = _availability_rosters(2025, 1, live_roster, pd.concat([weekly, stale]))
    current = population[population.week.eq(1)]
    assert set(current.loc[current.status.eq("ACT"), "player_id"]) == {"dak", "milton"}
    result = _build_inheritance_features(_qb_upcoming(), None, population)
    dak = result[result.player_id.eq("dak") & result.season.eq(2025) & result.week.eq(1)].iloc[0]
    assert dak.is_top_available == 1.0
    assert dak.inherited_opportunity == 22.0


def test_postseason_roster_cannot_override_regular_season_availability():
    postseason = _rosters("INA").assign(game_type="WC")
    baseline = _build_inheritance_features(_qb_history(), None, _rosters())
    actual = _build_inheritance_features(_qb_history(), None, pd.concat([_rosters(), postseason]))
    pd.testing.assert_frame_equal(baseline[FEATURES], actual[FEATURES])


def test_legacy_weekly_descriptor_overrides_season_status_but_ngs_status_is_weekly(caplog):
    roster = pd.DataFrame(
        {
            "season": [2012, 2015, 2015, 2015, 2016, 2025],
            "status": ["RES", "ACT", "ACT", "RES", "RES", "ACT"],
            "status_description_abbr": ["A01", "I01", "new-code", None, "A01", None],
        }
    )
    assert weekly_roster_status(roster).tolist() == [
        "ACT",
        "INA",
        "UNKNOWN",
        "UNKNOWN",
        "RES",
        "ACT",
    ]
    assert "2 legacy roster rows" in caplog.text


def test_legacy_descriptor_prevents_false_reserve_vacancy():
    history = _qb_upcoming().assign(season=lambda x: x.season - 12)
    roster = _rosters("RES").assign(season=2013, status_description_abbr="A01")
    result = _build_inheritance_features(history, None, roster)
    dak = result[result.player_id.eq("dak") & result.season.eq(2013) & result.week.eq(1)].iloc[0]
    assert dak.is_top_available == 0.0
    assert dak.inherited_opportunity == 0.0
    roster.loc[roster.player_id.eq("milton"), "status_description_abbr"] = "I01"
    result = _build_inheritance_features(history, None, roster)
    dak = result[result.player_id.eq("dak") & result.season.eq(2013) & result.week.eq(1)].iloc[0]
    assert dak.is_top_available == 1.0
    assert dak.inherited_opportunity == 22.0
    roster.loc[roster.player_id.eq("milton"), "status_description_abbr"] = "unknown"
    result = _build_inheritance_features(history, None, roster)
    dak = result[result.player_id.eq("dak") & result.season.eq(2013) & result.week.eq(1)].iloc[0]
    # An unknown teammate can outrank Dak; the partial population cannot prove top status.
    assert dak.is_top_available == 0.0
    assert dak.inherited_opportunity == 0.0
