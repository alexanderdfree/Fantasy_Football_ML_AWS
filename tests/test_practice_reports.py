"""Coverage, identity and freshness boundaries for official practice reports."""

import pandas as pd
import pytest

from src.serving import practice_reports as pr

pytestmark = pytest.mark.unit


def report_html(status="Limited Participation in Practice", name="Zay Flowers"):
    return f"""<select><option selected value="/injuries/league/2026/reg1">WEEK 1</option></select>
      <div class="nfl-t-stats__title"><div></div><div><span>Ravens</span></div></div>
      <table><tr><th>Player</th><th>Position</th><th>Injuries</th><th>Practice Status</th><th>Game Status</th></tr>
      <tr><td><a>{name}</a></td><td>WR</td><td>Hamstring</td><td>{status}</td><td></td></tr></table>"""


@pytest.fixture
def roster(monkeypatch):
    roster = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "espn_name": ["Zay Flowers", "Healthy Raven", "Patriot", "No Report"],
            "position": ["WR", "WR", "QB", "RB"],
            "recent_team": ["BAL", "BAL", "NE", "ATL"],
        }
    )
    primary = pd.DataFrame(
        {
            "season": [2026, 2025],
            "week": [1, 1],
            "team": ["NE", "ATL"],
            "gsis_id": ["c", "d"],
            "practice_status": ["Did Not Participate In Practice"] * 2,
        }
    )
    monkeypatch.setattr(pr.nfl_source, "injuries", lambda seasons: primary)
    monkeypatch.setattr(
        pr.nfl_source,
        "teams",
        lambda: pd.DataFrame(
            {"team_nick": ["Ravens", "Patriots", "Falcons"], "team_abbr": ["BAL", "NE", "ATL"]}
        ),
    )
    monkeypatch.setattr(
        pr, "_fetch_official", lambda s, w: pr.parse_practice_report(report_html(), s, w)
    )
    return roster


def test_official_source_fills_partial_primary_by_team(roster):
    result = pr.fetch_practice_report(2026, 1, roster)
    assert result.values == {"a": 1.0, "b": 2.0, "c": 0.0}
    assert result.metadata["covered_teams"] == ["BAL", "NE"]
    assert result.metadata["missing_teams"] == ["ATL"]
    assert result.metadata["unknown_players"] == 1


def test_wrong_week_is_rejected():
    with pytest.raises(ValueError, match="different season/week"):
        pr.parse_practice_report(report_html(), 2026, 2)


def test_missing_tables_is_not_a_healthy_report():
    with pytest.raises(ValueError, match="no published"):
        pr.parse_practice_report(
            '<option selected value="/injuries/league/2026/reg1">1</option>', 2026, 1
        )


def test_unknown_participation_is_not_full(roster, monkeypatch):
    monkeypatch.setattr(
        pr, "_fetch_official", lambda s, w: pr.parse_practice_report(report_html("Unknown"), s, w)
    )
    result = pr.fetch_practice_report(2026, 1, roster)
    assert "a" not in result.values
    assert result.values["b"] == 2.0


def test_new_official_full_report_overrides_old_limited(roster, monkeypatch):
    monkeypatch.setattr(
        pr.nfl_source,
        "injuries",
        lambda s: pd.DataFrame(
            {
                "season": [2026],
                "week": [1],
                "team": ["BAL"],
                "gsis_id": ["a"],
                "practice_status": ["Limited Participation in Practice"],
            }
        ),
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html("Full Participation in Practice"), s, w),
    )
    assert pr.fetch_practice_report(2026, 1, roster).values["a"] == 2.0


def test_source_failure_does_not_claim_missing_teams_healthy(roster, monkeypatch):
    def fail(*args):
        raise OSError("offline")

    monkeypatch.setattr(pr, "_fetch_official", fail)
    result = pr.fetch_practice_report(2026, 1, roster)
    assert result.values == {"c": 0.0}
    assert result.metadata["missing_teams"] == ["ATL", "BAL"]
    assert result.metadata["errors"]


def test_accented_report_names_match_current_roster(roster, monkeypatch):
    roster.loc[0, "espn_name"] = "Audric Estime"
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Audric Estimé"), s, w),
    )
    assert pr.fetch_practice_report(2026, 1, roster).values["a"] == 1.0


def test_ambiguous_name_does_not_assign_an_injury_to_wrong_player(roster):
    roster.loc[1, "espn_name"] = "Zay Flowers"
    result = pr.fetch_practice_report(2026, 1, roster)
    assert "a" not in result.values and "b" not in result.values


def test_historical_team_directory_aliases_are_canonicalized(roster, monkeypatch):
    roster.loc[:1, "recent_team"] = "LV"
    monkeypatch.setattr(
        pr.nfl_source,
        "teams",
        lambda: pd.DataFrame({"team_nick": ["Raiders", "Raiders"], "team_abbr": ["LV", "OAK"]}),
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html().replace("Ravens", "Raiders"), s, w),
    )
    report = pr.fetch_practice_report(2026, 1, roster)
    assert report.values["a"] == 1.0
    assert "LV" in report.metadata["covered_teams"]
