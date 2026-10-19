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
    assert result.metadata["source_result"]["status"] == "partial"
    assert result.metadata["source_result"]["coverage"]["known_players"] == 3
    assert result.metadata["source_result"]["effective_period"] == {"season": 2026, "week": 1}


def test_wrong_week_is_rejected():
    with pytest.raises(ValueError, match="different season/week"):
        pr.parse_practice_report(report_html(), 2026, 2)


def test_missing_tables_is_not_a_healthy_report():
    with pytest.raises(ValueError, match="no published"):
        pr.parse_practice_report(
            '<option selected value="/injuries/league/2026/reg1">WEEK 1</option>', 2026, 1
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


def test_official_gsis_alias_overrides_fallback_and_resolves_healthy_teammate(roster, monkeypatch):
    roster.loc[0, "espn_name"] = "Drew Ogletree"
    reference = pd.DataFrame(
        {
            "player_id": ["a"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Andrew Ogletree"],
        }
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Andrew Ogletree"), s, w),
    )
    monkeypatch.setattr(
        pr.nfl_source,
        "injuries",
        lambda _: pd.DataFrame(
            {
                "season": [2026],
                "week": [1],
                "team": ["BAL"],
                "gsis_id": ["a"],
                "practice_status": ["Full Participation in Practice"],
            }
        ),
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert report.values["a"] == 1.0
    assert report.values["b"] == 2.0
    assert report.metadata["unmatched_report_names"] == []


@pytest.mark.parametrize(
    ("column", "value"),
    [("season", 2025), ("week", 2), ("team", "NE"), ("position", "TE"), ("player_id", "elsewhere")],
)
def test_alias_from_other_roster_context_cannot_assign_practice(roster, monkeypatch, column, value):
    roster.loc[0, "espn_name"] = "Drew Ogletree"
    reference = pd.DataFrame(
        {
            "player_id": ["a"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Andrew Ogletree"],
        }
    )
    reference.loc[0, column] = value
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Andrew Ogletree"), s, w),
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert "a" not in report.values and "b" not in report.values


def test_alias_ambiguity_including_primary_name_stays_unknown(roster, monkeypatch):
    roster.loc[0, "espn_name"] = "Drew Ogletree"
    roster.loc[1, "espn_name"] = "Andrew Ogletree"
    reference = pd.DataFrame(
        {
            "player_id": ["a"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Andrew Ogletree"],
        }
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Andrew Ogletree"), s, w),
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert "a" not in report.values and "b" not in report.values


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


def test_year_selector_cannot_validate_a_different_week():
    html = '<select><option selected value="/injuries/league/2026/reg1">2026</option></select>'
    html += report_html().replace("reg1", "reg2").replace("WEEK 1", "WEEK 2")
    with pytest.raises(ValueError, match="different season/week"):
        pr.parse_practice_report(html, 2026, 1)
    assert pr.parse_practice_report(html, 2026, 2).covered == {"Ravens"}


@pytest.mark.parametrize("has_fallback", [True, False])
def test_unmatched_alias_keeps_known_status_or_unknown(roster, monkeypatch, has_fallback):
    roster.loc[0, "espn_name"] = "Drew Ogletree"
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Andrew Ogletree"), s, w),
    )
    fallback = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "team": ["BAL"],
            "gsis_id": ["a"],
            "practice_status": ["Limited Participation in Practice"],
        }
    )
    monkeypatch.setattr(
        pr.nfl_source, "injuries", lambda s: fallback if has_fallback else fallback.iloc[:0]
    )
    result = pr.fetch_practice_report(2026, 1, roster)
    if has_fallback:
        assert result.values["a"] == 1.0
    else:
        assert "a" not in result.values
    assert "b" not in result.values  # cannot establish who the unmatched report describes
    assert result.metadata["unmatched_report_names"] == ["Andrew Ogletree"]


def test_out_players_reports_do_not_make_projected_teammates_unknown(roster):
    # The live builder removes Out players before projecting. The full verified
    # roster still identifies the official report as Zay's, not Healthy Raven's.
    projected = roster.loc[roster.player_id.ne("a")].copy()
    report = pr.fetch_practice_report(2026, 1, projected, identity_roster=roster)
    assert report.values == {"b": 2.0, "c": 0.0}
    assert report.metadata["unknown_players"] == 1  # ATL has no published report.
    assert report.metadata["unmatched_report_names"] == []
    assert "a" not in report.values


@pytest.mark.parametrize("status", ["RES", "INA", "EXE", "DEV", "CUT", "RET"])
def test_current_week_nonparticipant_does_not_block_published_absence(roster, monkeypatch, status):
    reference = pd.DataFrame(
        {
            "player_id": ["reserve"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Reported Reserve"],
            "status": [status],
        }
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Reported Reserve"), s, w),
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert report.values["a"] == report.values["b"] == 2.0
    assert "reserve" not in report.values
    assert report.metadata["unmatched_report_names"] == []


@pytest.mark.parametrize("status", ["ACT", "UNKNOWN", None])
def test_unconfirmed_nonparticipant_identity_still_blocks_teammate_defaults(
    roster, monkeypatch, status
):
    reference = pd.DataFrame(
        {
            "player_id": ["elsewhere"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Andrew Ogletree"],
            "status": [status],
        }
    )
    roster.loc[0, "espn_name"] = "Drew Ogletree"
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Andrew Ogletree"), s, w),
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert "a" not in report.values and "b" not in report.values
    assert report.metadata["unmatched_report_names"] == ["Andrew Ogletree"]


def test_report_from_outside_projection_slate_is_not_an_unmatched_identity(roster, monkeypatch):
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html().replace("Ravens", "Seahawks"), s, w),
    )
    monkeypatch.setattr(
        pr.nfl_source,
        "teams",
        lambda: pd.DataFrame({"team_nick": ["Seahawks"], "team_abbr": ["SEA"]}),
    )
    report = pr.fetch_practice_report(2026, 1, roster)
    assert report.values == {"c": 0.0}
    assert report.metadata["unmatched_report_names"] == []
    assert report.metadata["missing_teams"] == ["ATL", "BAL"]


def test_same_name_nonparticipant_preserves_active_identity_ambiguity(roster, monkeypatch):
    reference = pd.DataFrame(
        {
            "player_id": ["reserve"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Zay Flowers"],
            "status": ["RES"],
        }
    )
    report = pr.fetch_practice_report(2026, 1, roster, rosters_df=reference)
    assert "a" not in report.values and "b" not in report.values
    assert "reserve" not in report.values
    assert report.metadata["unmatched_report_names"] == ["Zay Flowers"]


@pytest.mark.parametrize("other_team,other_position", [("NE", "WR"), ("BAL", "TE")])
def test_inactive_reference_cannot_retarget_projected_players(
    roster, monkeypatch, other_team, other_position
):
    projected = pd.DataFrame(
        {
            "player_id": ["current", "teammate"],
            "espn_name": ["Current Player", "Teammate"],
            "position": ["WR", other_position],
            "recent_team": ["BAL", other_team],
        }
    )
    reference = pd.DataFrame(
        {
            "player_id": ["current"],
            "full_name": ["Current Player"],
            "position": [other_position],
            "team": [other_team],
            "season": [2026],
            "week": [1],
            "status": ["CUT"],
        }
    )
    html = report_html("Full Participation in Practice", name="Current Player")
    conflicting = report_html("Did Not Participate In Practice", name="Current Player")
    conflicting = conflicting.replace("Ravens", "Patriots" if other_team == "NE" else "Ravens")
    html += conflicting.replace("<td>WR</td>", f"<td>{other_position}</td>")
    monkeypatch.setattr(pr, "_fetch_official", lambda s, w: pr.parse_practice_report(html, s, w))
    report = pr.fetch_practice_report(2026, 1, projected, rosters_df=reference)
    assert report.values["current"] == 2.0
    assert report.metadata["errors"] == []


def test_blank_nfl_status_requires_verified_club_absence(roster, monkeypatch):
    html = report_html("").replace("<td></td></tr>", "<td>Questionable</td></tr>")
    monkeypatch.setattr(pr, "_fetch_official", lambda s, w: pr.parse_practice_report(html, s, w))
    receipt = {
        "url": "https://official.test/team/injury-report/",
        "effective_period": {"season": 2026, "week": 1},
        "practice_dates": ["2026-09-09", "2026-09-10", "2026-09-11"],
        "observed_participation": None,
        "encoding": "derived_published_absence",
        "value": 2.0,
    }

    def club_absence(season, week, team, row, schedules, teams):
        assert row["game_status"] == "Questionable"
        assert row["raw_practice_status"] == ""
        return receipt

    monkeypatch.setattr(pr, "_club_absence", club_absence)
    result = pr.fetch_practice_report(2026, 1, roster)
    assert result.values["a"] == 2.0
    assert result.metadata["published_absence_evidence"] == [{"player_id": "a", **receipt}]
    assert result.metadata["derived_published_absence_players"] == 1
    assert result.metadata["source_result"]["coverage"]["derived_published_absence_players"] == 1


@pytest.mark.parametrize("outcome", ["missing", "failure"])
def test_blank_status_without_verified_absence_stays_unknown(roster, monkeypatch, outcome):
    monkeypatch.setattr(
        pr, "_fetch_official", lambda s, w: pr.parse_practice_report(report_html(""), s, w)
    )

    def club_absence(*args):
        if outcome == "failure":
            raise ValueError("wrong week")
        return None

    monkeypatch.setattr(pr, "_club_absence", club_absence)
    result = pr.fetch_practice_report(2026, 1, roster)
    assert "a" not in result.values
    assert result.metadata["derived_published_absence_players"] == 0
    assert result.metadata["unknown_players"] == 2


def test_unknown_token_never_uses_published_absence(roster, monkeypatch):
    monkeypatch.setattr(
        pr, "_fetch_official", lambda s, w: pr.parse_practice_report(report_html("Unknown"), s, w)
    )

    def unexpected(*args):
        pytest.fail("An unknown token is not a published absence")

    monkeypatch.setattr(pr, "_club_absence", unexpected)
    assert "a" not in pr.fetch_practice_report(2026, 1, roster).values


def test_official_inactive_identity_resolves_stale_active_reference(roster, monkeypatch):
    reference = pd.DataFrame(
        {
            "player_id": ["released"],
            "team": ["BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Released Receiver"],
            "status": ["ACT"],
        }
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Released Receiver"), s, w),
    )
    receipt = {"player_id": "released", "status": "CUT", "roster_url": "https://official.test"}

    class Verifier:
        def verify_inactive_identity(self, team, position, name, current):
            assert (team, position, name) == ("BAL", "WR", "Released Receiver")
            assert current.iloc[0]["season"] == 2026 and current.iloc[0]["week"] == 1
            return receipt

    result = pr.fetch_practice_report(
        2026, 1, roster, rosters_df=reference, official_roster_verifier=Verifier()
    )
    assert result.values["a"] == result.values["b"] == 2.0
    assert "released" not in result.values
    assert result.metadata["unmatched_report_names"] == []
    assert result.metadata["inactive_report_identities"] == [receipt]


@pytest.mark.parametrize("conflict", ["projected_id", "wrong_context", "missing", "failure"])
def test_official_identity_cannot_override_roster_conflict(roster, monkeypatch, conflict):
    reference = pd.DataFrame(
        {
            "player_id": ["a" if conflict == "projected_id" else "released"],
            "team": ["NE" if conflict in {"wrong_context", "projected_id"} else "BAL"],
            "position": ["WR"],
            "season": [2026],
            "week": [1],
            "full_name": ["Released Receiver"],
            "status": ["ACT"],
        }
    )
    monkeypatch.setattr(
        pr,
        "_fetch_official",
        lambda s, w: pr.parse_practice_report(report_html(name="Released Receiver"), s, w),
    )

    class Verifier:
        def verify_inactive_identity(self, *args):
            if conflict == "failure":
                raise OSError("offline")
            if conflict == "missing":
                return None
            return {"player_id": "a" if conflict == "projected_id" else "released"}

    result = pr.fetch_practice_report(
        2026, 1, roster, rosters_df=reference, official_roster_verifier=Verifier()
    )
    assert "a" not in result.values and "b" not in result.values
    assert result.metadata["unmatched_report_names"] == ["Released Receiver"]


def test_absence_without_schedule_context_is_not_verified(roster):
    row = pr.parse_practice_report(report_html(""), 2026, 1).records[0]
    assert pr._club_absence(2026, 1, "BAL", row, None, pd.DataFrame()) is None
