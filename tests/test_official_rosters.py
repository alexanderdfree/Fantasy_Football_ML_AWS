"""Inactive eligibility needs a current official table and corroborated identity."""

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import official_rosters as official

pytestmark = pytest.mark.unit

ROSTER_URL = "https://www.nfl.com/teams/chicago-bears/roster"
PROFILE_URL = "https://www.nfl.com/players/miller-moss/"


def roster_html(*, name="Miller Moss", position="QB", status="DEV", season=2026, link=None):
    # Reduced from the October 18 NFL roster's actual table/link structure.
    link = "/players/miller-moss/" if link is None else link
    header = "".join(f"<th>{column}</th>" for column in official._HEADERS)
    return (
        f"<title>Chicago Bears {season} Player Roster | NFL.com</title>"
        f"<table><thead><tr>{header}</tr></thead><tbody><tr>"
        '<td><div class="d3-o-media-object"><figure><img alt=""></figure>'
        f'<a class="nfl-o-roster__player-name nfl-o-cta--link" href="{link}">'
        f"{name}</a></div></td><td></td><td>{position}</td><td>{status}</td>"
        "<td>73</td><td>211</td><td>R</td><td>Louisville</td></tr></tbody></table>"
    )


def profile_document():
    # NFL profile JSON-LD uses startDate for debut, not the roster's season.
    return {
        "@type": "SportsTeam",
        "name": "Chicago Bears",
        "member": {
            "@type": "OrganizationRole",
            "startDate": "2026",
            "roleName": "QB",
            "member": {"@type": "Person", "name": "Miller Moss", "birthDate": "2002-03-19"},
        },
    }


def profile_html(document=None):
    return '<script type="application/ld+json">' + json.dumps(document) + "</script>"


@pytest.fixture
def sources(monkeypatch):
    document = profile_document()
    source = SimpleNamespace(
        pages={ROSTER_URL: roster_html(), PROFILE_URL: profile_html(document)},
        calls=[],
        document=document,
        reference=pd.DataFrame(
            [
                {
                    "season": 2026,
                    "week": 6,
                    "team": "CHI",
                    "position": "QB",
                    "player_id": "00-0041172",
                    "full_name": "Miller Moss",
                    "birth_date": "2002-03-19",
                    "espn_id": None,
                    "status": "DEV",
                }
            ]
        ),
        espn={
            "espn_id": "4431580",
            "espn_name": "Miller Moss",
            "position": "QB",
            "recent_team": "CHI",
            "roster_birth_date": "2002-03-19T08:00Z",
        },
    )

    def read(url):
        source.calls.append(url)
        value = source.pages[url]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(official, "_read_html", read)
    monkeypatch.setattr(
        official.nfl_source,
        "teams",
        lambda: pd.DataFrame(
            {
                "team_abbr": ["CHI", "LAC", "SD", "LA", "STL"],
                "team_name": [
                    "Chicago Bears",
                    "Los Angeles Chargers",
                    "San Diego Chargers",
                    "Los Angeles Rams",
                    "St. Louis Rams",
                ],
            }
        ),
    )
    return source


def verify(source, *, verifier=None, espn=False):
    verifier = verifier or official.OfficialRosterVerifier(2026, 6)
    return verifier.verify_inactive_identity(
        "CHI", "QB", "Miller Moss", source.reference, espn_player=source.espn if espn else None
    )


def test_known_practice_squad_identity_has_attributed_source_receipt(sources):
    result = verify(sources, espn=True)
    assert result["player_id"] == "00-0041172"
    assert result["status"] == "DEV"
    assert (result["team"], result["position"]) == ("CHI", "QB")
    assert result["source_season"] == 2026
    assert result["effective_period"] == {"season": 2026, "week": 6}
    assert result["roster_url"] == ROSTER_URL
    assert result["profile_url"] == PROFILE_URL
    assert result["retrieved_at"].endswith("+00:00")
    for key in ("content_id", "roster_sha256", "profile_sha256"):
        assert len(result[key]) == 64
    assert result["value_kind"] == "official_roster_inactive_identity"
    assert sources.calls == [ROSTER_URL, PROFILE_URL]


@pytest.mark.parametrize("status", ["DEV", "CUT"])
def test_verified_persistent_official_statuses_establish_exclusion(sources, status):
    sources.pages[ROSTER_URL] = roster_html(status=status)
    assert verify(sources)["status"] == status


@pytest.mark.parametrize(
    "status",
    ["ACT", "INA", "RES", "EXE", "RET", "SUS", "TRC", "TRD", "UNKNOWN", "", "active", "DEV ACT"],
)
def test_transient_unverified_or_unknown_status_does_not_establish_exclusion(sources, status):
    sources.pages[ROSTER_URL] = roster_html(status=status)
    assert verify(sources) is None
    assert sources.calls == [ROSTER_URL]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("season", 2025),
        ("week", 5),
        ("team", "LAC"),
        ("position", "WR"),
        ("full_name", "Other Quarterback"),
        ("birth_date", None),
        ("player_id", None),
    ],
)
def test_wrong_or_missing_weekly_context_does_not_establish_identity(sources, column, value):
    sources.reference.loc[0, column] = value
    assert verify(sources) is None
    assert not sources.calls


def test_missing_weekly_reference_cannot_resolve_from_name_alone(sources):
    sources.reference = None
    assert verify(sources) is None
    assert not sources.calls


def test_ambiguous_weekly_ids_and_conflicting_birthdays_remain_unresolved(sources):
    sources.reference = pd.concat(
        [sources.reference, sources.reference.assign(player_id="00-0041173")]
    )
    assert verify(sources) is None
    sources.reference.loc[:, "player_id"] = "00-0041172"
    sources.reference.iloc[1, sources.reference.columns.get_loc("birth_date")] = "2002-03-20"
    assert verify(sources) is None
    assert not sources.calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("espn_id", None),
        ("espn_name", "Other Quarterback"),
        ("roster_birth_date", "2002-03-20"),
        ("recent_team", "LAC"),
        ("position", "TE"),
    ],
)
def test_optional_espn_corroboration_must_agree(sources, field, value):
    sources.espn[field] = value
    assert verify(sources, espn=True) is None
    assert not sources.calls


def test_espn_id_conflict_in_weekly_reference_cannot_be_overridden(sources):
    sources.reference.loc[0, "espn_id"] = "123"
    assert verify(sources, espn=True) is None
    sources.reference.loc[0, "espn_id"] = "4431580.0"
    assert verify(sources, espn=True)["player_id"] == "00-0041172"


def test_conflicting_weekly_espn_ids_remain_unresolved_without_incoming_espn(sources):
    sources.reference = pd.concat(
        [sources.reference.assign(espn_id="123"), sources.reference.assign(espn_id="456")]
    )
    assert verify(sources) is None


@pytest.mark.parametrize("espn_id", ["invalid", "4431580.5", "-1"])
def test_malformed_nonmissing_weekly_espn_id_is_not_treated_as_absent(sources, espn_id):
    sources.reference.loc[0, "espn_id"] = espn_id
    assert verify(sources) is None
    assert not sources.calls


@pytest.mark.parametrize("change", ["season", "team", "headers", "name", "position", "duplicate"])
def test_wrong_or_ambiguous_official_roster_remains_unresolved(sources, change):
    html = roster_html()
    if change == "season":
        html = roster_html(season=2025)
    elif change == "team":
        html = html.replace("Chicago Bears", "Los Angeles Chargers")
    elif change == "headers":
        html = html.replace("<th>Status</th>", "<th>Eligibility</th>")
    elif change == "name":
        html = roster_html(name="Other Quarterback")
    elif change == "position":
        html = roster_html(position="TE")
    elif change == "duplicate":
        html += html
    sources.pages[ROSTER_URL] = html
    assert verify(sources) is None


@pytest.mark.parametrize(
    "link",
    [
        "https://other.example/players/miller-moss/",
        "http://www.nfl.com/players/miller-moss/",
        "/players/miller-moss/?season=2026",
        "/players/miller-moss/#info",
        "/news/miller-moss/",
        "",
    ],
)
def test_profile_must_be_an_unambiguous_official_player_link(sources, link):
    sources.pages[ROSTER_URL] = roster_html(link=link)
    assert verify(sources) is None
    assert sources.calls == [ROSTER_URL]


@pytest.mark.parametrize("change", ["team", "position", "name", "birthday", "missing", "duplicate"])
def test_conflicting_or_missing_profile_identity_remains_unresolved(sources, change):
    doc = sources.document
    if change == "team":
        doc["name"] = "Los Angeles Chargers"
    elif change == "position":
        doc["member"]["roleName"] = "TE"
    elif change == "name":
        doc["member"]["member"]["name"] = "Other Quarterback"
    elif change == "birthday":
        doc["member"]["member"]["birthDate"] = "2002-03-20"
    elif change == "missing":
        del doc["member"]["member"]["birthDate"]
    elif change == "duplicate":
        doc = [doc, copy.deepcopy(doc)]
    sources.pages[PROFILE_URL] = profile_html(doc)
    assert verify(sources) is None


def test_profile_badge_and_debut_year_do_not_override_current_roster(sources):
    sources.document["member"]["startDate"] = "2018"
    sources.reference.loc[0, "status"] = "ACT"
    sources.pages[ROSTER_URL] = roster_html(status="CUT")
    sources.pages[PROFILE_URL] = "<h3>active</h3>" + profile_html(sources.document)
    assert verify(sources)["status"] == "CUT"


def test_jsonld_graph_can_contain_profile_and_unrelated_metadata(sources):
    sources.pages[PROFILE_URL] = profile_html(
        {"@graph": [{"@type": "WebPage", "name": "Info"}, sources.document]}
    )
    assert verify(sources)["status"] == "DEV"


@pytest.mark.parametrize("unclosed", [False, True])
def test_valid_profile_cannot_hide_malformed_competing_jsonld(sources, unclosed):
    malformed = '<script type="application/ld+json">{"@type":"SportsTeam",'
    if not unclosed:
        malformed += "</script>"
    sources.pages[PROFILE_URL] += malformed
    assert verify(sources) is None


@pytest.mark.parametrize("url", [ROSTER_URL, PROFILE_URL])
def test_source_failure_remains_unresolved_and_is_cached_only_for_this_build(sources, url):
    good = sources.pages[url]
    sources.pages[url] = OSError("upstream unavailable")
    verifier = official.OfficialRosterVerifier(2026, 6)
    assert verify(sources, verifier=verifier) is None
    sources.pages[url] = good
    assert verify(sources, verifier=verifier) is None
    assert sources.calls.count(url) == 1
    assert verifier.errors
    assert verify(sources)["status"] == "DEV"


def test_successful_pages_are_reused_without_bypassing_fresh_identity_checks(sources):
    verifier = official.OfficialRosterVerifier(2026, 6)
    first = verify(sources, verifier=verifier)
    second = verify(sources, verifier=verifier)
    assert first == second
    assert sources.calls == [ROSTER_URL, PROFILE_URL]
    sources.reference.loc[0, "birth_date"] = "2002-03-20"
    assert verify(sources, verifier=verifier) is None
    assert sources.calls == [ROSTER_URL, PROFILE_URL]


def test_directory_failure_and_ambiguous_current_name_leave_gap_unresolved(sources, monkeypatch):
    monkeypatch.setattr(official.nfl_source, "teams", lambda: pd.DataFrame())
    assert verify(sources) is None
    monkeypatch.setattr(
        official.nfl_source,
        "teams",
        lambda: pd.DataFrame(
            {"team_abbr": ["CHI", "CHI"], "team_name": ["Chicago Bears", "Other Bears"]}
        ),
    )
    assert verify(sources) is None
    assert not sources.calls


def test_historical_directory_entry_cannot_replace_current_chargers_name(sources):
    sources.reference.loc[0, ["team", "position", "full_name", "birth_date", "player_id"]] = [
        "LAC",
        "TE",
        "Patrick Herbert",
        "2000-10-18",
        "00-0040331",
    ]
    roster_url = "https://www.nfl.com/teams/los-angeles-chargers/roster"
    profile_url = "https://www.nfl.com/players/patrick-herbert/"
    sources.pages[roster_url] = roster_html(
        name="Patrick Herbert", position="TE", link="/players/patrick-herbert/"
    ).replace("Chicago Bears", "Los Angeles Chargers")
    sources.document["name"] = "Los Angeles Chargers"
    sources.document["member"]["startDate"] = "2025"
    sources.document["member"]["roleName"] = "TE"
    sources.document["member"]["member"] = {
        "@type": "Person",
        "name": "Patrick Herbert",
        "birthDate": "2000-10-18",
    }
    sources.pages[profile_url] = profile_html(sources.document)
    result = official.OfficialRosterVerifier(2026, 6).verify_inactive_identity(
        "LAC", "TE", "Patrick Herbert", sources.reference
    )
    assert result["player_id"] == "00-0040331"
    assert sources.calls == [roster_url, profile_url]


def test_input_reference_is_not_mutated(sources):
    before = sources.reference.copy(deep=True)
    verify(sources)
    pd.testing.assert_frame_equal(sources.reference, before)
