"""Official roof decisions must belong to the projected game."""

import hashlib
import json
from html import escape

import pandas as pd
import pytest

from src.data import roof_reports as reports

pytestmark = pytest.mark.unit

TEAM_NAMES = {"ATL": "Atlanta Falcons", "CHI": "Chicago Bears", "SF": "San Francisco 49ers"}


def game(home="ATL", **changes):
    result = {
        "season": 2026,
        "week": 6,
        "game_type": "REG",
        "home_team": home,
        "away_team": "CHI" if home == "ATL" else "TEN",
        "kickoff": "2026-10-18T17:00:00Z",
        "neutral_site": False,
        "venue": {"fullName": "Mercedes-Benz Stadium" if home == "ATL" else "Lucas Oil Stadium"},
    }
    return {**result, **changes}


def atlanta_page(roof="Closed", **changes):
    event = {
        "@type": "SportsEvent",
        "homeTeam": {"name": "Atlanta Falcons"},
        "awayTeam": {"name": "Chicago Bears"},
        "location": {"name": "Mercedes-Benz Stadium"},
        "startDate": "2026-10-18T17:00:00.000Z",
        **changes,
    }
    return f"""
        <script type="application/ld+json">{json.dumps(event)}</script>
        <div class="events_feature_details-subtext-container">
          <div class="events_feature_details_label">Roof Status</div>
          <div class="events_feature_details_dt">{roof}</div>
        </div>"""


def indianapolis_page(roof="Open", **changes):
    module = {
        "GameDateTime": "2026-10-18T13:00:00-04:00",
        "IsNeutralSite": False,
        "GameInformation": {
            "Week": {"Season": 2026, "Week": 6, "SeasonType": "REG"},
            "Home": {"Abbreviation": "IND"},
            "Away": {"Abbreviation": "TEN"},
            "Venue": {"Name": "Lucas Oil Stadium"},
        },
        **changes,
    }
    payload = escape(json.dumps({"Name": "GameTileLiveItems", "Module": module}), quote=True)
    return f"""
        <div data-gameid="official-game-id" data-json-module="{payload}"></div>
        <div class="nfl-c-custom-promo__description">
          <h4>Roof &amp; Window</h4>
          <p><strong>🏟️ Roof:</strong> {roof}</p>
          <p><strong>🪟 Window:</strong> Closed</p>
          <p>Gates Open at 11:00 AM.</p>
        </div>"""


def test_official_roof_field_is_distinct_from_open_gates_and_window():
    assert reports.parse_roof_report(atlanta_page(), game(), team_names=TEAM_NAMES) == "closed"
    assert reports.parse_roof_report(indianapolis_page(), game("IND")) == "open"
    assert reports.parse_roof_report(indianapolis_page("Closed"), game("IND")) == "closed"


@pytest.mark.parametrize("roof", ["TBD", "Weather Permitting", "Open at kickoff", "", "Unknown"])
@pytest.mark.parametrize("home", ["ATL", "IND"])
def test_unannounced_or_conditional_roof_remains_unknown(roof, home):
    html = atlanta_page(roof) if home == "ATL" else indianapolis_page(roof)
    assert reports.parse_roof_report(html, game(home), team_names=TEAM_NAMES) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"awayTeam": {"name": "San Francisco 49ers"}},
        {"homeTeam": {"name": "Chicago Bears"}, "awayTeam": {"name": "Atlanta Falcons"}},
        {"location": {"name": "Soldier Field"}},
        {"startDate": "2025-10-18T17:00:00Z"},
        {"startDate": "2026-10-18T20:25:00Z"},
        {"startDate": "2026-10-18T17:00:00"},
        {"startDate": None},
    ],
)
def test_atlanta_event_identity_must_match(changes):
    assert reports.parse_roof_report(atlanta_page(**changes), game(), team_names=TEAM_NAMES) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"away_team": "HOU"},
        {"home_team": "TEN"},
        {"season": 2025},
        {"week": 7},
        {"kickoff": "2026-10-18T20:25:00Z"},
        {"venue": {"fullName": "Nissan Stadium"}},
        {"neutral_site": True},
        {"game_type": "PRE"},
    ],
)
def test_indianapolis_event_identity_must_match(changes):
    assert reports.parse_roof_report(indianapolis_page(), game("IND", **changes)) is None


@pytest.mark.parametrize("home", ["ATL", "IND"])
def test_conflicting_roof_fields_are_not_selected_arbitrarily(home):
    make = atlanta_page if home == "ATL" else indianapolis_page
    html = make("Open") + make("Closed")
    assert reports.parse_roof_report(html, game(home), team_names=TEAM_NAMES) is None


def test_indianapolis_static_page_requires_embedded_game_identity():
    html = '<div class="nfl-c-custom-promo__description"><h4>Roof &amp; Window</h4><p>Roof: Closed</p></div>'
    assert reports.parse_roof_report(html, game("IND")) is None


@pytest.mark.parametrize("home", ["ATL", "IND"])
def test_malformed_embedded_event_does_not_prove_roof(home):
    make = atlanta_page if home == "ATL" else indianapolis_page
    html = make().replace('"homeTeam": {"name": "Atlanta Falcons"}', '"homeTeam": "ATL"')
    if home == "IND":
        html = html.replace(
            "&quot;Module&quot;: {", "&quot;Module&quot;: null, &quot;unused&quot;: {"
        )
    assert reports.parse_roof_report(html, game(home), team_names=TEAM_NAMES) is None


def test_atlanta_discovery_uses_official_index_and_preserves_provenance(monkeypatch):
    index = "https://www.mercedesbenzstadium.com/"
    detail = index + "falcons-gameday/chicago-bears"
    pages = {
        index: f'<a href="{detail}">Game Details</a><a href="//unverified.example/falcons-gameday/chicago-bears">Roof Open</a>',
        detail: atlanta_page(),
    }
    visited = []

    def read(url):
        visited.append(url)
        return pages[url]

    monkeypatch.setattr(reports, "_read_html", read)
    monkeypatch.setattr(
        reports.nfl_source,
        "teams",
        lambda: pd.DataFrame(
            {"team_abbr": list(TEAM_NAMES), "team_name": list(TEAM_NAMES.values())}
        ),
    )
    result = reports.fetch_verified_roof(game())
    assert visited == [index, detail]
    assert result["roof"] == "closed"
    assert result["url"] == detail
    assert result["source_html_sha256"] == hashlib.sha256(pages[detail].encode("utf-8")).hexdigest()
    assert result["provider"] == "Mercedes-Benz Stadium"
    assert result["value_kind"] == "official_game_roof_announcement"
    assert result["effective_event"] == {
        "season": 2026,
        "week": 6,
        "home_team": "ATL",
        "away_team": "CHI",
        "venue": "Mercedes-Benz Stadium",
        "kickoff": "2026-10-18T17:00:00+00:00",
    }
    assert pd.Timestamp(result["retrieved_at"]).tzinfo is not None


@pytest.mark.parametrize(
    "current,modern_name,historical,old_name",
    [
        ("LAC", "Los Angeles Chargers", "SD", "San Diego Chargers"),
        ("LV", "Las Vegas Raiders", "OAK", "Oakland Raiders"),
        ("LA", "Los Angeles Rams", "STL", "St. Louis Rams"),
    ],
)
def test_historical_directory_rows_cannot_replace_current_opponent_name(
    monkeypatch, current, modern_name, historical, old_name
):
    index = "https://www.mercedesbenzstadium.com/"
    detail = index + "falcons-gameday/" + modern_name.lower().replace(" ", "-")
    pages = {
        index: f'<a href="{detail}">Game Details</a>',
        detail: atlanta_page(awayTeam={"name": modern_name}),
    }
    monkeypatch.setattr(reports, "_read_html", pages.__getitem__)
    monkeypatch.setattr(
        reports.nfl_source,
        "teams",
        lambda: pd.DataFrame(
            {
                "team_abbr": ["ATL", current, historical],
                "team_name": ["Atlanta Falcons", modern_name, old_name],
            }
        ),
    )
    result = reports.fetch_verified_roof(game(away_team=current))
    assert result["roof"] == "closed"
    assert result["url"] == detail


def test_indianapolis_discovery_follows_rotated_home_game_url(monkeypatch):
    index = "https://www.colts.com/game-day/"
    detail = "https://www.colts.com/game-day/regular-season-game-4"
    upcoming = game("IND", week=7, kickoff="2026-10-25T17:00:00Z")
    pages = {
        index: f'<a href="{detail}">Next Home Game</a>',
        detail: indianapolis_page("Closed")
        .replace("2026-10-18", "2026-10-25")
        .replace("&quot;Week&quot;: 6", "&quot;Week&quot;: 7"),
    }
    monkeypatch.setattr(reports, "_read_html", pages.__getitem__)
    result = reports.fetch_verified_roof(upcoming)
    assert result["roof"] == "closed"
    assert result["url"] == detail
    assert result["effective_event"]["week"] == 7


def test_stale_home_game_link_and_source_outage_remain_unknown(monkeypatch):
    index = "https://www.colts.com/game-day/"
    detail = "https://www.colts.com/game-day/regular-season-game-3"
    monkeypatch.setattr(
        reports,
        "_read_html",
        {index: f'<a href="{detail}">Next Home Game</a>', detail: indianapolis_page()}.__getitem__,
    )
    assert reports.fetch_verified_roof(game("IND", week=7)) is None

    def unavailable(url):
        raise OSError("upstream unavailable")

    monkeypatch.setattr(reports, "_read_html", unavailable)
    assert reports.fetch_verified_roof(game("IND")) is None


def test_no_request_for_unverified_venue(monkeypatch):
    monkeypatch.setattr(
        reports, "_read_html", lambda _: pytest.fail("Unexpected roof source request")
    )
    assert reports.fetch_verified_roof(game("ATL", neutral_site=True)) is None
    assert reports.fetch_verified_roof(game("ATL", venue={"fullName": "Other Venue"})) is None
