"""A neutral absence encoding requires explicit dated primary-source evidence."""

import json
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.data import club_practice as cp

pytestmark = pytest.mark.unit

PAGE_URL = "https://www.steelers.com/team/injury-report/"
ARTICLE_URL = "https://www.steelers.com/news/week-6-injury-report-buccaneers"


def _freeze_clock(monkeypatch, value):
    now = datetime.fromisoformat(value.replace("Z", "+00:00"))
    monkeypatch.setattr(cp, "datetime", SimpleNamespace(now=lambda tz: now.astimezone(tz)))


@pytest.fixture
def evidence(monkeypatch):
    _freeze_clock(monkeypatch, "2026-10-19T01:00Z")
    header = """<table><tr><th>Player</th><th>Position</th><th>Injury</th>
      <th>Wed</th><th>Thu</th><th>Fri</th><th>Game Status</th></tr>"""
    page = """<option selected value="/team/injury-report/week/REG-6">WEEK 6</option>
      <span class="nfl-o-injury-report__club-name">Pittsburgh Steelers</span>"""
    page += (
        header
        + """<tr><td>Pat Freiermuth</td><td>TE</td><td>Calf</td>
      <td>(-)</td><td>(-)</td><td>(-)</td><td>QUESTIONABLE</td></tr></table>
      <span class="nfl-o-injury-report__club-name">Tampa Bay Buccaneers</span>"""
    )
    page += (
        header
        + """</table><div class="d3-legend--content">Practice Status
      DNP - Did not participate LP - Limited FP - Full (-) - Not listed
      Game Status (-) - Not listed</div>
      <a data-link_type="Latest Injury Reports" data-link_name="Week 6 Injury Report"
      href="/news/week-6-injury-report-buccaneers">Week 6 Injury Report</a>"""
    )
    article = {
        "@type": "NewsArticle",
        "url": ARTICLE_URL,
        "headline": "Updated: Week 6 Injury Report (Buccaneers)",
        "keywords": ["Pittsburgh Steelers at Tampa Bay Buccaneers (2026-REG-6)"],
        "datePublished": "2026-10-17T18:00:00Z",
        "articleBody": """Saturday update: Pat Freiermuth questionable.
          Pittsburgh Steelers Participation/Injury Report, Week 7
          Game Status: Pat Freiermuth (Calf) - Questionable
          Friday, October 16
          No. 13 RB Rico Dowdle (Toe) - FP
          Thursday, October 15
          No. 13 RB Rico Dowdle (Toe) - FP
          Wednesday, October 14
          No. 13 RB Rico Dowdle (Toe) - LP
          Tampa Bay Buccaneers Participation/Injury Report, Week 7""",
    }
    files = {PAGE_URL: page, ARTICLE_URL: article}

    def fetch(url):
        content = files[url]
        if isinstance(content, dict):
            return '<script type="application/ld+json">' + json.dumps(content) + "</script>"
        return content

    monkeypatch.setattr(cp, "_fetch_html", fetch)
    return files


def verify(**updates):
    week = updates.get("week", 6)
    game = {
        "season": 2026,
        "week": week,
        "home_team": "TB",
        "away_team": "PIT",
        "kickoff": "2026-10-18T17:00Z",
    }
    report = {
        "name": "Pat Freiermuth",
        "position": "TE",
        "injuries": "Calf",
        "raw_practice_status": "",
        "game_status": "Questionable",
    }
    game.update(updates.get("game", {}))
    report.update(updates.get("report", {}))
    return cp.verify_published_absence(
        2026,
        week,
        updates.get("team", "PIT"),
        game,
        {"PIT": "Pittsburgh Steelers", "TB": "Tampa Bay Buccaneers"},
        report,
    )


def test_verified_absence_is_derived_and_never_claims_observed_full(evidence):
    result = verify()
    assert result["value"] == 2.0
    assert result["encoding"] == "derived_published_absence"
    assert result["observed_participation"] is None
    assert result["practice_dates"] == ["2026-10-14", "2026-10-15", "2026-10-16"]
    assert result["effective_period"] == {"season": 2026, "week": 6}
    assert result["url"] == PAGE_URL and result["dated_report_url"] == ARTICLE_URL


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("REG-6", "REG-5"),
        ("Tampa Bay Buccaneers", "Baltimore Ravens"),
        ("<td>(-)</td>", "<td></td>"),
        ("<td>(-)</td>", "<td>Unknown</td>"),
        ("<td>(-)</td>", "<td>FP</td>"),
        ("<th>Fri</th>", "<th>Sat</th>"),
        ("(-) - Not listed", "(-) - Unavailable"),
        ("<td>Calf</td>", "<td>Knee</td>"),
        ("<td>QUESTIONABLE</td>", "<td>OUT</td>"),
        ("<td>TE</td>", "<td>WR</td>"),
        ("Pat Freiermuth</td>", "Other Player</td>"),
        ('href="/news/', 'href="https://unrelated.test/news/'),
    ],
)
def test_unverified_club_table_is_rejected(evidence, before, after):
    evidence[PAGE_URL] = evidence[PAGE_URL].replace(before, after)
    with pytest.raises(ValueError):
        verify()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("keywords", ["Pittsburgh Steelers at Tampa Bay Buccaneers (2025-REG-6)"]),
        ("keywords", ["Pittsburgh Steelers at Tampa Bay Buccaneers (2026-REG-5)"]),
        ("keywords", ["Tampa Bay Buccaneers at Pittsburgh Steelers (2026-REG-6)"]),
        ("datePublished", "2026-10-10T18:00Z"),
        ("datePublished", "2026-10-15T18:00Z"),
        ("datePublished", "2026-10-19T18:00Z"),
        ("datePublished", "2026-10-17T18:00"),
        ("url", "https://www.steelers.com/news/another-report"),
        ("headline", "Week 5 Injury Report"),
    ],
)
def test_stale_or_wrong_article_is_rejected(evidence, key, value):
    evidence[ARTICLE_URL][key] = value
    with pytest.raises(ValueError):
        verify()


def test_missing_required_dated_report_is_rejected(evidence):
    evidence[ARTICLE_URL]["articleBody"] = evidence[ARTICLE_URL]["articleBody"].replace(
        "Friday, October 16", "Friday, October 9"
    )
    with pytest.raises(ValueError, match="dated practice"):
        verify()


def test_conflicting_article_participation_is_rejected(evidence):
    evidence[ARTICLE_URL]["articleBody"] = evidence[ARTICLE_URL]["articleBody"].replace(
        "Friday, October 16", "Friday, October 16\nPat Freiermuth (Calf) - FP"
    )
    with pytest.raises(ValueError, match="contradicts"):
        verify()


def test_empty_daily_sections_do_not_establish_published_coverage(evidence):
    evidence[ARTICLE_URL]["articleBody"] = evidence[ARTICLE_URL]["articleBody"].replace(
        "No. 13 RB Rico Dowdle (Toe) - FP", ""
    )
    with pytest.raises(ValueError, match="incomplete daily"):
        verify()


@pytest.mark.parametrize("status", [None, "Unknown", "Full Participation in Practice"])
def test_nonblank_nfl_status_cannot_use_absence_encoding(evidence, status):
    assert verify(report={"raw_practice_status": status}) is None


def test_unsupported_club_or_practice_schedule_remains_unknown(evidence):
    assert verify(team="TB") is None
    assert verify(game={"kickoff": "2026-10-19T00:20Z"}) is not None  # Sunday night in NY.
    assert verify(game={"kickoff": "2026-10-19T17:00Z"}) is None


def test_wrong_game_week_and_undated_kickoff_are_rejected(evidence):
    with pytest.raises(ValueError, match="season/week"):
        verify(game={"week": 5})
    with pytest.raises(ValueError, match="dated kickoff"):
        verify(game={"kickoff": "2026-10-18T13:00"})


def test_january_regular_season_uses_previous_calendar_year(evidence, monkeypatch):
    _freeze_clock(monkeypatch, "2027-01-11T01:00Z")
    new_url = ARTICLE_URL.replace("week-6", "week-18")
    article = evidence.pop(ARTICLE_URL)
    evidence[new_url] = article
    evidence[PAGE_URL] = (
        evidence[PAGE_URL]
        .replace("REG-6", "REG-18")
        .replace("Week 6", "Week 18")
        .replace("week-6-injury", "week-18-injury")
    )
    article["url"] = new_url
    article["headline"] = "Updated: Week 18 Injury Report (Buccaneers)"
    article["keywords"] = ["Pittsburgh Steelers at Tampa Bay Buccaneers (2026-REG-18)"]
    article["datePublished"] = "2027-01-09T18:00Z"
    article["articleBody"] = (
        article["articleBody"]
        .replace("Friday, October 16", "Friday, January 8")
        .replace("Thursday, October 15", "Thursday, January 7")
        .replace("Wednesday, October 14", "Wednesday, January 6")
    )
    result = verify(week=18, game={"kickoff": "2027-01-10T17:00Z"})
    assert result["effective_period"] == {"season": 2026, "week": 18}
    assert result["practice_dates"] == ["2027-01-06", "2027-01-07", "2027-01-08"]


def test_future_publication_cannot_establish_completed_practice(evidence, monkeypatch):
    _freeze_clock(monkeypatch, "2026-10-17T17:59:59Z")
    with pytest.raises(ValueError, match="future publication"):
        verify()


def test_publication_at_retrieval_time_is_valid_before_kickoff(evidence, monkeypatch):
    _freeze_clock(monkeypatch, "2026-10-17T18:00Z")
    result = verify()
    assert result["retrieved_at"] == result["published_at"] == "2026-10-17T18:00:00+00:00"


@pytest.mark.parametrize(
    "redirect",
    ["http://www.steelers.com/team/injury-report/", "https://unrelated.test/report"],
)
def test_fetch_rejects_insecure_or_official_host_redirects(monkeypatch, redirect):
    response = SimpleNamespace(geturl=lambda: redirect, read=lambda: b"untrusted")
    monkeypatch.setattr(cp.urllib.request, "urlopen", lambda *a, **kw: nullcontext(response))
    with pytest.raises(ValueError, match="HTTPS host"):
        cp._fetch_html(PAGE_URL)


def test_fetch_accepts_same_official_https_host(monkeypatch):
    response = SimpleNamespace(geturl=lambda: PAGE_URL, read=lambda: b"official HTML")
    monkeypatch.setattr(cp.urllib.request, "urlopen", lambda *a, **kw: nullcontext(response))
    assert cp._fetch_html(PAGE_URL) == "official HTML"
