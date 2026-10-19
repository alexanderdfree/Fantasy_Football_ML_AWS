"""Verify explicit absence from a dated official club practice report.

An NFL game-status addition can have no participation observation. A club's
explicit ``(-) = Not listed`` for every required practice day establishes the
same published-absence encoding used for unlisted players in training. It does
not establish measured full participation.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.roster_identity import name_key

# Add a club only after verifying its table, legend and dated article schema.
_CLUB_SITES = {"PIT": "https://www.steelers.com"}
_HEADERS = ["Player", "Position", "Injury", "Wed", "Thu", "Fri", "Game Status"]


class _ClubParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.selected = set()
        self.team = ""
        self.teams = set()
        self.rows = []
        self.links = []
        self.legend = []
        self._capture = None
        self._parts = []
        self._legend_depth = 0
        self._table = False
        self._valid = False
        self._row = []
        self._cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "option" and "selected" in attrs:
            self.selected.add(attrs.get("value", ""))
        if "nfl-o-injury-report__club-name" in classes:
            self._capture, self._parts = tag, []
        if tag == "div":
            if self._legend_depth:
                self._legend_depth += 1
            elif "d3-legend--content" in classes:
                self._legend_depth = 1
        if tag == "a" and attrs.get("data-link_type") == "Latest Injury Reports":
            self.links.append((attrs.get("data-link_name", ""), attrs.get("href", "")))
        if tag == "table":
            self._table, self._valid = True, False
        if self._table and tag == "tr":
            self._row = []
        if self._table and tag in ("td", "th"):
            self._cell = []

    def handle_data(self, data):
        if self._capture:
            self._parts.append(data)
        if self._legend_depth:
            self.legend.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == self._capture:
            self.team = " ".join(" ".join(self._parts).split())
            self._capture = None
        if tag == "div" and self._legend_depth:
            self._legend_depth -= 1
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        if tag == "tr" and self._table:
            if self._row == _HEADERS and self.team:
                self._valid = True
                self.teams.add(self.team)
            elif self._valid:
                self.rows.append((self.team, self._row))
        if tag == "table":
            self._table = False


class _ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.articles = []
        self._parts = None

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._parts = []

    def handle_data(self, data):
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._parts is not None:
            value = json.loads("".join(self._parts))
            if isinstance(value, dict) and value.get("@type") == "NewsArticle":
                self.articles.append(value)
            self._parts = None


def _fetch_html(url):
    with urllib.request.urlopen(url, timeout=25) as response:
        requested, received = urlsplit(url), urlsplit(response.geturl())
        if (
            requested.scheme != "https"
            or received.scheme != "https"
            or received.netloc != requested.netloc
        ):
            raise ValueError("Official club source redirected outside its HTTPS host")
        return response.read().decode("utf-8")


def verify_published_absence(season, week, team, game, team_names, report):
    """Return a source receipt only for explicit, fully dated non-listing.

    This bounded adapter currently verifies Sunday Wed/Thu/Fri club reports.
    Unsupported clubs/days and any mismatch remain unknown at the caller.
    """
    base = _CLUB_SITES.get(team)
    if base is None or report.get("raw_practice_status") != "":
        return None
    kickoff = pd.Timestamp(game["kickoff"])
    if kickoff.tzinfo is None:
        raise ValueError("Club practice verification requires a dated kickoff")
    day = kickoff.tz_convert("America/New_York").date()
    if day.weekday() != 6 or day.year - (day.month < 3) != season:
        return None
    if game.get("season") != season or game.get("week") != week:
        raise ValueError("Club practice game has a different season/week")
    home, away = game["home_team"], game["away_team"]
    if team not in (home, away):
        raise ValueError("Club practice game has a different team")
    expected_teams = {team_names[home], team_names[away]}
    url = base + "/team/injury-report/"
    page = _ClubParser()
    source_html = _fetch_html(url)
    page.feed(source_html)
    if page.selected != {f"/team/injury-report/week/REG-{week}"}:
        raise ValueError("Club practice page returned a different week")
    if page.teams != expected_teams:
        raise ValueError("Club practice page returned a different matchup")
    legend = " ".join(" ".join(page.legend).split())
    practice_legend = legend.split("Game Status", 1)[0]
    if not re.search(r"Practice Status.*\(-\)\s*-\s*Not listed", practice_legend):
        raise ValueError("Club practice page has no explicit not-listed legend")
    rows = [
        row
        for club, row in page.rows
        if club == team_names[team] and row and name_key(row[0]) == name_key(report["name"])
    ]
    if len(rows) != 1:
        raise ValueError("Club practice player identity is missing or ambiguous")
    row = rows[0]
    if (
        len(row) != 7
        or row[1] != report["position"]
        or name_key(row[2]) != name_key(report["injuries"])
        or row[6].casefold() != report["game_status"].casefold()
        or not row[2]
        or not row[6]
        or row[3:6] != ["(-)"] * 3
    ):
        raise ValueError("Club practice row does not verify explicit absence")
    links = {href for title, href in page.links if re.search(rf"\bWeek {week}\b", title)}
    if len(links) != 1:
        raise ValueError("Club practice page has no unique dated weekly article")
    article_url = urljoin(base, next(iter(links)))
    if urlsplit(article_url).netloc != urlsplit(base).netloc:
        raise ValueError("Club practice article must be on the official club site")
    article_page = _ArticleParser()
    article_html = _fetch_html(article_url)
    article_page.feed(article_html)
    game_keyword = f"{team_names[away]} at {team_names[home]} ({season}-REG-{week})"
    articles = [
        article
        for article in article_page.articles
        if game_keyword in article.get("keywords", [])
        and article.get("url") == article_url
        and re.search(rf"\bWeek {week}\b", article.get("headline", ""))
    ]
    if len(articles) != 1:
        raise ValueError("Club practice article has a different season/week/matchup")
    article = articles[0]
    published = pd.Timestamp(article["datePublished"])
    if published.tzinfo is None:
        raise ValueError("Club practice article has no timezone")
    published_day = published.tz_convert(ZoneInfo("America/New_York")).date()
    practice_days = [day - timedelta(days=offset) for offset in (4, 3, 2)]
    if not practice_days[-1] <= published_day <= day or published > kickoff:
        raise ValueError("Club practice article predates the final required practice")
    body = article.get("articleBody", "")
    section = re.search(re.escape(team_names[team]) + r"\s+Participation/Injury Report", body)
    opponent = away if team == home else home
    next_section = re.search(
        re.escape(team_names[opponent]) + r"\s+Participation/Injury Report",
        body[section.end() :] if section else "",
    )
    if section is None or next_section is None:
        raise ValueError("Club practice article has no scoped team practice section")
    body = body[section.end() : section.end() + next_section.start()]
    headings = list(
        re.finditer(
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), [A-Z][a-z]+ \d{1,2}\b",
            body,
        )
    )
    for practice_day in practice_days:
        heading = f"{practice_day:%A}, {practice_day:%B} {practice_day.day}"
        matching = [index for index, match in enumerate(headings) if match.group() == heading]
        if len(matching) != 1:
            raise ValueError("Club practice article is missing dated practice reports")
        index = matching[0]
        start = headings[index].end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        daily_report = body[start:end]
        if not re.search(r"\b(?:FP|LP|DNP)\b", daily_report):
            raise ValueError("Club article has an incomplete daily practice report")
        if name_key(report["name"]) in name_key(daily_report):
            raise ValueError("Club article contradicts the not-listed practice table")
    return {
        "provider": "Official club practice report",
        "url": url,
        "dated_report_url": article_url,
        "published_at": published.isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(source_html.encode()).hexdigest(),
        "dated_report_sha256": hashlib.sha256(article_html.encode()).hexdigest(),
        "effective_period": {"season": season, "week": week},
        "team": team,
        "opponent": away if team == home else home,
        "practice_dates": [value.isoformat() for value in practice_days],
        "observed_participation": None,
        "game_status": report["game_status"],
        "practice_listing": "not_listed",
        "encoding": "derived_published_absence",
        "value": 2.0,
    }
