"""Game-specific roof announcements from official club and stadium pages.

The venue's retractable roof and ESPN's indoor flag describe the building,
not its state for a particular game. Accept a published open/closed decision
only after the page's event identity agrees with the live schedule.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import pandas as pd

from src.data import nfl_source
from src.data.identity import normalize_team_code, schedule_team_code_normalization

_SOURCES = {
    "ATL": ("https://www.mercedesbenzstadium.com/", "/falcons-gameday/", "Mercedes-Benz Stadium"),
    "IND": (
        "https://www.colts.com/game-day/",
        "/game-day/regular-season-game-",
        "Lucas Oil Stadium",
    ),
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass
class _Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.walk()

    def text(self):
        return " ".join(
            " ".join(
                child.text() if isinstance(child, _Node) else child for child in self.children
            ).split()
        )

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()


class _Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _kickoff(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Roof announcement requires a timezone-aware kickoff")
    return stamp


def _source(game):
    source = _SOURCES.get(game.get("home_team"))
    if (
        source is None
        or game.get("neutral_site")
        or game.get("game_type", "REG") != "REG"
        or (game.get("venue") or {}).get("fullName") != source[2]
    ):
        return None
    return source


def _only_roof(values):
    roofs = {value.strip().lower() for value in values}
    return next(iter(roofs)) if len(roofs) == 1 and roofs <= {"open", "closed"} else None


def _sports_events(value):
    if isinstance(value, list):
        for item in value:
            yield from _sports_events(item)
    elif isinstance(value, dict):
        if value.get("@type") == "SportsEvent":
            yield value
        yield from _sports_events(value.get("@graph", []))


def _atlanta_roof(page, game, team_names):
    events = []
    for node in page.root.walk():
        if node.tag == "script" and node.attrs.get("type") == "application/ld+json":
            events.extend(_sports_events(json.loads(node.text())))
    if len(events) != 1:
        return None
    event = events[0]
    for side in ("home", "away"):
        expected = team_names.get(normalize_team_code(game[f"{side}_team"]))
        if not expected or (event.get(f"{side}Team") or {}).get("name") != expected:
            return None
    if (event.get("location") or {}).get("name") != _SOURCES["ATL"][2] or _kickoff(
        event.get("startDate")
    ) != _kickoff(game["kickoff"]):
        return None
    values = []
    for node in page.root.walk():
        if not node.has_class("events_feature_details-subtext-container"):
            continue
        labels = [n.text() for n in node.walk() if n.has_class("events_feature_details_label")]
        if labels == ["Roof Status"]:
            values.extend(n.text() for n in node.walk() if n.has_class("events_feature_details_dt"))
    return _only_roof(values)


def _indianapolis_roof(page, game):
    modules = []
    for node in page.root.walk():
        raw = node.attrs.get("data-json-module")
        if raw:
            payload = json.loads(raw)
            if payload.get("Name") == "GameTileLiveItems":
                modules.append(payload["Module"])
    if len(modules) != 1:
        return None
    module = modules[0]
    info = module["GameInformation"]
    period = info["Week"]
    if (
        period.get("Season") != int(game["season"])
        or period.get("Week") != int(game["week"])
        or period.get("SeasonType") != "REG"
        or module.get("IsNeutralSite") is not False
        or _kickoff(module.get("GameDateTime")) != _kickoff(game["kickoff"])
        or (info.get("Venue") or {}).get("Name") != _SOURCES["IND"][2]
    ):
        return None
    for side in ("home", "away"):
        if normalize_team_code(
            (info.get(side.title()) or {}).get("Abbreviation")
        ) != normalize_team_code(game[f"{side}_team"]):
            return None
    values = []
    for node in page.root.walk():
        if not node.has_class("nfl-c-custom-promo__description"):
            continue
        if not any(n.tag == "h4" and n.text() == "Roof & Window" for n in node.walk()):
            continue
        for paragraph in node.walk():
            if paragraph.tag == "p":
                match = re.search(r"(?:^|\s)Roof:\s*(.*)$", paragraph.text())
                if match:
                    values.append(match.group(1))
    return _only_roof(values)


def parse_roof_report(html: str, game: dict, *, team_names: dict | None = None) -> str | None:
    """Parse a verified decision; stale, incomplete or conflicting pages give None.

    ``team_names`` maps canonical team abbreviations to NFL directory names
    for Atlanta's JSON-LD, which identifies its opponents by name.
    """
    if _source(game) is None:
        return None
    try:
        page = _Page(html)
        if game["home_team"] == "ATL":
            return _atlanta_roof(page, game, team_names or {})
        return _indianapolis_roof(page, game)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _read_html(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        if urlsplit(response.geturl()).netloc != urlsplit(url).netloc:
            raise ValueError("Official roof source redirected to a different host")
        return response.read().decode("utf-8")


def _detail_urls(html, index_url, path_prefix):
    urls = set()
    for node in _Page(html).root.walk():
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        url = urljoin(index_url, node.attrs["href"])
        parts = urlsplit(url)
        if (
            parts.scheme == "https"
            and parts.netloc == urlsplit(index_url).netloc
            and parts.path.startswith(path_prefix)
            and not parts.query
            and not parts.fragment
        ):
            urls.add(url)
    return sorted(urls)


def fetch_verified_roof(game: dict) -> dict | None:
    """Discover the current official game page and return attributed evidence.

    No roof value or mutable game-page ordinal is frozen here. A failed
    lookup leaves the caller's unknown roof available for normal imputation.
    """
    source = _source(game)
    if source is None:
        return None
    index_url, path_prefix, venue = source
    try:
        team_names = {}
        if game["home_team"] == "ATL":
            teams = nfl_source.teams()
            # Historical relocation rows follow modern rows in the directory.
            # Exclude those names before aliases collapse onto current codes.
            teams = teams[~teams["team_abbr"].isin(schedule_team_code_normalization())]
            team_names = dict(
                zip(teams["team_abbr"].map(normalize_team_code), teams["team_name"], strict=True)
            )
        urls = _detail_urls(_read_html(index_url), index_url, path_prefix)
        if game["home_team"] == "ATL":
            away_name = team_names.get(normalize_team_code(game["away_team"]), "")
            slug = re.sub(r"[^a-z0-9]+", "-", away_name.lower()).strip("-")
            urls = (
                [url for url in urls if urlsplit(url).path.rstrip("/").endswith("/" + slug)]
                if slug
                else []
            )
        for url in urls:
            html = _read_html(url)
            roof = parse_roof_report(html, game, team_names=team_names)
            if roof is not None:
                return {
                    "roof": roof,
                    "provider": "Mercedes-Benz Stadium"
                    if game["home_team"] == "ATL"
                    else "Indianapolis Colts",
                    "url": url,
                    "source_html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "value_kind": "official_game_roof_announcement",
                    "effective_event": {
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "venue": venue,
                        "kickoff": _kickoff(game["kickoff"]).isoformat(),
                    },
                }
    except Exception as exc:  # real source boundary; unknown remains unknown
        print(f"[roof_reports] {game.get('home_team')} announcement unavailable: {exc!r}")
    return None
