"""Official current-week practice reports, joined to the live player roster.

Unlisted players are healthy only when their team's report was actually
published. Missing team reports remain unknown and are imputed to the fitted
training mean by serving, rather than advertised as full participation.
"""

from __future__ import annotations

import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser

import pandas as pd

from src.data import nfl_source
from src.data.nflcom_loader import normalize_player_name, schedule_team_code_normalization

_STATUS = {
    "Full Participation in Practice": 2.0,
    "Limited Participation in Practice": 1.0,
    "Did Not Participate In Practice": 0.0,
}
_HEADERS = ["Player", "Position", "Injuries", "Practice Status", "Game Status"]


def _name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return normalize_player_name(ascii_name)


class _PracticeParser(HTMLParser):
    """Read report tables and their team titles, without a DOM dependency."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.selected = set()
        self.option_path = None
        self.option_parts = []
        self.records = []
        self.covered = set()
        self.team = ""
        self.title_depth = 0
        self.title_parts = []
        self.in_table = False
        self.valid_table = False
        self.row = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "option" and "selected" in attrs:
            self.option_path = attrs.get("value", "")
            self.option_parts = []
        if tag == "div":
            if self.title_depth:
                self.title_depth += 1
            elif "nfl-t-stats__title" in attrs.get("class", "").split():
                self.title_depth = 1
                self.title_parts = []
        if tag == "table":
            self.in_table = True
            self.valid_table = False
        if self.in_table and tag == "tr":
            self.row = []
        if self.in_table and tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.option_path is not None:
            self.option_parts.append(data)
        if self.title_depth:
            self.title_parts.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "option" and self.option_path is not None:
            # The year selector links to reg1 even on later-week pages. Only
            # the selected WEEK option proves which report we actually read.
            label = " ".join(" ".join(self.option_parts).split()).upper()
            if label.startswith("WEEK "):
                self.selected.add(self.option_path)
            self.option_path = None
        if tag == "div" and self.title_depth:
            self.title_depth -= 1
            if not self.title_depth:
                self.team = " ".join(" ".join(self.title_parts).split())
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.in_table:
            if self.row == _HEADERS and self.team:
                self.valid_table = True
                self.covered.add(self.team)
            elif self.valid_table and len(self.row) == 5:
                self.records.append(
                    {
                        "team_name": self.team,
                        "name": self.row[0],
                        "position": self.row[1],
                        "practice_status": _STATUS.get(self.row[3]),
                    }
                )
        if tag == "table":
            self.in_table = False


def parse_practice_report(html: str, season: int, week: int) -> _PracticeParser:
    parser = _PracticeParser()
    parser.feed(html)
    if f"/injuries/league/{season}/reg{week}" not in parser.selected:
        raise ValueError("NFL practice page returned a different season/week")
    if not parser.covered:
        raise ValueError("NFL practice page has no published report tables")
    return parser


def _fetch_official(season: int, week: int) -> _PracticeParser:
    url = f"https://www.nfl.com/injuries/league/{season}/reg{week}"
    with urllib.request.urlopen(url, timeout=25) as response:
        return parse_practice_report(response.read().decode("utf-8"), season, week)


@dataclass
class PracticeReport:
    values: dict[str, float]
    metadata: dict


def fetch_practice_report(season: int, week: int, roster: pd.DataFrame) -> PracticeReport:
    """Combine per-team coverage; current official tables override older feeds."""
    values = {}
    covered = set()
    errors = []
    try:
        injuries = nfl_source.injuries([season])
        injuries = injuries[injuries["season"].eq(season) & injuries["week"].eq(week)]
        for row in injuries.to_dict("records"):
            status = _STATUS.get(row.get("practice_status"))
            pid = row.get("gsis_id")
            if pd.notna(pid) and status is not None:
                values[str(pid)] = min(status, values.get(str(pid), 2.0))
                covered.add(str(row["team"]))
    except Exception as exc:  # real upstream boundary; preserve other source coverage
        errors.append(f"nflverse: {exc!r}")

    official_teams = set()
    unmatched = []
    try:
        official = _fetch_official(season, week)
        teams = nfl_source.teams()
        # The team directory also contains historical OAK/SD/STL entries with
        # the same nicknames. Canonicalize before last-value deduplication.
        names = dict(
            zip(
                teams["team_nick"],
                teams["team_abbr"].replace(schedule_team_code_normalization()),
                strict=True,
            )
        )
        official_teams = {names[name] for name in official.covered if name in names}
        lookup = {}
        reported = {}
        unknown = set()
        unresolved_groups = set()
        for row in roster.to_dict("records"):
            pid, team = str(row["player_id"]), row["recent_team"]
            key = (team, row["position"], _name(row["espn_name"]))
            lookup.setdefault(key, set()).add(pid)
        for row in official.records:
            key = (names.get(row["team_name"]), row["position"], _name(row["name"]))
            matches = lookup.get(key, set())
            if len(matches) != 1:
                unknown.update(matches)
                unresolved_groups.add(key[:2])
                if row["position"] in {"QB", "RB", "WR", "TE"}:
                    unmatched.append(row["name"])
                continue
            pid = next(iter(matches))
            if row["practice_status"] is None:
                unknown.add(pid)  # an unknown status is not a healthy report
            else:
                reported[pid] = row["practice_status"]
        for row in roster.to_dict("records"):
            pid = str(row["player_id"])
            if pid in unknown:
                values.pop(pid, None)
            elif pid in reported:
                values[pid] = reported[pid]
            elif (
                row["recent_team"] in official_teams
                and (row["recent_team"], row["position"]) not in unresolved_groups
            ):
                values[pid] = 2.0  # demonstrably absent from a published report
            # An unmatched name may be a roster alias (Andrew vs Drew). Keep
            # an ID-matched fallback, or unknown, until that group is resolved.
        covered.update(official_teams)
    except Exception as exc:
        errors.append(f"NFL.com: {exc!r}")

    expected = set(roster["recent_team"])
    # For nflverse-only teams, unlisted players may also be healthy. A team
    # with no report in either source gets no synthetic healthy values.
    for row in roster.to_dict("records"):
        if row["recent_team"] in covered - official_teams:
            values.setdefault(str(row["player_id"]), 2.0)
    roster_ids = set(roster["player_id"].astype(str))
    values = {pid: value for pid, value in values.items() if pid in roster_ids}
    metadata = {
        "provider": "NFL.com official injury reports; nflverse fallback",
        "url": f"https://www.nfl.com/injuries/league/{season}/reg{week}",
        "fetched_at": datetime.now(UTC).isoformat(),
        "covered_teams": sorted(expected & covered),
        "missing_teams": sorted(expected - covered),
        "known_players": len(values),
        "unknown_players": len(roster_ids - set(values)),
        "unmatched_report_names": sorted(set(unmatched)),
        "errors": errors,
    }
    return PracticeReport(values, metadata)
