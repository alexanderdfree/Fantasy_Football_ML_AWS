"""Corroborate inactive live-roster identities with the NFL's current roster.

The roster table supplies eligibility; profile JSON-LD supplies date of birth.
Neither a profile's display badge nor a weekly reference alone can establish
current inactive status. This adapter never creates an active roster member.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import pandas as pd

from src.data import nfl_source
from src.data.identity import normalize_team_code, schedule_team_code_normalization
from src.data.roster_identity import current_rosters, name_key, player_id
from src.data.source_result import content_identity

_HEADERS = ["Player", "No", "Pos", "Status", "Height", "Weight", "Experience", "College"]
# These persistent states are corroborated by current club roster/transaction
# evidence. A season roster's INA may describe a prior game's inactive list;
# other status codes need their own verified semantics before being accepted.
_INACTIVE = frozenset({"DEV", "CUT"})


def _text(parts) -> str:
    return " ".join(" ".join(parts).split())


def _team_code(value) -> str:
    code = normalize_team_code(value)
    return "LA" if code == "LAR" else code


def _birthday(value) -> str | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp.date().isoformat() if pd.notna(timestamp) else None


def _espn_id(value) -> str | None:
    number = pd.to_numeric(value, errors="coerce")
    return str(int(number)) if pd.notna(number) and number > 0 and number % 1 == 0 else None


class _RosterParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.in_title = False
        self.in_table = False
        self.valid_table = False
        self.row = []
        self.cell = None
        self.links = []
        self.records = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "table":
            self.in_table = True
            self.valid_table = False
        if self.in_table and tag == "tr":
            self.row = []
            self.links = []
        if self.in_table and tag in {"td", "th"}:
            self.cell = []
        if (
            self.in_table
            and tag == "a"
            and self.cell is not None
            and not self.row
            and "nfl-o-roster__player-name" in attrs.get("class", "").split()
        ):
            self.links.append(attrs.get("href", ""))

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(_text(self.cell))
            self.cell = None
        if tag == "tr" and self.in_table:
            if self.row == _HEADERS:
                self.valid_table = True
            elif self.valid_table and len(self.row) == len(_HEADERS):
                self.records.append(
                    {"name": self.row[0], "position": self.row[2], "status": self.row[3]}
                    | {"profile_links": self.links.copy()}
                )
        if tag == "table":
            self.in_table = False
            self.valid_table = False


class _ProfileParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.script = None
        self.documents = []
        self.invalid_jsonld = False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self.script = []

    def handle_data(self, data):
        if self.script is not None:
            self.script.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            try:
                self.documents.append(json.loads("".join(self.script)))
            except ValueError:
                # Discarding malformed data could conceal a competing identity
                # alongside another, valid SportsTeam record.
                self.invalid_jsonld = True
            self.script = None


def _sports_teams(value):
    if isinstance(value, list):
        return [item for entry in value for item in _sports_teams(entry)]
    if isinstance(value, dict):
        if value.get("@type") == "SportsTeam":
            return [value]
        return _sports_teams(value.get("@graph", []))
    return []


def _read_html(url: str) -> str:
    with urllib.request.urlopen(url, timeout=25) as response:
        final_url = urlsplit(response.url)
        if final_url.scheme != "https" or final_url.netloc != "www.nfl.com":
            raise ValueError("NFL roster source redirected outside its official host")
        return response.read().decode("utf-8")


@dataclass(frozen=True)
class _Page:
    html: str
    sha256: str
    retrieved_at: str


class OfficialRosterVerifier:
    """One build's lazy source cache; discarded before the next refresh.

    NFL roster pages confirm a live season, not a historical weekly snapshot.
    ``effective_period`` records the caller's requested-week identity join; the
    separately recorded source season and retrieval time retain that distinction.
    """

    def __init__(self, season: int, week: int):
        self.season = int(season)
        self.week = int(week)
        self._pages: dict[str, _Page | None] = {}
        self._directory = None
        self._directory_loaded = False
        self.errors: list[str] = []

    def _page(self, url: str) -> _Page | None:
        if url not in self._pages:
            try:
                html = _read_html(url)
                self._pages[url] = _Page(
                    html, hashlib.sha256(html.encode()).hexdigest(), datetime.now(UTC).isoformat()
                )
            except Exception as exc:  # real network boundary; callers retain unresolved gaps
                self.errors.append(f"{url}: {exc!r}")
                self._pages[url] = None
        return self._pages[url]

    def _team_name(self, team: str) -> str | None:
        if not self._directory_loaded:
            self._directory_loaded = True
            try:
                self._directory = nfl_source.teams()
            except Exception as exc:
                self.errors.append(f"NFL team directory: {exc!r}")
        frame = self._directory
        if frame is None or not {"team_abbr", "team_name"}.issubset(frame):
            return None
        # The directory also contains historical SD/OAK/STL rows. They must not
        # replace the modern name used in the NFL URL and its season title.
        current = frame.loc[~frame["team_abbr"].isin(schedule_team_code_normalization())]
        names = set(current.loc[current["team_abbr"].map(_team_code).eq(team), "team_name"])
        return next(iter(names)) if len(names) == 1 else None

    def verify_inactive_identity(
        self,
        team: str,
        position: str,
        name: str,
        reference: pd.DataFrame,
        *,
        espn_player: dict | None = None,
    ) -> dict | None:
        """Return attributed inactive evidence after a scoped name/DOB join.

        Missing, ambiguous and conflicting identities stay unresolved. ESPN
        corroboration, when supplied, must agree with every available weekly ID.
        """
        try:
            return self._verify(team, position, name, reference, espn_player)
        except Exception as exc:  # malformed upstream schemas cannot establish eligibility
            self.errors.append(f"{team} {position} {name}: {exc!r}")
            return None

    def _verify(self, team, position, name, reference, espn_player):
        if not isinstance(name, str) or not name.strip():
            return None
        team = _team_code(team)
        position = "K" if position == "PK" else position
        current = current_rosters(reference, self.season, self.week)
        if current.empty or "birth_date" not in current:
            return None
        candidates = []
        for row in current.loc[
            current["team"].map(_team_code).eq(team) & current["position"].eq(position)
        ].to_dict("records"):
            names = {
                name_key(row[column])
                for column in ("full_name", "player_name", "player_display_name")
                if pd.notna(row.get(column))
            }
            if name_key(name) in names:
                candidates.append(row)
        identities = {player_id(row["player_id"]) for row in candidates}
        birthdays = {_birthday(row.get("birth_date")) for row in candidates}
        if len(identities) != 1 or len(birthdays) != 1 or None in birthdays:
            return None
        pid, birthday = next(iter(identities)), next(iter(birthdays))
        reference_espn = {
            _espn_id(row.get("espn_id"))
            for row in candidates
            if player_id(row.get("espn_id")) is not None
        }
        if None in reference_espn or len(reference_espn) > 1:
            return None
        if espn_player is not None:
            incoming = _espn_id(espn_player.get("espn_id"))
            if (
                name_key(espn_player.get("espn_name")) != name_key(name)
                or _birthday(espn_player.get("roster_birth_date")) != birthday
                or _team_code(espn_player.get("recent_team")) != team
                or espn_player.get("position") != position
                or incoming is None
                or (reference_espn and reference_espn != {incoming})
            ):
                return None
        team_name = self._team_name(team)
        if not isinstance(team_name, str) or not team_name:
            return None
        slug = re.sub(r"[^a-z0-9]+", "-", team_name.lower()).strip("-")
        roster_url = f"https://www.nfl.com/teams/{slug}/roster"
        roster_page = self._page(roster_url)
        if roster_page is None:
            return None
        roster = _RosterParser()
        roster.feed(roster_page.html)
        if _text(roster.title_parts) != f"{team_name} {self.season} Player Roster | NFL.com":
            return None
        rows = [
            row
            for row in roster.records
            if name_key(row["name"]) == name_key(name) and row["position"] == position
        ]
        if len(rows) != 1 or rows[0]["status"] not in _INACTIVE:
            return None
        row = rows[0]
        if len(row["profile_links"]) != 1:
            return None
        profile_url = urljoin(roster_url, row["profile_links"][0])
        parts = urlsplit(profile_url)
        if (
            parts.scheme != "https"
            or parts.netloc != "www.nfl.com"
            or not re.fullmatch(r"/players/[a-z0-9-]+/?", parts.path)
            or parts.query
            or parts.fragment
        ):
            return None
        profile_page = self._page(profile_url)
        if profile_page is None:
            return None
        profile = _ProfileParser()
        profile.feed(profile_page.html)
        if profile.invalid_jsonld or profile.script is not None:
            return None
        teams = _sports_teams(profile.documents)
        if len(teams) != 1:
            return None
        role = teams[0].get("member") or {}
        person = role.get("member") or {}
        if (
            teams[0].get("name") != team_name
            or role.get("@type") != "OrganizationRole"
            or role.get("roleName") != position
            or person.get("@type") != "Person"
            or name_key(person.get("name")) != name_key(name)
            or _birthday(person.get("birthDate")) != birthday
        ):
            return None
        evidence = {
            "player_id": pid,
            "team": team,
            "position": position,
            "name": row["name"],
            "status": row["status"],
            "provider": "NFL.com official team roster and player profile",
            "roster_url": roster_url,
            "profile_url": profile_url,
            "retrieved_at": max(roster_page.retrieved_at, profile_page.retrieved_at),
            "effective_period": {"season": self.season, "week": self.week},
            "source_season": self.season,
            "roster_sha256": roster_page.sha256,
            "profile_sha256": profile_page.sha256,
            "value_kind": "official_roster_inactive_identity",
        }
        evidence["content_id"] = content_identity(
            {key: value for key, value in evidence.items() if key != "retrieved_at"}
        )
        return evidence
