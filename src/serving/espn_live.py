"""Live NFL data from ESPN's public (unofficial) API, for the upcoming-week
predictions homepage.

Historical stats stay on nflverse (stable, 10+ years, cached in ``data/raw``);
only the **forward-looking** inputs — schedule, Vegas lines, injuries, active
rosters — come from here, so they're fresh and not gated on nflverse's release
cadence (its schedules carry only the *closing* line post-hoc, and roster /
injury feeds lag).

ESPN's endpoints are undocumented and can change without notice, so:
  * every network call goes through ``_get_json`` (timeout + a few retries);
  * parsing is isolated in pure ``_parse_*`` functions that are fixture-tested,
    so a schema drift fails *this* module with a clear error instead of
    silently zero-filling a feature;
  * the public ``fetch_*`` helpers never raise into the poller — on a hard
    failure they return an empty result and the caller treats the cycle as
    "no upcoming data".

Player identity is bridged ESPN ``athlete.id`` -> ``espn_id`` -> ``gsis_id``
(our ``player_id``) via the stable nflverse crosswalk
(:func:`src.data.nfl_source.player_ids`).
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request

import pandas as pd

from src.data import nfl_source

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
_TIMEOUT_S = 15
_RETRIES = 3
_RETRY_BACKOFF_S = 1.5
_MAX_REG_WEEK = 18

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
_SKILL_POSITION_SET = set(SKILL_POSITIONS)

# ESPN team abbreviation -> nflverse code. Verified against our data's
# ``recent_team`` set: only the Rams (LAR->LA) and Washington (WSH->WAS) differ;
# the other 30 abbreviations match exactly.
_ESPN_TEAM_CODE_MAP = {"LAR": "LA", "WSH": "WAS"}

# ESPN roster groups whose players are NOT active for the upcoming game.
_INACTIVE_ROSTER_GROUPS = {"injuredReserveOrOut", "suspended", "practiceSquad"}

# ESPN item-level injury status -> the ``report_status`` build_features'
# inheritance feature expects. Only ``Out``/``Doubtful`` count; everything else
# (Active/Questionable) is ignored. Compared case-insensitively.
_INJURY_STATUS_MAP = {
    "out": "Out",
    "injured reserve": "Out",
    "ir": "Out",
    "suspension": "Out",
    "doubtful": "Doubtful",
}

# Module-cached espn_id -> gsis_id crosswalk (built once; nflverse is stable).
_espn_to_gsis: dict[str, str] | None = None
_crosswalk_lock = threading.Lock()


# --------------------------------------------------------------------------
# Mapping helpers
# --------------------------------------------------------------------------
def espn_to_nflverse_team(abbr: str | None) -> str | None:
    """Map an ESPN team abbreviation to our nflverse ``recent_team`` code."""
    if abbr is None:
        return None
    return _ESPN_TEAM_CODE_MAP.get(abbr, abbr)


def _norm_espn_id(value) -> str | None:
    """Normalize an ESPN athlete id to a bare string of digits.

    ESPN serves ids as strings (``"4678006"``); the nflverse crosswalk stores
    ``espn_id`` as a float/int/str depending on season. Coerce both to a plain
    integer-string so the lookup keys line up (``4678006.0`` -> ``"4678006"``).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s or None


def _extract_espn_id(athlete: dict) -> str | None:
    """Pull the ESPN athlete id from a headshot/link href.

    The injuries endpoint's ``athlete`` object omits a bare ``id`` field, but the
    id appears in the headshot URL (``.../full/2578570.png``) and the player link
    (``.../id/2578570/...``).
    """
    href = (athlete.get("headshot") or {}).get("href") or ""
    m = re.search(r"/(\d+)\.png", href)
    if m:
        return m.group(1)
    for link in athlete.get("links") or []:
        m = re.search(r"/id/(\d+)", link.get("href") or "")
        if m:
            return m.group(1)
    return None


def espn_to_gsis_map() -> dict[str, str]:
    """Build (once) the ESPN-athlete-id -> nflverse ``player_id`` (gsis) map.

    Sourced from :func:`src.data.nfl_source.player_ids` (the ff_playerids
    crosswalk). Cached module-level — the crosswalk is stable, so a single
    build serves every poll. On a fetch failure returns ``{}`` (callers then
    drop all players, logging the count) rather than raising.
    """
    global _espn_to_gsis
    if _espn_to_gsis is not None:
        return _espn_to_gsis
    with _crosswalk_lock:
        if _espn_to_gsis is not None:
            return _espn_to_gsis
        mapping: dict[str, str] = {}
        try:
            ids = nfl_source.player_ids()
            for espn_id, gsis_id in zip(
                ids.get("espn_id", []), ids.get("gsis_id", []), strict=False
            ):
                key = _norm_espn_id(espn_id)
                gsis = None if gsis_id is None else str(gsis_id).strip()
                if key and gsis and gsis.lower() not in ("nan", "none", "<na>"):
                    mapping[key] = gsis
        except Exception as e:  # noqa: BLE001 - network/data boundary
            print(f"[espn_live] player-id crosswalk build failed: {e!r}")
        _espn_to_gsis = mapping
        return _espn_to_gsis


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
def _get_json(url: str) -> dict:
    """GET ``url`` and parse JSON, with timeout + retries. Raises on final fail."""
    last_err: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ff-predictor/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            last_err = e
            if attempt < _RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"ESPN GET failed after {_RETRIES} tries: {url} ({last_err!r})")


def _scoreboard_url(season: int, week: int) -> str:
    return f"{_ESPN_BASE}/scoreboard?dates={season}&seasontype=2&week={week}"


# --------------------------------------------------------------------------
# Pure parsers (fixture-tested, no network)
# --------------------------------------------------------------------------
def _parse_scoreboard_games(payload: dict) -> list[dict]:
    """Normalize a scoreboard payload into per-game dicts.

    Team codes are mapped to nflverse space and ``spread_line`` is converted to
    nflverse convention (**positive = home favored**; ESPN's ``odds.spread`` is
    the opposite). ``home_score``/``away_score`` are left ``None`` for unplayed
    games.
    """
    season = (payload.get("season") or {}).get("year")
    week = (payload.get("week") or {}).get("number")
    games: list[dict] = []
    for ev in payload.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        status = ((comp.get("status") or {}).get("type") or {}).get("name", "")
        odds_list = comp.get("odds") or []
        odds = odds_list[0] if odds_list else {}
        espn_spread = odds.get("spread")
        total_line = odds.get("overUnder")
        games.append(
            {
                "game_id": ev.get("id"),
                "season": season,
                "week": week,
                "home_team": espn_to_nflverse_team(home["team"].get("abbreviation")),
                "away_team": espn_to_nflverse_team(away["team"].get("abbreviation")),
                "home_team_id": str(home["team"].get("id")) if home["team"].get("id") else None,
                "away_team_id": str(away["team"].get("id")) if away["team"].get("id") else None,
                "is_scheduled": status == "STATUS_SCHEDULED",
                # ESPN spread is negative = home favored; flip to nflverse sign.
                "spread_line": (-espn_spread if espn_spread is not None else None),
                "total_line": total_line,
            }
        )
    return games


def _parse_roster_players(payload: dict, team_code: str) -> list[dict]:
    """Normalize a team-roster payload into active skill-player dicts.

    Skips inactive groups (IR / suspended / practice squad). ``id`` is the ESPN
    athlete id (still a string); the gsis mapping is applied by the caller.
    """
    players: list[dict] = []
    for group in payload.get("athletes", []) or []:
        if group.get("position") in _INACTIVE_ROSTER_GROUPS:
            continue
        for item in group.get("items", []) or []:
            pos = (item.get("position") or {}).get("abbreviation")
            if pos not in _SKILL_POSITION_SET:
                continue
            espn_id = _norm_espn_id(item.get("id"))
            if not espn_id:
                continue
            players.append(
                {
                    "espn_id": espn_id,
                    "espn_name": item.get("displayName"),
                    "position": pos,
                    "recent_team": team_code,
                }
            )
    return players


def _parse_injuries(payload: dict) -> list[dict]:
    """Normalize the league injuries payload into per-player status dicts.

    The athlete object carries ``position`` + ``team`` directly but omits a bare
    id, so it's pulled from the headshot/link href (:func:`_extract_espn_id`).
    """
    out: list[dict] = []
    for team_block in payload.get("injuries", []) or []:
        for item in team_block.get("injuries", []) or []:
            athlete = item.get("athlete") or {}
            team = athlete.get("team") or {}
            out.append(
                {
                    "espn_id": _extract_espn_id(athlete),
                    "espn_name": athlete.get("displayName"),
                    "position": (athlete.get("position") or {}).get("abbreviation"),
                    "team": espn_to_nflverse_team(team.get("abbreviation")),
                    "status": (item.get("status") or "").strip(),
                }
            )
    return out


# --------------------------------------------------------------------------
# Public fetchers (defensive — never raise into the poller)
# --------------------------------------------------------------------------
def fetch_games(season: int, week: int) -> list[dict]:
    """Fetch + parse the scoreboard for one (season, week). ``[]`` on failure."""
    try:
        return _parse_scoreboard_games(_get_json(_scoreboard_url(season, week)))
    except Exception as e:  # noqa: BLE001 - network boundary
        print(f"[espn_live] fetch_games({season}, {week}) failed: {e!r}")
        return []


def next_unplayed_week(season: int, lookahead_seasons: int = 1) -> tuple[int, int] | None:
    """Auto-detect the next unplayed REG week.

    Returns ``(season, week)`` for the smallest week that still has a
    ``STATUS_SCHEDULED`` game, rolling into ``season + 1`` if the requested
    season is fully played, up to ``lookahead_seasons`` ahead. Returns ``None``
    when nothing scheduled is found (true offseason / schedule not yet posted).
    """
    for s in range(season, season + lookahead_seasons + 1):
        for w in range(1, _MAX_REG_WEEK + 1):
            games = fetch_games(s, w)
            if not games:
                continue
            if any(g["is_scheduled"] for g in games):
                return (s, w)
    return None


def fetch_slate(season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(team_rows, schedule_rows)`` for one upcoming (season, week).

    * ``team_rows``: one row per team with ``recent_team``, ``opponent_team``,
      ``is_home``, ``spread_line``, ``total_line`` (for the synthetic player
      skeleton join).
    * ``schedule_rows``: nflverse-schema game rows (``game_id``, ``season``,
      ``week``, ``game_type='REG'``, ``home_team``, ``away_team``,
      ``home_score``/``away_score``=NA, ``spread_line``, ``total_line``) for the
      schedules-parquet augmentation (so ``implied_team_total`` computes).

    Both empty on failure.
    """
    games = fetch_games(season, week)
    team_rows: list[dict] = []
    sched_rows: list[dict] = []
    for g in games:
        sched_rows.append(
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "week": g["week"],
                "game_type": "REG",
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_score": pd.NA,
                "away_score": pd.NA,
                "spread_line": g["spread_line"],
                "total_line": g["total_line"],
            }
        )
        for side, team, opp in (
            ("home", g["home_team"], g["away_team"]),
            ("away", g["away_team"], g["home_team"]),
        ):
            team_rows.append(
                {
                    "recent_team": team,
                    "opponent_team": opp,
                    "is_home": 1 if side == "home" else 0,
                    "spread_line": g["spread_line"],
                    "total_line": g["total_line"],
                    "team_id": g["home_team_id"] if side == "home" else g["away_team_id"],
                }
            )
    return pd.DataFrame(team_rows), pd.DataFrame(sched_rows)


def fetch_active_rosters(team_id_to_code: dict[str, str]) -> pd.DataFrame:
    """Fetch active skill-position rosters for the given teams.

    ``team_id_to_code`` maps each ESPN team id to its nflverse ``recent_team``
    code (built from the slate). Returns a DataFrame with ``player_id`` (gsis),
    ``position``, ``recent_team`` — players whose ESPN id can't be mapped to a
    gsis id are dropped (count logged). Empty on total failure.
    """
    crosswalk = espn_to_gsis_map()
    rows: list[dict] = []
    unmapped = 0
    for team_id, team_code in team_id_to_code.items():
        if not team_id:
            continue
        try:
            payload = _get_json(f"{_ESPN_BASE}/teams/{team_id}/roster")
        except Exception as e:  # noqa: BLE001 - network boundary
            print(f"[espn_live] roster fetch failed for team {team_id}: {e!r}")
            continue
        for p in _parse_roster_players(payload, team_code=team_code):
            gsis = crosswalk.get(p["espn_id"])
            if not gsis:
                unmapped += 1
                continue
            rows.append(
                {
                    "player_id": gsis,
                    "position": p["position"],
                    "recent_team": team_code,
                    "espn_name": p["espn_name"],
                    "espn_id": p["espn_id"],
                }
            )
    if unmapped:
        print(f"[espn_live] dropped {unmapped} roster players with no gsis mapping")
    # A player can appear on two ESPN rosters mid-offseason churn; keep the last.
    if rows:
        return pd.DataFrame(rows).drop_duplicates(subset="player_id", keep="last")
    return pd.DataFrame(rows)


_INJURIES_COLUMNS = ["gsis_id", "position", "team", "season", "week", "report_status"]


def fetch_injuries_df(season: int, week: int) -> pd.DataFrame:
    """Return OUT/Doubtful players shaped for ``build_features``' role-inheritance
    feature: columns ``(gsis_id, position, team, season, week, report_status)``.

    ``gsis_id`` = nflverse ``player_id`` (via the ESPN id crosswalk); rows whose
    ESPN id can't be mapped, or whose status isn't Out/Doubtful, are dropped.
    Empty (with the right columns) on failure — the inheritance feature then
    degrades to 0 cleanly.
    """
    crosswalk = espn_to_gsis_map()
    rows: list[dict] = []
    try:
        records = _parse_injuries(_get_json(f"{_ESPN_BASE}/injuries"))
    except Exception as e:  # noqa: BLE001 - network boundary
        print(f"[espn_live] injuries fetch failed: {e!r}")
        return pd.DataFrame(columns=_INJURIES_COLUMNS)
    for rec in records:
        report = _INJURY_STATUS_MAP.get(rec["status"].lower())
        if report is None:
            continue
        gsis = crosswalk.get(rec["espn_id"]) if rec["espn_id"] else None
        if not gsis:
            continue
        rows.append(
            {
                "gsis_id": gsis,
                "position": rec["position"],
                "team": rec["team"],
                "season": season,
                "week": week,
                "report_status": report,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_INJURIES_COLUMNS)
    return pd.DataFrame(rows)
