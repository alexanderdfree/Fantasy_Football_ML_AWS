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
    "no upcoming data". The ONE deliberate exception is
    :func:`next_unplayed_week`, which raises :class:`EspnUnreachableError`
    when ESPN can't be reached at all: the CI artifact builder must
    distinguish a *verified* offseason (ESPN answered: nothing scheduled)
    from an outage — in 2026-08 a scoreboard 403 block read as "offseason",
    the builder exited green without uploading, and serving froze on a
    20-day-old artifact.

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
# Depth charts live on the separate "core" host (the site API doesn't expose
# them); fetched through the same _get_json timeout/retry treatment.
_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
# ESPN's fantasy API (weekly point projections) lives on a third host; the
# public "leaguedefaults/3" league is ESPN's stock PPR league, whose
# kona_player_info view carries every player's weekly projection.
_FANTASY_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
_TIMEOUT_S = 15
_RETRIES = 3
_RETRY_BACKOFF_S = 1.5
_MAX_REG_WEEK = 18
# Abort the next_unplayed_week scan after this many consecutive failed
# scoreboard probes (each probe already carries _get_json's own _RETRIES): a
# hard outage/WAF block fails every week identically, so grinding through all
# 2x18 weekly probes only wastes ~3 min of CI and hammers a blocking host.
_CONSECUTIVE_FAILURE_ABORT = 3


class EspnUnreachableError(RuntimeError):
    """ESPN could not be reached at all (distinct from a verified empty slate).

    Raised by :func:`next_unplayed_week` when scoreboard probes error and no
    scheduled week was found — callers must NOT treat that as "offseason".
    """


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

# ESPN item-level status -> game_status numeric, matching the TRAINING encoding
# in src/data/loader.py (status_map): Questionable 0.5 / Doubtful 0.1 / Out 0.0.
# Worst-per-player is taken by fetch_injury_status_map; Active/unknown statuses
# are omitted so those rows keep build_features' healthy default (1.0).
_GAME_STATUS_NUM = {
    "out": 0.0,
    "injured reserve": 0.0,
    "ir": 0.0,
    "doubtful": 0.1,
    "questionable": 0.5,
}

# nflverse depth_chart_rank is capped at 3 (the training distribution is
# {-1, 1, 2, 3} for every skill position), so ESPN's per-position depth ORDER is
# clamped to [1, 3] to stay on the scale the models trained on.
_MAX_DEPTH_RANK = 3
# ESPN depthchart position keys -> our skill position. Some teams split WR into
# lwr/rwr/swr slots; those pool into WR (so order is depth within the WR corps,
# mirroring nflverse, which can carry >1 rank-1 WR via alignment).
_DEPTHCHART_POS_KEYS = {"qb": "QB", "rb": "RB", "hb": "RB", "wr": "WR", "te": "TE"}

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
def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    """GET ``url`` and parse JSON, with timeout + retries. Raises on final fail.

    Deliberately sends NO custom ``User-Agent`` (urllib's honest default,
    ``Python-urllib/3.x``, goes out instead). ESPN's WAF started 403-ing both
    the old custom UA (``ff-predictor/1.0``) and spoofed-browser UAs from
    non-browser clients in 2026-08 — every scheduled CI build silently served
    a frozen artifact for 20 days — while honest tool UAs (``Python-urllib``,
    ``curl``, ``python-requests``) pass. Don't "improve" this back to a
    custom or browser UA. ``headers`` is for endpoint-specific extras (the
    fantasy API's ``X-Fantasy-Filter``) and must never smuggle a User-Agent in.
    """
    last_err: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers or {})
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


def _norm_depth_pos(key: str | None) -> str | None:
    """Map an ESPN depthchart position key to our skill position (or ``None``).

    Handles the consolidated ``wr`` key and split ``lwr``/``rwr``/``swr`` slots
    (all -> WR), plus ``hb`` -> RB.
    """
    k = (key or "").lower()
    if k in _DEPTHCHART_POS_KEYS:
        return _DEPTHCHART_POS_KEYS[k]
    if k.endswith("wr"):
        return "WR"
    return None


def _parse_depthchart(payload: dict) -> list[dict]:
    """Normalize an ESPN core ``/depthcharts`` payload into ordered skill entries:
    ``[{espn_id, position, order}]`` where ``order`` is 1-based depth WITHIN that
    position (1 = starter).

    ESPN's raw per-athlete ``rank`` is grid-numbered (WRs come back 1, 4, 7, 10),
    so we RE-RANK by sorted order rather than trusting the raw value, pooling
    split WR slots into one WR list. The caller clamps to the nflverse [1, 3]
    scale and maps espn_id -> gsis.
    """
    pooled: dict[str, list[tuple[float, str]]] = {}
    for group in payload.get("items", []) or []:
        for key, pos_obj in (group.get("positions") or {}).items():
            pos = _norm_depth_pos(key)
            if pos is None:
                continue
            for a in pos_obj.get("athletes") or []:
                ref = (a.get("athlete") or {}).get("$ref") or ""
                m = re.search(r"/athletes/(\d+)", ref)
                if not m:
                    continue
                rank = a.get("rank")
                pooled.setdefault(pos, []).append(
                    (float(rank) if rank is not None else float("inf"), m.group(1))
                )
    out: list[dict] = []
    for pos, lst in pooled.items():
        for order, (_, espn_id) in enumerate(sorted(lst, key=lambda t: t[0]), start=1):
            out.append({"espn_id": espn_id, "position": pos, "order": order})
    return out


# --------------------------------------------------------------------------
# Public fetchers (defensive — never raise into the poller)
# --------------------------------------------------------------------------
def fetch_games(season: int, week: int, *, raise_on_error: bool = False) -> list[dict]:
    """Fetch + parse the scoreboard for one (season, week).

    Default: ``[]`` on failure (the legacy defensive contract). With
    ``raise_on_error=True`` a network/parse failure raises instead, so callers
    can distinguish "ESPN answered: no games" from "ESPN unreachable".
    """
    try:
        return _parse_scoreboard_games(_get_json(_scoreboard_url(season, week)))
    except Exception as e:  # noqa: BLE001 - network boundary
        if raise_on_error:
            raise
        print(f"[espn_live] fetch_games({season}, {week}) failed: {e!r}")
        return []


def next_unplayed_week(season: int, lookahead_seasons: int = 1) -> tuple[int, int] | None:
    """Auto-detect the next unplayed REG week.

    Returns ``(season, week)`` for the smallest week that still has a
    ``STATUS_SCHEDULED`` game, rolling into ``season + 1`` if the requested
    season is fully played, up to ``lookahead_seasons`` ahead. Returns ``None``
    only for a VERIFIED offseason: every scoreboard probe answered and none had
    a scheduled game (true offseason / schedule not yet posted).

    Raises :class:`EspnUnreachableError` when probes errored and no scheduled
    week was found — either early (``_CONSECUTIVE_FAILURE_ABORT`` consecutive
    failures = a hard outage/WAF block) or at scan end (any failures at all:
    an unverifiable week can't be ruled out, so "offseason" would be a guess).
    The 2026-08 incident this guards: a scoreboard 403 block made every probe
    return empty, the CI builder wrote "offseason" and exited green, and the
    served artifact silently froze for 20 days.
    """
    failures = 0
    consecutive = 0
    last_err: Exception | None = None
    for s in range(season, season + lookahead_seasons + 1):
        for w in range(1, _MAX_REG_WEEK + 1):
            try:
                games = fetch_games(s, w, raise_on_error=True)
            except Exception as e:  # noqa: BLE001 - network boundary
                failures += 1
                consecutive += 1
                last_err = e
                print(f"[espn_live] fetch_games({s}, {w}) failed: {e!r}")
                if consecutive >= _CONSECUTIVE_FAILURE_ABORT:
                    raise EspnUnreachableError(
                        f"aborting week scan at {s} W{w}: {consecutive} consecutive "
                        f"scoreboard failures (last: {e!r})"
                    ) from e
                continue
            consecutive = 0
            if games and any(g["is_scheduled"] for g in games):
                return (s, w)
    if failures:
        raise EspnUnreachableError(
            f"no scheduled week found and {failures} scoreboard probe(s) failed "
            f"(last: {last_err!r}); cannot distinguish offseason from an ESPN outage"
        ) from last_err
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
                    # spread_line is home-perspective from ESPN; negate for the
                    # away team so each row carries its own team's spread, matching
                    # the per-team convention in weather_features.py / dst/data.py.
                    # Odds can be off-board (spread_line=None from the parser) —
                    # keep None rather than TypeError-ing the whole slate (#1400).
                    "spread_line": g["spread_line"]
                    if side == "home" or g["spread_line"] is None
                    else -g["spread_line"],
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


def fetch_depth_chart_ranks(season: int, team_id_to_code: dict[str, str]) -> dict[str, float]:
    """``{player_id: depth_chart_rank}`` from ESPN's live depth charts.

    For each team, GET the core ``/depthcharts`` endpoint, re-rank each skill
    position by order (1 = starter), clamp to nflverse's [1, 3] scale, and map
    espn_id -> gsis. Falls back to the prior season's chart per team when the
    requested season isn't posted yet (offseason). ``{}`` on total failure —
    callers then keep the carry-forward / default behavior.
    """
    crosswalk = espn_to_gsis_map()
    ranks: dict[str, float] = {}
    for team_id in team_id_to_code:
        if not team_id:
            continue
        entries: list[dict] = []
        for yr in (season, season - 1):
            try:
                payload = _get_json(f"{_CORE_BASE}/seasons/{yr}/teams/{team_id}/depthcharts")
            except Exception as e:  # noqa: BLE001 - network boundary
                print(f"[espn_live] depthchart fetch failed for team {team_id} ({yr}): {e!r}")
                continue
            entries = _parse_depthchart(payload)
            if entries:
                break
        for e in entries:
            gsis = crosswalk.get(e["espn_id"])
            if not gsis:
                continue
            ranks[gsis] = float(min(e["order"], _MAX_DEPTH_RANK))
    return ranks


# --------------------------------------------------------------------------
# Fantasy point projections (the "ESPN" expert column on the upcoming-week tab)
# --------------------------------------------------------------------------
# ESPN fantasy defaultPositionId -> skill position (5=K, 16=DST excluded —
# the upcoming-week artifact is QB/RB/WR/TE only).
_FANTASY_POS_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
# kona_player_info stat-entry discriminators: statSourceId 1 = projection
# (0 = actual); statSplitTypeId 1 = single-week split (0 = full season).
_STAT_SOURCE_PROJECTION = 1
_STAT_SPLIT_WEEKLY = 1
# ESPN stat id for projected receptions inside a stat entry's raw ``stats``
# dict. Verified empirically (2026-08-24): leaguedefaults/3 (PPR) appliedTotal
# minus leaguedefaults/1 (standard) appliedTotal equals this stat exactly for
# every sampled pass-catcher, and the two are equal for QBs — the formats
# differ only by the reception weight, same as src/config.py's SCORING_* dicts.
_STAT_ID_RECEPTIONS = "53"
# X-Fantasy-Filter player cap: comfortably above the ~485 players ESPN projects
# for a week and the ~775-player active skill roster set.
_FANTASY_FILTER_LIMIT = 1500


def _fantasy_projections_url(season: int, week: int) -> str:
    return (
        f"{_FANTASY_BASE}/seasons/{season}/segments/0/leaguedefaults/3"
        f"?view=kona_player_info&scoringPeriodId={week}"
    )


def _parse_fantasy_projections(payload: dict, season: int, week: int) -> list[dict]:
    """Normalize a ``kona_player_info`` payload into per-player projection dicts.

    Emits ``{espn_id, position, ppr_total, receptions}`` for skill players
    carrying a weekly projection entry for (season, week) with a nonzero
    ``appliedTotal``. ESPN stamps 0.00 placeholders on deep-bench players —
    those are "no genuine projection" (the roster-placeholder lesson), so they
    are dropped and serialize as null, matching the other experts' semantics.
    """
    out: list[dict] = []
    for entry in payload.get("players") or []:
        pl = entry.get("player") or {}
        pos = _FANTASY_POS_MAP.get(pl.get("defaultPositionId"))
        espn_id = _norm_espn_id(pl.get("id"))
        if not pos or not espn_id:
            continue
        for s in pl.get("stats") or []:
            if (
                s.get("seasonId") == season
                and s.get("scoringPeriodId") == week
                and s.get("statSourceId") == _STAT_SOURCE_PROJECTION
                and s.get("statSplitTypeId") == _STAT_SPLIT_WEEKLY
            ):
                total = s.get("appliedTotal")
                if not total:  # None or the 0.00 deep-bench placeholder
                    break
                raw = s.get("stats") or {}
                try:
                    receptions = float(raw.get(_STAT_ID_RECEPTIONS) or 0.0)
                except (TypeError, ValueError):
                    receptions = 0.0
                out.append(
                    {
                        "espn_id": espn_id,
                        "position": pos,
                        "ppr_total": float(total),
                        "receptions": receptions,
                    }
                )
                break
    return out


def fetch_fantasy_projections(season: int, week: int) -> pd.DataFrame:
    """ESPN weekly fantasy point projections for the skill positions.

    One GET of the public stock-PPR league (``leaguedefaults/3``) with an
    ``X-Fantasy-Filter`` limit covering every fantasy-relevant player; espn_id
    -> gsis via the roster crosswalk (unmapped rows dropped, count logged).
    Columns: ``player_id, position, season, week, espn_ppr_total,
    espn_receptions``. PPR points are ESPN's own ``appliedTotal``;
    standard/half-PPR derive exactly by removing the reception weight (see
    ``_STAT_ID_RECEPTIONS``). Empty frame on any failure — expert data is
    optional, so the builder degrades to a null column exactly like the
    NFL.com/RotoWire boundaries.
    """
    try:
        flt = json.dumps(
            {
                "players": {
                    "limit": _FANTASY_FILTER_LIMIT,
                    "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
                }
            }
        )
        payload = _get_json(
            _fantasy_projections_url(season, week), headers={"X-Fantasy-Filter": flt}
        )
        records = _parse_fantasy_projections(payload, season, week)
    except Exception as e:  # noqa: BLE001 - network boundary
        print(f"[espn_live] fantasy projections fetch failed: {e!r}")
        return pd.DataFrame()
    crosswalk = espn_to_gsis_map()
    rows: list[dict] = []
    unmapped = 0
    for r in records:
        gsis = crosswalk.get(r["espn_id"])
        if not gsis:
            unmapped += 1
            continue
        rows.append(
            {
                "player_id": gsis,
                "position": r["position"],
                "season": season,
                "week": week,
                "espn_ppr_total": r["ppr_total"],
                "espn_receptions": r["receptions"],
            }
        )
    if unmapped:
        print(f"[espn_live] dropped {unmapped} fantasy-projection rows with no gsis mapping")
    return pd.DataFrame(rows)


def fetch_injury_status_map(season: int, week: int) -> dict[str, float]:
    """``{player_id: game_status}`` from ESPN injuries, on the training encoding.

    Maps ESPN status -> the same numeric scale as src/data/loader.py's
    ``status_map`` (Questionable 0.5 / Doubtful 0.1 / Out 0.0), worst-per-player.
    Active/unknown statuses are omitted so those rows keep build_features' healthy
    default (1.0). ESPN exposes no practice participation, so ``practice_status``
    is deliberately NOT filled here. ``{}`` on failure.
    """
    crosswalk = espn_to_gsis_map()
    out: dict[str, float] = {}
    try:
        records = _parse_injuries(_get_json(f"{_ESPN_BASE}/injuries"))
    except Exception as e:  # noqa: BLE001 - network boundary
        print(f"[espn_live] injury status fetch failed: {e!r}")
        return {}
    for rec in records:
        num = _GAME_STATUS_NUM.get((rec.get("status") or "").lower())
        if num is None:
            continue
        gsis = crosswalk.get(rec["espn_id"]) if rec.get("espn_id") else None
        if not gsis:
            continue
        out[gsis] = min(num, out.get(gsis, 1.0))
    return out
