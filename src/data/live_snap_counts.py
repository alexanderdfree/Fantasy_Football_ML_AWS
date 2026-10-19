"""Recover missing live-season offensive participation from public Sleeper stats.

Only existing, completed player-games can receive an observed snap share. The
historical archive and native observations retain precedence. Source game IDs
are opaque: date and matchup bind them to the NFL schedule, rather than an
assumed equivalence between providers' identifiers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from numbers import Real
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd

from src.config import SEASONS
from src.data import nfl_source
from src.data.identity import normalize_team_code

_BASE_URL = "https://api.sleeper.com/stats/nfl"
_USER_AGENT = "Final-Project/1.0"
_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
_MAX_BYTES = 10 * 1024 * 1024
_TIMEOUT_SECONDS = 15
_CURRENT_USAGE = ("attempts", "carries", "targets", "receptions", "sacks")
_SOURCE_USAGE = ("pass_att", "rush_att", "rec_tgt", "rec", "pass_sack")


def _source_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.sleeper.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Sleeper response must remain on its public HTTPS API host")


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _source_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ResponseError(ValueError):
    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Sleeper JSON contains duplicate object fields")
        result[key] = value
    return result


def _get_json(url: str) -> tuple[object, dict]:
    """Fetch with bounded transport retries and a receipt for exact source bytes."""
    _source_url(url)
    opener = build_opener(_SafeRedirect())
    for attempt in range(1, 3):
        try:
            with opener.open(
                Request(url, headers={"Accept": "application/json", "User-Agent": _USER_AGENT}),
                timeout=_TIMEOUT_SECONDS,
            ) as response:
                final_url = response.geturl()
                _source_url(final_url)
                body = response.read(_MAX_BYTES + 1)
            receipt = {
                "url": url,
                "response_url": final_url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "attempts": attempt,
                "user_agent": _USER_AGENT,
            }
            if len(body) > _MAX_BYTES:
                receipt["truncated"] = True
                raise _ResponseError("Sleeper response exceeds the size limit", receipt)
            try:
                payload = json.loads(body, object_pairs_hook=_unique_object)
            except (ValueError, UnicodeError) as exc:
                raise _ResponseError("Sleeper response is not unambiguous JSON", receipt) from exc
            return payload, receipt
        except HTTPError as exc:
            if attempt == 2 or (exc.code != 429 and not 500 <= exc.code < 600):
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    raise RuntimeError("Sleeper request did not complete")  # pragma: no cover


def _integer(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        numeric = float(value)
    except (ValueError, OverflowError):
        return None
    return int(numeric) if math.isfinite(numeric) and numeric.is_integer() else None


def _period(value: object) -> int | None:
    if isinstance(value, str) and value.isdecimal():
        try:
            return int(value)
        except ValueError:
            return None
    return _integer(value)


def _identifier(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "nan", "nat", "<na>", "null"}:
            return None
        try:
            return str(int(text)) if text.isdecimal() else text
        except ValueError:
            return None
    number = _integer(value)
    return str(number) if number is not None else None


def _team(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = normalize_team_code(value.strip().upper())
    return "LA" if normalized == "LAR" else normalized


def _position(value: object) -> str:
    text = value.strip().upper() if isinstance(value, str) else ""
    return "RB" if text == "FB" else text


def _day(value: object) -> str | None:
    if isinstance(value, (date, datetime, pd.Timestamp)) and not pd.isna(value):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value).isoformat()
        return parsed if parsed == value else None
    except ValueError:
        return None


def _completed_games(schedules: pd.DataFrame, season: int) -> dict:
    required = {
        "season",
        "week",
        "game_type",
        "gameday",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    if not required.issubset(schedules) or not schedules.columns.is_unique:
        raise ValueError("NFL schedule lacks required completed-game identity fields")
    contexts = defaultdict(set)
    ids = defaultdict(set)
    now = pd.Timestamp.now("UTC")
    for row in schedules.to_dict("records"):
        if (
            _period(row["season"]) != season
            or not isinstance(row["game_type"], str)
            or row["game_type"] != "REG"
        ):
            continue
        week, day = _period(row["week"]), _day(row["gameday"])
        home, away, game_id = (
            _team(row["home_team"]),
            _team(row["away_team"]),
            _identifier(row["game_id"]),
        )
        scores = [_integer(row[column]) for column in ("home_score", "away_score")]
        if (
            week is None
            or not 1 <= week <= 18
            or not day
            or day > now.date().isoformat()
            or not home
            or not away
            or home == away
            or not game_id
            or any(score is None or score < 0 for score in scores)
        ):
            continue
        game_time = row.get("gametime")
        if isinstance(game_time, str) and game_time:
            try:
                kickoff = pd.Timestamp(f"{day} {game_time}").tz_localize("America/New_York")
            except (ValueError, TypeError):
                continue
            if kickoff > now:
                continue
        context = (game_id, day, home, away)
        ids[game_id].add((week, day, home, away))
        for team in (home, away):
            contexts[(week, team)].add(context)
    return {
        key: next(iter(values))
        for key, values in contexts.items()
        if len(values) == 1 and len(ids[next(iter(values))[0]]) == 1
    }


def _id_bridge(ids: pd.DataFrame) -> dict[str, str]:
    if not {"sleeper_id", "gsis_id"}.issubset(ids):
        raise ValueError("Player crosswalk lacks Sleeper/GSIS identity fields")
    pairs = {
        (sid, gid)
        for sid, gid in (
            (_identifier(s), _identifier(g))
            for s, g in ids[["sleeper_id", "gsis_id"]].itertuples(index=False, name=None)
        )
        if sid and gid
    }
    sources, targets = defaultdict(set), defaultdict(set)
    for sid, gid in pairs:
        sources[sid].add(gid)
        targets[gid].add(sid)
    return {sid: gid for sid, gid in pairs if len(sources[sid]) == 1 and len(targets[gid]) == 1}


def _usage_floor(row: dict, columns: tuple[str, ...]) -> float:
    """One observed counter can bound snaps; overlapping counters cannot be summed."""
    floor = 0.0
    for column in columns:
        value = row.get(column)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            continue
        try:
            number = float(value)
        except (ValueError, OverflowError):
            return math.inf
        if math.isfinite(number) and number > floor:
            floor = number
    return floor


def _source_observation(source: dict, season: int, week: int, games: dict) -> dict | None:
    """Validate actual game evidence independently of our modeled player cohort."""
    team = _team(source.get("team"))
    context = games.get((week, team))
    player, stats = source.get("player"), source.get("stats")
    sid, source_id = _identifier(source.get("player_id")), _identifier(source.get("game_id"))
    if (
        source.get("category") != "stat"
        or source.get("sport") != "nfl"
        or _period(source.get("season")) != season
        or _period(source.get("week")) != week
        or source.get("season_type") != "regular"
        or context is None
        or _day(source.get("date")) != context[1]
        or _team(source.get("opponent")) != (context[3] if team == context[2] else context[2])
        or not sid
        or not source_id
        or not isinstance(player, dict)
        or not _position(player.get("position"))
        or not isinstance(stats, dict)
    ):
        return None
    count, total = _integer(stats.get("off_snp")), _integer(stats.get("tm_off_snp"))
    if (
        _integer(stats.get("gp")) != 1
        or count is None
        or total is None
        or total <= 0
        or not 0 <= count <= total
        or count < _usage_floor(stats, _SOURCE_USAGE)
    ):
        return None
    return {
        "week": week,
        "sid": sid,
        "count": count,
        "total": total,
        "game": context[0],
        "source_game": source_id,
        "team": team,
        "company": source.get("company"),
        "position": _position(player.get("position")),
        "roles": player.get("fantasy_positions"),
    }


def recover_snap_counts(
    current: pd.DataFrame, schedules: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, dict]:
    """Fill only missing offensive shares for uniquely bound completed live games."""
    result = current.copy()
    metadata = {
        "source": "Sleeper public NFL actual stats",
        "url": _BASE_URL,
        "season": season,
        "retrieved_at": None,
        "eligible_rows": 0,
        "observed_before_rows": 0,
        "observed_rows": 0,
        "missing_before_rows": 0,
        "missing_rows": 0,
        "recovered_rows": 0,
        "recovered_cells": 0,
        "rejected_rows": 0,
        "needed_weeks": [],
        "requested_weeks": [],
        "validated_games": [],
        "validated_source_game_ids": [],
        "receipts": [],
        "errors": [],
    }

    def finish() -> tuple[pd.DataFrame, dict]:
        total, observed = metadata["eligible_rows"], metadata["observed_rows"]
        metadata["status"] = (
            "not_required"
            if not total and not metadata["errors"]
            else "available"
            if total and observed == total
            else "partial"
            if observed
            else "unavailable"
        )
        return result, metadata

    if season <= max(SEASONS):
        metadata["reason"] = "historical_replay"
        return finish()
    required = {"player_id", "season", "week", "position", "recent_team", "opponent_team"}
    if not required.issubset(result) or not result.columns.is_unique:
        metadata["errors"].append(
            {"week": None, "error": "Current player-game schema is incomplete"}
        )
        return finish()
    try:
        games = _completed_games(schedules, season)
    except ValueError as exc:
        metadata["errors"].append({"week": None, "error": str(exc)})
        return finish()
    if "snap_pct" not in result:
        result["snap_pct"] = np.nan
    columns = list(required | {"snap_pct", "season_type", "game_id", *_CURRENT_USAGE})
    rows = result[[column for column in columns if column in result]].to_dict("records")
    eligible = {}
    for index, row in enumerate(rows):
        pid, week, position = (
            _identifier(row["player_id"]),
            _period(row["week"]),
            row["position"].strip().upper() if isinstance(row["position"], str) else "",
        )
        if (
            not pid
            or _period(row["season"]) != season
            or position not in _POSITIONS
            or not isinstance(row.get("season_type", "REG"), str)
            or row.get("season_type", "REG") != "REG"
        ):
            continue
        context = games.get((week, _team(row["recent_team"])))
        if context is None:
            continue
        game_id, day, home, away = context
        team, opponent = _team(row["recent_team"]), _team(row["opponent_team"])
        if opponent != (away if team == home else home):
            continue
        if "game_id" in row and _identifier(row["game_id"]) != game_id:
            continue
        eligible[index] = {
            **row,
            "_pid": pid,
            "_week": week,
            "_position": position,
            "_team": team,
            "_opponent": opponent,
            "_context": context,
        }
    metadata["eligible_rows"] = len(eligible)
    observed = sum(pd.notna(row["snap_pct"]) for row in eligible.values())
    metadata["observed_before_rows"] = metadata["observed_rows"] = int(observed)
    needed = {index for index, row in eligible.items() if pd.isna(row["snap_pct"])}
    metadata["missing_before_rows"] = metadata["missing_rows"] = len(needed)
    metadata["needed_weeks"] = sorted({eligible[index]["_week"] for index in needed})
    if not needed:
        return finish()
    key_counts = Counter((row["_week"], row["_pid"]) for row in eligible.values())
    targets = {
        (row["_week"], row["_pid"]): index
        for index, row in eligible.items()
        if key_counts[(row["_week"], row["_pid"])] == 1
    }
    if not needed.intersection(targets.values()):
        metadata["errors"].append(
            {"week": None, "error": "Current player-week identities are ambiguous"}
        )
        return finish()
    try:
        bridge = _id_bridge(nfl_source.player_ids())
    except Exception as exc:  # optional provider/identity boundary
        metadata["errors"].append({"week": None, "error": str(exc)[:300]})
        return finish()
    payloads = {}
    metadata["requested_weeks"] = metadata["needed_weeks"].copy()
    with ThreadPoolExecutor(max_workers=min(4, len(metadata["requested_weeks"]))) as pool:
        futures = {
            pool.submit(_get_json, f"{_BASE_URL}/{season}/{week}?season_type=regular"): week
            for week in metadata["requested_weeks"]
        }
        for future in as_completed(futures):
            week = futures[future]
            try:
                payload, receipt = future.result()
                metadata["receipts"].append({**receipt, "week": week})
                if not isinstance(payload, list) or any(
                    not isinstance(row, dict) for row in payload
                ):
                    raise ValueError("Sleeper weekly stats must be a list of player records")
                payloads[week] = payload
            except Exception as exc:  # optional HTTP/schema boundary
                if isinstance(exc, _ResponseError):
                    metadata["receipts"].append({**exc.receipt, "week": week})
                metadata["errors"].append({"week": week, "error": str(exc)[:300]})
    metadata["receipts"].sort(key=lambda receipt: receipt["week"])
    if metadata["receipts"]:
        metadata["retrieved_at"] = max(r["retrieved_at"] for r in metadata["receipts"])
    observations = []
    invalid_targets = set()
    for week, payload in sorted(payloads.items()):
        for source in payload:
            sid = _identifier(source.get("player_id"))
            target = targets.get((week, bridge.get(sid)))
            observation = _source_observation(source, season, week, games)
            if observation is None:
                if target is not None:
                    metadata["rejected_rows"] += 1
                    invalid_targets.add(target)
                continue
            if target is not None:
                row, roles = eligible[target], observation["roles"]
                if (
                    observation["game"] != row["_context"][0]
                    or observation["team"] != row["_team"]
                    or observation["count"] < _usage_floor(row, _CURRENT_USAGE)
                    or not (
                        observation["position"] == row["_position"]
                        or isinstance(roles, list)
                        and row["_position"] in {_position(role) for role in roles}
                    )
                ):
                    metadata["rejected_rows"] += 1
                    invalid_targets.add(target)
                    target = None
            observations.append({**observation, "target": target})
    source_games, nfl_games, totals, player_values = (defaultdict(set) for _ in range(4))
    for observation in observations:
        source_games[observation["source_game"]].add(observation["game"])
        nfl_games[observation["game"]].add(observation["source_game"])
        totals[(observation["game"], observation["team"])].add(observation["total"])
        player_values[(observation["week"], observation["sid"])].add(
            (observation["source_game"], observation["count"], observation["total"])
        )
    validated_games, source_ids, companies = set(), set(), set()
    for observation in observations:
        target = observation["target"]
        if (
            target in invalid_targets
            or len(source_games[observation["source_game"]]) != 1
            or len(nfl_games[observation["game"]]) != 1
            or len(totals[(observation["game"], observation["team"])]) != 1
            or len(player_values[(observation["week"], observation["sid"])]) != 1
        ):
            metadata["rejected_rows"] += 1
            continue
        if target not in needed:
            continue
        # Native PFR shares use ties-to-even decimal hundredths. Binary-float
        # round(42 / 80, 2), for example, incorrectly produces 0.53 rather than 0.52.
        with localcontext() as decimal_context:
            decimal_context.prec = max(28, len(str(observation["total"])) + 4)
            share = (Decimal(observation["count"]) / Decimal(observation["total"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_EVEN
            )
        result.iat[target, result.columns.get_loc("snap_pct")] = float(share)
        needed.remove(target)
        metadata["recovered_rows"] += 1
        validated_games.add(observation["game"])
        source_ids.add(observation["source_game"])
        if isinstance(observation["company"], str):
            companies.add(observation["company"])
    metadata["recovered_cells"] = metadata["recovered_rows"]
    metadata["observed_rows"] += metadata["recovered_rows"]
    metadata["missing_rows"] -= metadata["recovered_rows"]
    metadata["validated_games"] = sorted(validated_games)
    metadata["validated_source_game_ids"] = sorted(source_ids)
    metadata["companies"] = sorted(companies)
    metadata["errors"].sort(key=lambda error: -1 if error["week"] is None else error["week"])
    return finish()
