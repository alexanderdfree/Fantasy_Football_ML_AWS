"""Conservative, season-bound identity fallbacks for live roster consumers."""

from __future__ import annotations

import unicodedata

import pandas as pd

from src.data.identity import normalize_player_name, schedule_team_code_normalization


def player_id(value) -> str | None:
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan", "null", "<na>"}:
        return None
    return str(value).strip()


def name_key(value) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return normalize_player_name(value)


def current_rosters(rosters: pd.DataFrame | None, season: int | None, week: int | None):
    """Only the requested week's identities may establish a live fallback."""
    required = {"player_id", "season", "week", "team", "position"}
    if rosters is None or not required.issubset(rosters) or season is None or week is None:
        return pd.DataFrame()
    current = rosters.loc[
        pd.to_numeric(rosters["season"], errors="coerce").eq(season)
        & pd.to_numeric(rosters["week"], errors="coerce").eq(week)
    ].copy()
    current["player_id"] = current["player_id"].map(player_id)
    current = current.loc[current["player_id"].notna()]
    current["team"] = current["team"].replace(schedule_team_code_normalization())
    current["position"] = current["position"].replace({"PK": "K"})
    return current


def _names(row: dict) -> set[str]:
    return {
        name_key(row[column])
        for column in ("full_name", "player_name", "player_display_name")
        if pd.notna(row.get(column)) and name_key(row[column])
    }


def _birth_date(value):
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp.date() if pd.notna(timestamp) else None


def resolve_player(player: dict, reference: pd.DataFrame) -> tuple[str | None, str]:
    """Match a missing crosswalk entry by scoped name AND date of birth.

    A unique name alone is insufficient. ESPN IDs, when present in the weekly
    reference, must agree too. Conflicting IDs or eligibility remain explicit.
    """
    if reference.empty:
        return None, "identity_reference_unavailable"
    group = reference.loc[
        reference["team"].eq(player["recent_team"]) & reference["position"].eq(player["position"])
    ]
    birthday = _birth_date(player.get("roster_birth_date"))
    candidates = []
    for row in group.to_dict("records"):
        if name_key(player["espn_name"]) not in _names(row):
            continue
        if birthday is None or _birth_date(row.get("birth_date")) != birthday:
            continue
        espn = pd.to_numeric(row.get("espn_id"), errors="coerce")
        if pd.notna(espn) and (espn % 1 or str(int(espn)) != player["espn_id"]):
            return None, "conflicting_espn_identity"
        candidates.append(row)
    identities = {row["player_id"] for row in candidates}
    if len(identities) != 1:
        return None, "ambiguous_identity" if identities else "unmatched_identity"
    pid = next(iter(identities))
    # An exempt/reserve player can remain in ESPN's offense group. A fallback
    # requires corroborated active eligibility; don't silently add that player.
    statuses = {str(row.get("status", "UNKNOWN")) for row in candidates}
    if statuses != {"ACT"}:
        return None, "eligibility_unconfirmed:" + ",".join(sorted(statuses))
    return pid, "weekly_roster_name_dob"


def practice_alias_lookup(roster: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Join official name variants through GSIS, preserving every ambiguity."""
    lookup: dict[tuple, set[str]] = {}
    contexts = set()
    for row in roster.to_dict("records"):
        pid = player_id(row["player_id"])
        if pid is None:
            continue
        context = (row["recent_team"], row["position"])
        contexts.add((pid, *context))
        lookup.setdefault((*context, name_key(row["espn_name"])), set()).add(pid)
    for row in reference.to_dict("records"):
        if (row["player_id"], row["team"], row["position"]) not in contexts:
            continue
        for name in _names(row):
            key = (row["team"], row["position"], name)
            lookup.setdefault(key, set()).add(row["player_id"])
    return lookup
