"""Shared NFL team and player identity normalization.

The roster/NFL.com and schedule/weekly join universes deliberately retain
separate Rams aliases; this module owns that existing distinction.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet

_SUFFIX_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Historical -> canonical NFL team-abbr mapping. Both NFL.com and nflverse have
# historically inconsistent codes; canonicalize one side so the join works.
#
# This is the **canonical project-wide team-code normalization base map** — the
# single source of truth for relocation/abbr aliases. It targets the
# *NFL.com/roster* join universe, which canonicalizes the Rams to ``"LAR"``
# (``import_seasonal_rosters`` and NFL.com projection CSVs). Callers should
# import ``TEAM_CODE_MAP`` / call ``normalize_team_code(code)`` rather than
# hand-maintaining a parallel dictionary.
#
# Join-universe caveat (do NOT "fix" by collapsing the two): the nflverse
# *schedule* + *weekly* releases use ``"LA"`` for the Rams (verified across
# 2016-2025: ``import_schedules`` and ``import_weekly_data`` both emit ``LA``,
# never ``LAR``). A merge that maps the schedule's ``LA`` to ``LAR`` while the
# player frame still carries ``LA`` silently misses every Rams row. The
# schedule-join consumers therefore derive their normalization from this base
# via ``schedule_team_code_normalization()`` below, which remaps the Rams back
# to ``LA`` and drops the historical ``STL`` to ``LA`` (not ``LAR``). That
# helper is the *one* place the schedule-universe variant is defined;
# ``src/shared/weather_features.TEAM_CODE_NORMALIZATION`` is built from it.
TEAM_CODE_MAP: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
    "JAX": "JAX",
    "JAC": "JAX",
    "LA": "LAR",
}


def normalize_team_code(code: str | None) -> str:
    """Map a historical NFL team code to its current canonical abbreviation.

    Canonical project-wide helper for team-code normalization — importable
    from ``src.data.identity`` by any bundle that needs to align
    franchise codes across data sources (NFL.com projections, nflverse
    schedules/PBP, internal stats). Examples:

        >>> normalize_team_code("OAK")
        'LV'
        >>> normalize_team_code("STL")
        'LAR'
        >>> normalize_team_code("@OAK")  # NFL.com prefixes opponent for away games
        'LV'
        >>> normalize_team_code(None)
        ''

    Parameters
    ----------
    code : str | None
        The raw team code. ``None`` / NaN / empty string returns ``""``.
        A leading ``"@"`` (NFL.com away-game prefix) is stripped.
        Unknown codes pass through unchanged (after upper-case + strip).

    Returns
    -------
    str
        The canonical NFL team abbreviation, or ``""`` for missing input.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return ""
    s = str(code).strip().upper()
    # NFL.com sometimes prefixes opponent with '@' for away games — strip.
    s = s.lstrip("@")
    return TEAM_CODE_MAP.get(s, s)


def schedule_team_code_normalization() -> dict[str, str]:
    """Return the relocation map for the *nflverse schedule/weekly* join universe.

    Derived from :data:`TEAM_CODE_MAP` (the single source of truth) with the
    one documented join-direction difference: nflverse schedules and weekly
    data canonicalize the Rams to ``"LA"`` (not ``"LAR"``), so the historical
    ``STL`` maps to ``LA`` and the modern ``LA`` is left untouched (no
    ``LA -> LAR`` rewrite, which would break the Rams join against player rows
    that already carry ``LA``). The ``WSH/JAX/JAC`` entries are dropped because
    nflverse already emits ``WAS``/``JAX`` consistently — only the three
    relocated franchises ever differ between the schedule's historical codes
    and the player frame's modern codes.

    Consumed by ``src.shared.weather_features.TEAM_CODE_NORMALIZATION`` so the
    schedule-side normalization has exactly one definition.
    """
    base = {k: v for k, v in TEAM_CODE_MAP.items() if k in ("OAK", "SD", "STL")}
    base["STL"] = "LA"  # nflverse schedule/weekly uses LA for the Rams, not LAR.
    return base


def normalize_player_name(name: str | None) -> str:
    """Canonicalize a player name for cross-source joining.

    - Lowercase, strip leading/trailing whitespace.
    - Drop trailing suffix tokens (Jr, Sr, II, III, IV, V).
    - Drop punctuation entirely (apostrophes, periods, hyphens collapse to "").
    - Collapse internal whitespace.

    Examples:
        "Patrick Mahomes II"   -> "patrick mahomes"
        "Marvin Harrison Jr."  -> "marvin harrison"
        "Ja'Marr Chase"        -> "jamarr chase"
        "A.J. Brown"           -> "aj brown"
        "Foo  Bar"             -> "foo bar"
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip().lower()
    if not s:
        return ""
    # Drop punctuation. Keep whitespace and ascii letters/digits.
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    # Tokenize, drop trailing suffix tokens (only at the end; "II Smith" stays).
    tokens = s.split()
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


_MISSING_IDS = {"", "none", "nan", "null", "<na>"}
_TEAM_CODES = {
    **schedule_team_code_normalization(),
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LA",
}
# The Falcons' Nathan Carter roster URL displays his NFL name, Nate Carter:
# https://www.atlantafalcons.com/team/players-roster/nathan-carter/situational
# Scope this documented alias to his GSIS identity and roster team-season.
_VERIFIED_ALIASES = {"00-0040547": ("Nathan Carter",)}


def valid_player_ids(values: pd.Series) -> pd.Series:
    """Do not let pandas match missing identifiers to other missing identifiers."""
    return values.notna() & ~values.astype(str).str.strip().str.lower().isin(_MISSING_IDS)


def load_player_id_bridge(cache_dir: str) -> pd.DataFrame:
    """Cache the crosswalk with the training inputs instead of fetching it per run."""
    from src.data.release import assert_source_fetch_allowed

    path = os.path.join(cache_dir, "player_id_bridge_v2.parquet")
    if os.path.exists(path):
        cached = pd.read_parquet(path)
        if {"pfr_id", "gsis_id"}.issubset(cached):
            return cached
    assert_source_fetch_allowed(path)
    ids = nfl_source.player_ids()
    columns = [c for c in ("pfr_id", "gsis_id", "espn_id") if c in ids]
    result = ids[columns].copy()
    atomic_write_parquet(result, path, index=False)
    return result


def load_player_metadata(cache_dir: str) -> pd.DataFrame:
    """Persist the name-variant dependency used only for unresolved identities."""
    from src.data.release import assert_source_fetch_allowed

    path = os.path.join(cache_dir, "player_metadata_v1.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    assert_source_fetch_allowed(path)
    source = nfl_source.player_metadata()
    columns = [
        c
        for c in (
            "gsis_id",
            "pfr_id",
            "display_name",
            "common_first_name",
            "first_name",
            "last_name",
            "football_name",
            "espn_id",
        )
        if c in source
    ]
    result = source[columns].copy()
    atomic_write_parquet(result, path, index=False)
    return result


def bridge_snap_counts(
    snaps: pd.DataFrame,
    ids: pd.DataFrame,
    rosters: pd.DataFrame,
    weekly: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bridge PFR IDs, then unique roster IDs and exact team-season names.

    Name matching is an exact normalized match scoped to team and season,
    never a fuzzy/global-name guess. Ambiguous fallback keys stay unresolved.
    Existing authoritative crosswalk matches always win.
    """
    out = snaps.reset_index(drop=True).copy()
    out["gsis_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if {"pfr_id", "gsis_id"}.issubset(ids):
        primary = ids.loc[
            valid_player_ids(ids["pfr_id"]) & valid_player_ids(ids["gsis_id"]),
            ["pfr_id", "gsis_id"],
        ].sort_values(["pfr_id", "gsis_id"])
        primary = primary.drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
        out["gsis_id"] = out["pfr_player_id"].map(primary).astype("string")

    def fill_unique(reference: pd.DataFrame, keys: list[str]) -> None:
        if reference.empty or not {*keys, "gsis_id"}.issubset(reference):
            return
        ref = reference.loc[valid_player_ids(reference["gsis_id"]), [*keys, "gsis_id"]]
        ref = ref.dropna(subset=keys).drop_duplicates()
        ref = ref.loc[ref.groupby(keys)["gsis_id"].transform("nunique").eq(1)]
        ref = ref.drop_duplicates(keys)
        missing = out["gsis_id"].isna()
        query = out.loc[missing, keys].copy()
        query["_row"] = query.index
        found = query.merge(ref, on=keys, how="left", validate="many_to_one").set_index("_row")
        out.loc[found.index, "gsis_id"] = found["gsis_id"].astype("string")

    if {"pfr_id", "season", "player_id"}.issubset(rosters):
        # rosters can already carry gsis_id alongside its player_id alias.
        ref = rosters[["pfr_id", "season", "player_id"]].rename(
            columns={"pfr_id": "pfr_player_id", "player_id": "gsis_id"}
        )
        ref = ref.loc[valid_player_ids(ref["pfr_player_id"])]
        fill_unique(ref, ["pfr_player_id", "season"])

    if {"pfr_id", "espn_id"}.issubset(ids) and {"espn_id", "season", "player_id"}.issubset(rosters):
        crosswalk = ids[["pfr_id", "espn_id"]].copy()
        roster_ids = rosters[["espn_id", "season", "player_id"]].copy()
        for source in (crosswalk, roster_ids):
            source["espn_id"] = pd.to_numeric(source["espn_id"], errors="coerce")
        crosswalk = crosswalk.dropna(subset=["espn_id"])
        roster_ids = roster_ids.dropna(subset=["espn_id"])
        ref = crosswalk.merge(roster_ids, on="espn_id").rename(
            columns={"pfr_id": "pfr_player_id", "player_id": "gsis_id"}
        )
        ref = ref.loc[valid_player_ids(ref["pfr_player_id"])]
        fill_unique(ref, ["pfr_player_id", "season"])

    if {"player", "team", "season"}.issubset(out):
        out["_name"] = out["player"].fillna("").map(normalize_player_name)
        out["_team"] = out["team"].replace(_TEAM_CODES)
        candidates = []
        for source, team_col, names in (
            (rosters, "team", ("full_name", "player_name")),
            (weekly, "recent_team", ("player_display_name",)),
        ):
            if not {"player_id", "season", team_col}.issubset(source):
                continue
            for name in names:
                if name not in source:
                    continue
                ref = source[["player_id", "season", team_col, name]].copy()
                ref["_name"] = ref[name].fillna("").map(normalize_player_name)
                ref["_team"] = ref[team_col].replace(_TEAM_CODES)
                ref = ref.rename(columns={"player_id": "gsis_id"})
                candidates.append(ref[["gsis_id", "season", "_team", "_name"]])
        if metadata is not None and {"gsis_id", "last_name"}.issubset(metadata):
            aliases = []
            for column in ("display_name", "first_name", "common_first_name", "football_name"):
                if column not in metadata:
                    continue
                values = metadata[column].fillna("")
                if column != "display_name":
                    values = values + " " + metadata["last_name"].fillna("")
                aliases.append(
                    pd.DataFrame(
                        {"gsis_id": metadata["gsis_id"], "_name": values.map(normalize_player_name)}
                    )
                )
            aliases.append(
                pd.DataFrame(
                    [
                        {"gsis_id": pid, "_name": normalize_player_name(name)}
                        for pid, names in _VERIFIED_ALIASES.items()
                        for name in names
                    ]
                )
            )
            if {"player_id", "season", "team"}.issubset(rosters):
                contexts = (
                    rosters[["player_id", "season", "team"]]
                    .rename(columns={"player_id": "gsis_id", "team": "_team"})
                    .drop_duplicates()
                )
                contexts["_team"] = contexts["_team"].replace(_TEAM_CODES)
                ref = contexts.merge(pd.concat(aliases, ignore_index=True), on="gsis_id")
                candidates.append(ref)
        if candidates:
            # Partial metadata must not erase a competing roster identity.
            # Establish uniqueness over every alias source together.
            ref = pd.concat(candidates, ignore_index=True)
            ref = ref.loc[ref["_name"].ne("")]
            fill_unique(ref, ["season", "_team", "_name"])
        out.drop(columns=["_name", "_team"], inplace=True)
    return out
