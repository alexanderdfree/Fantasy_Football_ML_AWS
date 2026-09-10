"""Conservative player identity joins for historical participation records."""

from __future__ import annotations

import os

import pandas as pd

from src.data import nfl_source
from src.data.cache_io import atomic_write_parquet
from src.data.nflcom_loader import normalize_player_name, schedule_team_code_normalization

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
