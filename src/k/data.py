import os

import pandas as pd
import pyarrow.parquet as pq

from src.config import CACHE_DIR
from src.config import SEASONS as GLOBAL_SEASONS
from src.data import nfl_source
from src.k.config import POSITION_CONFIG
from src.shared.weather_features import TEAM_CODE_NORMALIZATION

SEASONS = POSITION_CONFIG.seasons
MIN_GAMES = POSITION_CONFIG.min_games

# Last season that belongs to the training split (val=2024, test=2025). Used
# both by ``season_split`` to carve the train rows and by ``load_data``'s
# Vegas-line imputation to fit the fill statistic on train rows only — keep
# the two in lockstep so the imputation never leaks val/test into the median.
_TRAIN_MAX_SEASON = 2023

# ---------------------------------------------------------------------------
# PBP-based kicker reconstruction (≤ 2024; lower bound comes from ``SEASONS``)
# ---------------------------------------------------------------------------

# Columns the current aggregation produces that downstream targets/features
# depend on. A cached parquet missing any of these was written by an older
# code version and must be regenerated rather than silently fed forward —
# `compute_targets` does `fillna(0)`, so a missing column turns into all-zero
# targets without an exception (cf. the Apr-19 cache that zeroed
# `fg_yard_points` for 2015-2024 and collapsed K projections to ~3 fpts).
_REQUIRED_PBP_COLUMNS = frozenset(
    {
        "player_id",
        "season",
        "week",
        "recent_team",
        "fg_att",
        "fg_made",
        "fg_missed",
        "fg_yards_made",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        # Missed-distance buckets and PBP-only situational aggregates produced
        # by ``reconstruct_kicker_weekly_from_pbp``. Listed explicitly so a
        # cache written before any one of these columns was added gets
        # regenerated rather than silently fed forward with the column
        # missing — same stale-cache trap that motivated the rest of this
        # allowlist (cf. the Apr-19 cache that zeroed ``fg_yard_points``).
        "fg_missed_40_49",
        "fg_missed_50_59",
        "fg_missed_60_",
        "q4_fg_att",
        "q4_fg_made",
        "long_fg_att",
        "long_fg_made",
        "pat_att",
        "pat_made",
        "pat_missed",
        "avg_fg_distance",
        "avg_fg_prob",
        # Sentinel for the XP-venue backfill that populates roof/surface for
        # XP-only kicker-weeks. Caches written before this fix don't carry the
        # column → schema check fails → cache rejected → regenerated with the
        # post-fix logic. Keeps train (cache) in lockstep with test (live
        # backfill in `_backfill_2025_pbp_columns` + `load_data` schedules
        # fallback) so the model doesn't see a different distribution at
        # train vs inference time.
        "_xp_venue_backfilled",
    }
)


def _cached_pbp_is_current(cache_path: str) -> bool:
    """Return True only if the cached parquet at ``cache_path`` has every
    column the current aggregation produces.

    Reads just the parquet schema (no row data) so the check is cheap. A
    False return signals the caller to ignore the cache and regenerate from
    PBP.
    """
    schema_names = set(pq.read_schema(cache_path).names)
    missing = _REQUIRED_PBP_COLUMNS - schema_names
    if missing:
        print(f"  Stale cache at {cache_path} missing {sorted(missing)}; regenerating")
        return False
    return True


def reconstruct_kicker_weekly_from_pbp(
    seasons: list[int],
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Reconstruct weekly kicker stats from play-by-play data.

    PBP has FG/XP play-level data from 1999+, while import_weekly_data()
    only has kicker columns starting in 2025. This function aggregates PBP
    plays into a weekly kicker-level dataframe matching the schema expected
    by src/k/targets.py, plus additional PBP-derived columns for features.
    """
    # Resolve at call time so module-level monkeypatches of CACHE_DIR take
    # effect (using `cache_dir: str = CACHE_DIR` as a default would freeze
    # the value at function-definition time).
    if cache_dir is None:
        cache_dir = CACHE_DIR
    if not seasons:
        # Empty-seasons guard (mirrors reconstruct_kicker_kicks_from_pbp) so the
        # seasons[0]/[-1] cache-path build below can't IndexError. (#409)
        return pd.DataFrame()
    cache_path = f"{cache_dir}/kicker_pbp_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(cache_path) and _cached_pbp_is_current(cache_path):
        return pd.read_parquet(cache_path)

    all_weekly = []
    skipped_seasons: list[int] = []
    for yr in seasons:
        print(f"  Loading PBP for {yr}...")
        # Wrap the entire per-year extraction so a 502 / empty frame / schema
        # change in one season doesn't abort the whole load. Mirrors the
        # defensive posture of reconstruct_kicker_kicks_from_pbp.
        try:
            pbp = nfl_source.pbp_data([yr], nfl_source.PBP_KICKER_COLS)
            # Keep only regular season
            pbp = pbp[pbp["season_type"] == "REG"]

            # --- Field goals ---
            fg = pbp[pbp["field_goal_attempt"] == 1].copy()
            fg["fg_made_flag"] = (fg["field_goal_result"] == "made").astype(int)
            fg["fg_missed_flag"] = (fg["field_goal_result"] != "made").astype(int)
            # Distance buckets matching weekly_data schema
            d = fg["kick_distance"]
            made = fg["fg_made_flag"].astype(bool)
            fg["fg_made_0_19"] = ((d < 20) & made).astype(int)
            fg["fg_made_20_29"] = ((d >= 20) & (d < 30) & made).astype(int)
            fg["fg_made_30_39"] = ((d >= 30) & (d < 40) & made).astype(int)
            fg["fg_made_40_49"] = ((d >= 40) & (d < 50) & made).astype(int)
            fg["fg_made_50_59"] = ((d >= 50) & (d < 60) & made).astype(int)
            fg["fg_made_60_"] = ((d >= 60) & made).astype(int)
            # Missed buckets
            missed = fg["fg_missed_flag"].astype(bool)
            fg["fg_missed_40_49"] = ((d >= 40) & (d < 50) & missed).astype(int)
            fg["fg_missed_50_59"] = ((d >= 50) & (d < 60) & missed).astype(int)
            fg["fg_missed_60_"] = ((d >= 60) & missed).astype(int)
            # PBP-only situational flags
            fg["is_q4"] = (fg["qtr"] >= 4).astype(int)
            fg["q4_made"] = (fg["is_q4"] & fg["fg_made_flag"]).astype(int)
            fg["is_long"] = (d >= 40).astype(int)
            fg["long_made"] = (fg["is_long"] & fg["fg_made_flag"]).astype(int)
            # Sum of kick_distance restricted to made FGs — per-attempt contribution
            # to the `fg_yards_made` season-week aggregate consumed by src.k.targets.
            fg["_fg_yards_made_flag"] = fg["fg_made_flag"] * fg["kick_distance"]

            weekly_fg = (
                fg.groupby(["kicker_player_id", "kicker_player_name", "posteam", "season", "week"])
                .agg(
                    fg_att=("fg_made_flag", "count"),
                    fg_made=("fg_made_flag", "sum"),
                    fg_missed=("fg_missed_flag", "sum"),
                    fg_made_0_19=("fg_made_0_19", "sum"),
                    fg_made_20_29=("fg_made_20_29", "sum"),
                    fg_made_30_39=("fg_made_30_39", "sum"),
                    fg_made_40_49=("fg_made_40_49", "sum"),
                    fg_made_50_59=("fg_made_50_59", "sum"),
                    fg_made_60_=("fg_made_60_", "sum"),
                    fg_missed_40_49=("fg_missed_40_49", "sum"),
                    fg_missed_50_59=("fg_missed_50_59", "sum"),
                    fg_missed_60_=("fg_missed_60_", "sum"),
                    fg_yards_made=("_fg_yards_made_flag", "sum"),
                    # PBP-derived aggregates
                    avg_fg_distance=("kick_distance", "mean"),
                    avg_fg_prob=("fg_prob", "mean"),
                    q4_fg_att=("is_q4", "sum"),
                    q4_fg_made=("q4_made", "sum"),
                    long_fg_att=("is_long", "sum"),
                    long_fg_made=("long_made", "sum"),
                    # Weather (game-level, same for all plays in a game)
                    game_wind=("wind", "first"),
                    game_temp=("temp", "first"),
                    roof=("roof", "first"),
                    surface=("surface", "first"),
                )
                .reset_index()
            )

            # --- Extra points ---
            # XP groupby pulls player_name/posteam/roof/surface alongside the
            # XP counts so XP-only kicker-weeks (no FG attempts) still have
            # those identity + venue fields populated after the outer join
            # below — the FG groupby is the only other source and yields no
            # row for XP-only games.
            xp = pbp[pbp["extra_point_attempt"] == 1].copy()
            xp["xp_made"] = (xp["extra_point_result"] == "good").astype(int)
            xp["xp_missed"] = (xp["extra_point_result"] != "good").astype(int)

            weekly_xp = (
                xp.groupby(["kicker_player_id", "season", "week"])
                .agg(
                    pat_att=("xp_made", "count"),
                    pat_made=("xp_made", "sum"),
                    pat_missed=("xp_missed", "sum"),
                    kicker_player_name_xp=("kicker_player_name", "first"),
                    posteam_xp=("posteam", "first"),
                    roof_xp=("roof", "first"),
                    surface_xp=("surface", "first"),
                )
                .reset_index()
            )

            # Merge FG + XP (outer so XP-only games survive).
            weekly_k = weekly_fg.merge(
                weekly_xp, on=["kicker_player_id", "season", "week"], how="outer"
            )

            # For XP-only games, FG columns are NaN — backfill identity + venue
            # from the XP-side copies, then drop the auxiliary columns.
            for col in ("kicker_player_name", "posteam", "roof", "surface"):
                weekly_k[col] = weekly_k[col].fillna(weekly_k[f"{col}_xp"])
                weekly_k.drop(columns=[f"{col}_xp"], inplace=True)

            # Sentinel column for cache-version detection. Stale caches written
            # before the XP-venue backfill landed don't have it, so the schema
            # check (`_REQUIRED_PBP_COLUMNS`) rejects them and forces regen —
            # otherwise the 589 historical XP-only games would keep their
            # `is_dome=0` / `roof=NaN` defaults and the train distribution
            # would diverge from the post-fix test backfill.
            weekly_k["_xp_venue_backfilled"] = True

            all_weekly.append(weekly_k)
        except Exception as e:
            print(f"  WARNING: PBP weekly extraction failed for {yr} ({e}); skipping")
            skipped_seasons.append(yr)
            continue

    if not all_weekly:
        # Every season failed (e.g. nflverse-wide outage). Return empty so the
        # caller can decide how to proceed; do NOT cache.
        return pd.DataFrame()

    result = pd.concat(all_weekly, ignore_index=True)

    # Fill NaN for kicker-weeks with only FGs or only XPs.
    # game_temp / game_wind are excluded: they're game-level weather, not
    # per-kick stats, and nfl_data_py reports NaN for ~100% of dome games.
    # The dome rewrite below sets them to (65, 0) explicitly; outdoor NaN is
    # left for the downstream fill_nans_with_train_means handler.
    for col in result.columns:
        if col not in [
            "kicker_player_id",
            "kicker_player_name",
            "posteam",
            "season",
            "week",
            "roof",
            "surface",
            "game_temp",
            "game_wind",
        ]:
            result[col] = result[col].fillna(0)

    # Rename to match expected pipeline columns
    result.rename(
        columns={
            "kicker_player_id": "player_id",
            "kicker_player_name": "player_name",
            "posteam": "recent_team",
        },
        inplace=True,
    )
    result["position"] = "K"
    result["season_type"] = "REG"

    # Derive venue features
    result["is_dome"] = result["roof"].isin(["dome", "closed"]).astype(int)
    # Mirror the canonical temp_adjusted/wind_adjusted dome handling in
    # src/shared/weather_features.py: dome games are 65 F / 0 mph regardless
    # of whether nfl_data_py recorded raw weather. Without this, the blanket
    # fillna above used to map dome NaN -> 0 F, which confounds dome games
    # with extreme-cold outdoor games.
    dome_mask = result["is_dome"] == 1
    result.loc[dome_mask, "game_temp"] = 65.0
    result.loc[dome_mask, "game_wind"] = 0.0

    if skipped_seasons:
        # Don't poison the combined cache key with a partial result — the next
        # call would treat it as authoritative for the full range and silently
        # serve incomplete data.
        print(f"  Skipped seasons {skipped_seasons}; not caching partial result to {cache_path}")
        return result

    os.makedirs(cache_dir, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"  Cached PBP kicker data: {len(result)} rows -> {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load kicker data combining PBP reconstruction (≤ 2024) + weekly (≥ 2025).

    The lower bound on the PBP arm comes from ``SEASONS`` (which starts at
    2015 per the post-PAT-rule-change cutoff).

    Merges schedule info for Vegas lines and home/away.
    """
    pbp_seasons = [s for s in SEASONS if s <= 2024]
    weekly_seasons = [s for s in SEASONS if s >= 2025]

    parts = []

    # --- PBP-reconstructed data (≤ 2024 lower-bounded by SEASONS) ---
    if pbp_seasons:
        print("Reconstructing kicker weekly stats from PBP...")
        pbp_df = reconstruct_kicker_weekly_from_pbp(pbp_seasons)
        parts.append(pbp_df)

    # --- Existing weekly data for 2025+ ---
    if weekly_seasons:
        weekly = pd.read_parquet(
            f"{CACHE_DIR}/weekly_{GLOBAL_SEASONS[0]}_{GLOBAL_SEASONS[-1]}.parquet"
        )
        k_weekly = weekly[
            (weekly["position"] == "K")
            & (weekly["season_type"] == "REG")
            & (weekly["season"].isin(weekly_seasons))
        ].copy()
        k_weekly = k_weekly[k_weekly["fg_att"].fillna(0) + k_weekly["pat_att"].fillna(0) > 0].copy()
        # Add PBP-derived columns with NaN (will be filled later)
        for col in [
            "avg_fg_distance",
            "avg_fg_prob",
            "q4_fg_att",
            "q4_fg_made",
            "long_fg_att",
            "long_fg_made",
            "game_wind",
            "game_temp",
            "roof",
            "surface",
            "is_dome",
            "fg_yards_made",
        ]:
            if col not in k_weekly.columns:
                k_weekly[col] = float("nan")
        parts.append(k_weekly)

    k_df = pd.concat(parts, ignore_index=True)

    # --- Backfill PBP-derived columns for 2025 from PBP ---
    if weekly_seasons:
        _backfill_2025_pbp_columns(k_df, weekly_seasons)

    # NOTE: the per-(player_id, season) ``MIN_GAMES`` filter is applied
    # **train-only** inside ``season_split`` (mirroring the shared pipeline's
    # train-only filter for other positions at
    # ``src/shared/pipeline.py``::``MIN_GAMES_PER_SEASON``). Applying it here
    # would drop val/test rows that the other positions keep — asymmetric and
    # underestimates real-world generalization.

    # --- Merge schedule info (is_home, Vegas lines, venue fallback) ---
    # roof/surface from the schedules parquet are folded into the same
    # home/away → recent_team reshape so we get them via one merge. PBP
    # already populates roof/surface for FG-attempt games; the schedules
    # values backfill the XP-only kicker-weeks (and 2025 games not yet
    # in the cached PBP). Schedules uses historical team codes (OAK/SD/STL)
    # while PBP normalises to current codes; map back so the join still hits
    # for pre-2017 rows.
    schedules = pd.read_parquet(
        f"{CACHE_DIR}/schedules_{GLOBAL_SEASONS[0]}_{GLOBAL_SEASONS[-1]}.parquet"
    )
    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()
    has_venue = {"roof", "surface"}.issubset(schedules_reg.columns)
    venue_cols = ["roof", "surface"] if has_venue else []
    base_cols = ["season", "week", "spread_line", "total_line"] + venue_cols

    home = schedules_reg[base_cols + ["home_team"]].rename(columns={"home_team": "recent_team"})
    home["is_home"] = 1
    home["implied_team_total"] = (home["total_line"] - home["spread_line"]) / 2

    away = schedules_reg[base_cols + ["away_team"]].rename(columns={"away_team": "recent_team"})
    away["is_home"] = 0
    away["implied_team_total"] = (away["total_line"] + away["spread_line"]) / 2

    schedule_info = pd.concat([home, away], ignore_index=True).drop(columns=["spread_line"])
    # Apply team-code normalization shared with weather_features so pre-relocation
    # OAK/SD/STL games still match modern LV/LAC/LAR rows. Hoisted out of the
    # ``if has_venue:`` branch — schedules use historical codes regardless of
    # whether roof/surface are present, so the schedule merge below
    # (``is_home`` + Vegas lines) misses for all pre-2017 OAK/SD/STL games
    # without this normalisation, not just the venue-fallback pass.
    schedule_info["recent_team"] = schedule_info["recent_team"].replace(TEAM_CODE_NORMALIZATION)
    # Suffix the venue cols so we don't overwrite PBP-populated values on rows
    # where PBP already has them — we only want to fill where PBP is NaN/"".
    if has_venue:
        schedule_info = schedule_info.rename(
            columns={"roof": "roof_sched", "surface": "surface_sched"}
        )

    k_df = k_df.merge(schedule_info, on=["recent_team", "season", "week"], how="left")

    # Fill missing Vegas lines with the TRAIN-only median, applied to every
    # split. Fitting the median on the full concatenated frame (train+val+test)
    # leaked the holdout Vegas-line distribution into the fill value the model
    # sees at train time — the same cross-fold leak the shared pipeline avoids
    # by imputing snap_pct on train rows only (``src/data/split.py``). The
    # per-(player_id, season) MIN_GAMES filter is applied later inside
    # ``season_split``, so the train mask here is the raw season cut; that's
    # fine — dropping a few low-game train rows from the median fit doesn't
    # reintroduce leakage. Falls back to the full-frame median only when there
    # are no train rows (synthetic fixtures), preserving prior behaviour there.
    train_mask = k_df["season"] <= _TRAIN_MAX_SEASON
    for col in ["total_line", "implied_team_total"]:
        train_vals = k_df.loc[train_mask, col]
        median_val = train_vals.median()
        if pd.isna(median_val):
            median_val = k_df[col].median()
        k_df[col] = k_df[col].fillna(median_val)
    if "is_home" not in k_df.columns or k_df["is_home"].isna().any():
        k_df["is_home"] = k_df["is_home"].fillna(0)

    # Fold roof/surface from schedules into the existing columns wherever the
    # PBP-derived value is NaN or "" — surface is occasionally an empty string
    # in PBP for games nflverse hasn't fully populated (e.g. 2025 KC@LAC wk 1).
    if has_venue:
        for col in ("roof", "surface"):
            sched_col = f"{col}_sched"
            if sched_col not in k_df.columns:
                continue
            if k_df[col].dtype != object:
                k_df[col] = k_df[col].astype(object)
            empty_or_nan = k_df[col].isna() | (k_df[col] == "")
            k_df.loc[empty_or_nan, col] = k_df.loc[empty_or_nan, sched_col]
        k_df.drop(columns=["roof_sched", "surface_sched"], inplace=True)
        # Re-derive is_dome where roof is now populated but is_dome was missed.
        if "is_dome" not in k_df.columns:
            k_df["is_dome"] = float("nan")
        needs_redrv = k_df["is_dome"].isna() & k_df["roof"].notna()
        k_df.loc[needs_redrv, "is_dome"] = (
            k_df.loc[needs_redrv, "roof"].isin(["dome", "closed"]).astype(int)
        )

    # Sentinel so the downstream shared ``merge_schedule_features`` (called by
    # both training pipeline and serving) becomes a no-op for K — K does its
    # own schedule merge with kicker-specific roof/surface fallback handling,
    # and the shared merge would otherwise re-populate is_dome/total_line/
    # implied_team_total with values that are already present (silent overwrite
    # of equivalent data, but wasted work + drift risk).
    k_df["_schedule_merged"] = True

    print(
        f"  Kicker data: {len(k_df)} rows, {k_df['player_id'].nunique()} kickers, "
        f"seasons {int(k_df['season'].min())}-{int(k_df['season'].max())}"
    )

    return k_df


def _backfill_2025_pbp_columns(k_df: pd.DataFrame, seasons: list[int]) -> None:
    """Backfill PBP-derived columns for 2025 rows from PBP data."""
    mask = k_df["season"].isin(seasons)
    if not mask.any():
        return

    backfill_cols = [
        "avg_fg_distance",
        "avg_fg_prob",
        "q4_fg_att",
        "q4_fg_made",
        "long_fg_att",
        "long_fg_made",
        "game_wind",
        "game_temp",
        "roof",
        "surface",
        "is_dome",
        "fg_yards_made",
    ]

    try:
        all_weekly = []
        all_game_venue = []
        for yr in seasons:
            pbp = nfl_source.pbp_data([yr], nfl_source.PBP_KICKER_COLS)
            pbp = pbp[pbp["season_type"] == "REG"]

            # Game-level venue/weather lookup keyed on (season, week, posteam).
            # Sourced from any PBP row (not just FGs) so kickers in XP-only
            # games still get roof/surface/wind/temp — fixes the 55 NaN test
            # rows the signal-floor diagnostic surfaced (see TODO archive).
            game_venue = (
                pbp.dropna(subset=["posteam"])
                .groupby(["season", "week", "posteam"])
                .agg(
                    game_wind=("wind", "first"),
                    game_temp=("temp", "first"),
                    roof=("roof", "first"),
                    surface=("surface", "first"),
                )
                .reset_index()
            )
            game_venue["is_dome"] = game_venue["roof"].isin(["dome", "closed"]).astype(int)
            # Mirror reconstruct_kicker_weekly_from_pbp: dome -> (65 F, 0 mph).
            dome_mask = game_venue["is_dome"] == 1
            game_venue.loc[dome_mask, "game_temp"] = 65.0
            game_venue.loc[dome_mask, "game_wind"] = 0.0
            all_game_venue.append(game_venue)

            fg = pbp[pbp["field_goal_attempt"] == 1].copy()
            d = fg["kick_distance"]
            fg["fg_made_flag"] = (fg["field_goal_result"] == "made").astype(int)
            fg["is_q4"] = (fg["qtr"] >= 4).astype(int)
            fg["q4_made"] = (fg["is_q4"] & fg["fg_made_flag"]).astype(int)
            fg["is_long"] = (d >= 40).astype(int)
            fg["long_made"] = (fg["is_long"] & fg["fg_made_flag"]).astype(int)
            # Sum of kick_distance restricted to made FGs — mirrors the
            # historical reconstruction so `fg_yard_points` target is
            # available for 2025 rows as well.
            fg["_fg_yards_made_flag"] = fg["fg_made_flag"] * fg["kick_distance"]

            weekly_pbp = (
                fg.groupby(["kicker_player_id", "season", "week"])
                .agg(
                    avg_fg_distance=("kick_distance", "mean"),
                    avg_fg_prob=("fg_prob", "mean"),
                    q4_fg_att=("is_q4", "sum"),
                    q4_fg_made=("q4_made", "sum"),
                    long_fg_att=("is_long", "sum"),
                    long_fg_made=("long_made", "sum"),
                    fg_yards_made=("_fg_yards_made_flag", "sum"),
                )
                .reset_index()
            )
            weekly_pbp.rename(columns={"kicker_player_id": "player_id"}, inplace=True)
            all_weekly.append(weekly_pbp)

        pbp_all = pd.concat(all_weekly, ignore_index=True)
        venue_all = pd.concat(all_game_venue, ignore_index=True)

        # roof/surface are initialized as float NaN upstream; cast to object so
        # DataFrame.update can write the string values pulled from PBP.
        for str_col in ("roof", "surface"):
            if k_df[str_col].dtype != object:
                k_df[str_col] = k_df[str_col].astype(object)

        # First pass: FG-derived columns keyed on (player_id, season, week).
        fg_backfill_cols = [
            c
            for c in backfill_cols
            if c not in ("game_wind", "game_temp", "roof", "surface", "is_dome")
        ]
        key = ["player_id", "season", "week"]
        # DataFrame.update aligns on index and overwrites non-NaN values from the source.
        # Wrap in try/finally so a failure inside update() can't leave k_df stuck
        # with the composite index (which would break downstream groupby calls).
        k_df.set_index(key, inplace=True)
        try:
            k_df.update(pbp_all.set_index(key)[fg_backfill_cols])
        finally:
            k_df.reset_index(inplace=True)

        # Second pass: game-level venue/weather keyed on (season, week,
        # recent_team). Independent of whether the kicker had FG attempts —
        # so XP-only games get populated too. Skip if k_df doesn't carry
        # recent_team (synthetic fixtures may omit it; production rows always
        # have it from the weekly parquet).
        if "recent_team" in k_df.columns:
            venue_cols = ["game_wind", "game_temp", "roof", "surface", "is_dome"]
            venue_lookup = venue_all.rename(columns={"posteam": "recent_team"})
            venue_key = ["season", "week", "recent_team"]
            k_df.set_index(venue_key, inplace=True)
            try:
                k_df.update(venue_lookup.set_index(venue_key)[venue_cols])
            finally:
                k_df.reset_index(inplace=True)
    except Exception as e:
        print(f"  WARNING: 2025 PBP backfill failed ({e}), PBP features will be NaN for 2025")
        # #815: a swallowed failure that leaves fg_yards_made all-NaN silently
        # zeros fg_yard_points for the WHOLE season (src.k.targets fillna(0)) —
        # indistinguishable from "made no FGs" (the ~3-fpt K collapse; cf. the
        # Apr-19 cache). Fail loud in that case. A venue-only failure that still
        # populated fg_yards_made (the FG first-pass succeeded) is tolerated.
        if "fg_yards_made" in k_df.columns and k_df.loc[mask, "fg_yards_made"].isna().all():
            raise RuntimeError(
                f"2025 PBP backfill failed and left fg_yards_made all-NaN for "
                f"seasons {seasons} — fg_yard_points would silently zero the whole "
                f"season. Check nfl_source PBP availability / schema."
            ) from e


_KICKS_SCHEMA = [
    "player_id",
    "season",
    "week",
    "play_id",
    "is_fg",
    "is_xp",
    "kick_distance",
    "kick_made",
    "fg_prob",
    "is_q4",
    "score_diff",
    "game_wind",
]

# Columns the current per-kick aggregation produces that downstream consumers
# rely on. ``play_id`` in particular gates the deterministic most-recent
# truncation in ``build_nested_kick_history`` — a cache written before that
# column was added would silently fall back to insertion-order sorting.
_REQUIRED_KICK_PBP_COLUMNS = frozenset(_KICKS_SCHEMA)


def _cached_kick_pbp_is_current(cache_path: str) -> bool:
    """Return True only if the cached per-kick parquet has every column the
    current aggregation produces.

    Mirrors ``_cached_pbp_is_current`` for the weekly cache: reads just the
    parquet schema (no row data) so the check is cheap. A False return signals
    the caller to ignore the cache and regenerate from PBP.
    """
    schema_names = set(pq.read_schema(cache_path).names)
    missing = _REQUIRED_KICK_PBP_COLUMNS - schema_names
    if missing:
        print(f"  Stale kick cache at {cache_path} missing {sorted(missing)}; regenerating")
        return False
    return True


def reconstruct_kicker_kicks_from_pbp(
    seasons: list[int],
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Extract individual FG + XP records from play-by-play data.

    Returns one row per kick attempt (FG or XP). Feeds the attention NN's
    inner pool as a variable-length sequence per game — complements the
    weekly-aggregated rows produced by `reconstruct_kicker_weekly_from_pbp`.

    XP rows have `kick_distance=0` and `fg_prob=0`; the `is_xp` flag
    disambiguates (don't conflate with a 0%-probability FG). The `play_id`
    column carries PBP's stable per-play sequence number so downstream
    consumers can sort kicks within a game and apply deterministic
    most-recent truncation.
    """
    if not seasons:
        return pd.DataFrame(columns=_KICKS_SCHEMA)

    cache_path = f"{cache_dir}/kicker_kicks_pbp_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(cache_path) and _cached_kick_pbp_is_current(cache_path):
        return pd.read_parquet(cache_path)

    all_kicks = []
    skipped_seasons: list[int] = []
    for yr in seasons:
        print(f"  Loading per-kick PBP for {yr}...")
        # Wrap the entire per-year extraction (not just import_pbp_data) so a
        # missing column or unexpected schema in one season doesn't abort the
        # whole load. Mirrors the defensive posture of _backfill_2025_pbp_columns.
        try:
            pbp = nfl_source.pbp_data([yr], nfl_source.PBP_KICKER_COLS)
            pbp = pbp[pbp["season_type"] == "REG"]

            fg_rows = pbp[pbp["field_goal_attempt"] == 1]
            # Per-kick dome-aware wind: mirror the weekly builder
            # (``reconstruct_kicker_weekly_from_pbp``) which sets dome games to
            # 0 mph explicitly and leaves outdoor-missing as NaN for the
            # downstream attention input layer to handle. The previous
            # ``wind.fillna(0)`` conflated outdoor-missing with dome — a true
            # 30 mph outdoor game with a missing nfl_data_py weather row would
            # have looked identical to a Mercedes-Benz Stadium kick.
            # NaN propagates to ``build_nested_kick_history`` where
            # ``np.nan_to_num`` zeros it out as a last-resort default; the
            # weekly builder's matching convention is at lines 222-238 / 252-261.
            fg_is_dome = fg_rows["roof"].isin(("dome", "closed"))
            fg_wind = fg_rows["wind"].where(~fg_is_dome, 0.0)
            fg_kicks = pd.DataFrame(
                {
                    "player_id": fg_rows["kicker_player_id"],
                    "season": fg_rows["season"],
                    "week": fg_rows["week"],
                    "play_id": fg_rows["play_id"].astype("int64"),
                    "is_fg": 1,
                    "is_xp": 0,
                    "kick_distance": fg_rows["kick_distance"].fillna(0).astype(float),
                    "kick_made": (fg_rows["field_goal_result"] == "made").astype(int),
                    "fg_prob": fg_rows["fg_prob"].fillna(0).astype(float),
                    "is_q4": (fg_rows["qtr"] >= 4).astype(int),
                    "score_diff": fg_rows["score_differential"].fillna(0).astype(float),
                    "game_wind": fg_wind.astype(float),
                }
            )

            xp_rows = pbp[pbp["extra_point_attempt"] == 1]
            xp_is_dome = xp_rows["roof"].isin(("dome", "closed"))
            xp_wind = xp_rows["wind"].where(~xp_is_dome, 0.0)
            xp_kicks = pd.DataFrame(
                {
                    "player_id": xp_rows["kicker_player_id"],
                    "season": xp_rows["season"],
                    "week": xp_rows["week"],
                    "play_id": xp_rows["play_id"].astype("int64"),
                    "is_fg": 0,
                    "is_xp": 1,
                    "kick_distance": 0.0,
                    "kick_made": (xp_rows["extra_point_result"] == "good").astype(int),
                    "fg_prob": 0.0,
                    "is_q4": (xp_rows["qtr"] >= 4).astype(int),
                    "score_diff": xp_rows["score_differential"].fillna(0).astype(float),
                    "game_wind": xp_wind.astype(float),
                }
            )

            all_kicks.append(pd.concat([fg_kicks, xp_kicks], ignore_index=True))
        except Exception as e:
            print(f"  WARNING: per-kick PBP extraction failed for {yr} ({e}); skipping")
            skipped_seasons.append(yr)
            continue

    if not all_kicks:
        return pd.DataFrame(columns=_KICKS_SCHEMA)

    result = pd.concat(all_kicks, ignore_index=True)
    result = result.dropna(subset=["player_id"]).reset_index(drop=True)
    # Sort by (player_id, season, week, play_id) so downstream truncation by
    # most-recent kicks within a game has well-defined semantics.
    result = result.sort_values(
        ["player_id", "season", "week", "play_id"], kind="stable"
    ).reset_index(drop=True)

    if skipped_seasons:
        # Don't poison the combined cache key with a partial result — the next
        # call would treat it as authoritative for the full range and silently
        # serve kickers from the skipped season(s) with zero kick history.
        # Mirrors the same guard in ``reconstruct_kicker_weekly_from_pbp``; the
        # per-kick path was added later and originally cached unconditionally,
        # so a transient nflverse 502 on one season permanently corrupted the
        # attention NN's inner-pool inputs for that season.
        print(f"  Skipped seasons {skipped_seasons}; not caching partial result to {cache_path}")
        return result

    os.makedirs(cache_dir, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"  Cached per-kick data: {len(result)} kicks -> {cache_path}")
    return result


def load_kicks(k_df: pd.DataFrame) -> pd.DataFrame:
    """Load per-kick records aligned with the weekly kicker DataFrame.

    Called separately from `load_data` so the serving path (app.py)
    doesn't pay the PBP re-parse cost. Returns a DataFrame restricted to the
    (player_id, season) pairs present in the caller's weekly frame, with
    `is_home` merged from the schedule-joined weekly DataFrame.
    """
    kicks_df = reconstruct_kicker_kicks_from_pbp(SEASONS)

    valid_keys = k_df[["player_id", "season"]].drop_duplicates()
    kicks_df = kicks_df.merge(valid_keys, on=["player_id", "season"], how="inner")

    home_lookup = k_df[["player_id", "season", "week", "is_home"]].drop_duplicates()
    kicks_df = kicks_df.merge(home_lookup, on=["player_id", "season", "week"], how="left")
    kicks_df["is_home"] = kicks_df["is_home"].fillna(0).astype(int)

    return kicks_df


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Identity filter — kicker data is pre-filtered."""
    return df.copy()


def season_split(k_df: pd.DataFrame) -> tuple:
    """Split kicker data by season (cross-season, matching other positions).

    Train: 2015-2023, Val: 2024, Test: 2025

    Applies the ``MIN_GAMES`` per-(player_id, season) filter to **train
    only**, mirroring the shared pipeline's train-only ``MIN_GAMES_PER_SEASON``
    filter at ``src/shared/pipeline.py:508``. Val/test rows are kept
    regardless of season-game-count so K's holdout distribution matches the
    other positions.
    """
    train = k_df[k_df["season"] <= _TRAIN_MAX_SEASON].copy()
    # Derive val/test from the train ceiling instead of hardcoding 2024/2025 so
    # the three boundaries can't drift apart (val == +1, test == +2). Inert today
    # (2024 == _TRAIN_MAX_SEASON + 1, 2025 == + 2). (#362 F3)
    val = k_df[k_df["season"] == _TRAIN_MAX_SEASON + 1].copy()
    test = k_df[k_df["season"] == _TRAIN_MAX_SEASON + 2].copy()

    # Train-only MIN_GAMES filter: regress on starters with enough sample for
    # rolling features to stabilize, but evaluate on every active kicker.
    train_games = train.groupby(["player_id", "season"])["week"].transform("count")
    train = train[train_games >= MIN_GAMES].copy()

    print("K cross-season split:")
    if len(train) > 0:
        print(
            f"  Train: {int(train['season'].min())}-{int(train['season'].max())} "
            f"({len(train)} rows)"
        )
    else:
        print("  Train: (empty)")
    print(f"  Val:   2024 ({len(val)} rows)")
    print(f"  Test:  2025 ({len(test)} rows)")

    return train, val, test
