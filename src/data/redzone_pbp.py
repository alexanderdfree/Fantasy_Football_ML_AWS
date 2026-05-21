"""Red-zone & goal-line per-game aggregates from play-by-play data.

Produces a (player_id, season, week, recent_team) frame with per-game red-zone
touch counts that feed into the attention NN's history sequence. The
motivation is the RB attention-NN's weakness on the two sparse TD heads —
``rushing_tds`` (+0.080 MAE vs Ridge) and ``receiving_tds`` (+0.035) — where
the model has no upstream red-zone-usage signal to learn TD propensity from.
The ``hurdle_poisson`` architectural fix was rejected (TODO.md archive) because
it regressed aggregate FP MAE; this loader adds the missing feature instead.

Mirrors the schema-gated cache pattern from :mod:`src.k.data` (see the
``[FIXED] K underprojection: stale PBP cache survived fg_yards_made schema
addition`` archive entry — a cache that survives a schema change silently
zeros downstream targets when ``fillna(0)`` runs).
"""

from __future__ import annotations

import os

import nfl_data_py as nfl
import pandas as pd
import pyarrow.parquet as pq

from src.config import CACHE_DIR

# Feature columns the current implementation produces (excluding the four
# ``player_id``/``season``/``week``/``recent_team`` merge keys). Single
# source of truth for both the schema-gate below and ``src/data/loader.py``'s
# zero-backfill so adding a sixth column here doesn't require a coordinated
# edit in two places — silent drift between the two lists used to risk a
# new aggregation surviving the merge but missing the backfill, leaving NaN
# rows that look identical to "player did not appear in PBP".
RZ_PBP_FEATURE_COLUMNS: tuple[str, ...] = (
    "redzone_carries",
    "redzone_targets",
    "inside10_carries",
    "inside5_carries",
    "redzone_target_share",
)

# Per-game aggregate columns the current implementation produces. A cached
# parquet missing any of these was written by an older code version and must
# be regenerated rather than served — downstream merges left-join then
# fillna(0), so a missing column would silently zero the feature across the
# entire training range (same bug class as the K fg_yards_made incident).
_REQUIRED_RZ_PBP_COLUMNS = frozenset(
    ("player_id", "season", "week", "recent_team") + RZ_PBP_FEATURE_COLUMNS
)


def _cached_rz_pbp_is_current(cache_path: str) -> bool:
    """Return True only if the cached parquet has every column the current
    aggregation produces.

    Reads just the parquet schema (no row data) so the check is cheap. A
    False return signals the caller to ignore the cache and regenerate.
    """
    schema_names = set(pq.read_schema(cache_path).names)
    missing = _REQUIRED_RZ_PBP_COLUMNS - schema_names
    if missing:
        print(f"  Stale cache at {cache_path} missing {sorted(missing)}; regenerating")
        return False
    return True


def _aggregate_one_season(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one season's PBP frame into per-(player, week, team) red-zone counts.

    Three independent aggregations:

    1. Rushing-side: for rows with a ``rusher_player_id``, count plays with
       ``yardline_100 <= {20, 10, 5}`` grouped by player-week-team.
    2. Receiving-side: for pass-attempt rows with a ``receiver_player_id``,
       count plays with ``yardline_100 <= 20`` grouped by player-week-team.
    3. Team-level denominator: for all pass attempts, count plays with
       ``yardline_100 <= 20`` grouped by team-week. Used to convert
       ``redzone_targets`` (count) into ``redzone_target_share`` (rate).

    The receiving filter requires ``pass_attempt == 1`` so designed runs from
    shotgun (with a ``receiver_player_id`` field set by the parser) don't
    leak into the target count.

    Regular-season-only filter applied defensively. The public caller in
    this module already filters before delegating; the redundant guard here
    keeps the invariant local so direct/ad-hoc callers (notebooks, REPL)
    can't accidentally fold playoff plays into the per-game aggregates.
    """
    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"] == "REG"]
    in_rz = pbp["yardline_100"] <= 20

    # --- Rushing-side aggregates ---
    rush = pbp[pbp["rusher_player_id"].notna()].copy()
    rush_yl = rush["yardline_100"]
    rush["_rz_carry"] = (rush_yl <= 20).astype("int32")
    rush["_in10_carry"] = (rush_yl <= 10).astype("int32")
    rush["_in5_carry"] = (rush_yl <= 5).astype("int32")

    rushing = (
        rush.groupby(["rusher_player_id", "season", "week", "posteam"], dropna=True)
        .agg(
            redzone_carries=("_rz_carry", "sum"),
            inside10_carries=("_in10_carry", "sum"),
            inside5_carries=("_in5_carry", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id", "posteam": "recent_team"})
    )

    # --- Receiving-side aggregates ---
    recv = pbp[(pbp["receiver_player_id"].notna()) & (pbp["pass_attempt"] == 1)].copy()
    recv["_rz_target"] = (recv["yardline_100"] <= 20).astype("int32")

    receiving = (
        recv.groupby(["receiver_player_id", "season", "week", "posteam"], dropna=True)
        .agg(redzone_targets=("_rz_target", "sum"))
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "recent_team"})
    )

    # --- Team red-zone pass attempts (denominator for redzone_target_share) ---
    team_rz_pass = (
        pbp[(pbp["pass_attempt"] == 1) & in_rz]
        .groupby(["posteam", "season", "week"], dropna=True)
        .size()
        .rename("team_rz_pass_attempts")
        .reset_index()
        .rename(columns={"posteam": "recent_team"})
    )

    # --- Merge rushing + receiving on player-week-team ---
    merged = rushing.merge(
        receiving, on=["player_id", "season", "week", "recent_team"], how="outer"
    )
    # Players who only rushed have NaN targets and vice versa; zero them.
    for col in ("redzone_carries", "inside10_carries", "inside5_carries", "redzone_targets"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int32")

    # --- Compute redzone_target_share ---
    merged = merged.merge(team_rz_pass, on=["recent_team", "season", "week"], how="left")
    denom = merged["team_rz_pass_attempts"]
    # 0/0 → 0 (player has no RZ targets in a game with no team RZ pass attempts).
    merged["redzone_target_share"] = (merged["redzone_targets"] / denom.where(denom > 0)).fillna(
        0.0
    )
    merged.drop(columns=["team_rz_pass_attempts"], inplace=True)

    return merged


def reconstruct_redzone_from_pbp(
    seasons: list[int],
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Reconstruct per-(player_id, season, week, recent_team) red-zone aggregates from PBP.

    Returned columns:
        player_id, season, week, recent_team,
        redzone_carries, redzone_targets, inside10_carries, inside5_carries,
        redzone_target_share

    Caches the aggregated frame at
    ``{cache_dir}/redzone_pbp_{seasons[0]}_{seasons[-1]}.parquet``. The cache
    is schema-gated by :data:`_REQUIRED_RZ_PBP_COLUMNS` — a parquet missing
    any required column is regenerated rather than served (mirrors the K
    pattern so the same class of bug can't ship again).

    Args:
        seasons: list of season years to extract.
        cache_dir: override for ``CACHE_DIR``. Resolved at call time so
            module-level monkeypatches of ``CACHE_DIR`` take effect.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_path = f"{cache_dir}/redzone_pbp_{seasons[0]}_{seasons[-1]}.parquet"
    if os.path.exists(cache_path) and _cached_rz_pbp_is_current(cache_path):
        return pd.read_parquet(cache_path)

    all_weekly: list[pd.DataFrame] = []
    skipped_seasons: list[int] = []
    for yr in seasons:
        print(f"  Loading PBP red-zone aggregates for {yr}...")
        # Per-year try/except so a single 502 / schema break doesn't abort the
        # whole load — mirrors the defensive posture of
        # reconstruct_kicker_weekly_from_pbp.
        try:
            pbp = nfl.import_pbp_data([yr], downcast=True)
            pbp = pbp[pbp["season_type"] == "REG"]
            weekly = _aggregate_one_season(pbp)
            all_weekly.append(weekly)
        except Exception as e:
            print(f"  WARNING: red-zone PBP extraction failed for {yr} ({e}); skipping")
            skipped_seasons.append(yr)
            continue

    if not all_weekly:
        return pd.DataFrame(columns=sorted(_REQUIRED_RZ_PBP_COLUMNS))

    result = pd.concat(all_weekly, ignore_index=True)

    if skipped_seasons:
        # Don't poison the cache with a partial result — the next call would
        # treat it as authoritative for the full range and silently serve
        # incomplete data. Same rule as K's loader.
        print(f"  Skipped seasons {skipped_seasons}; not caching partial result to {cache_path}")
        return result

    os.makedirs(cache_dir, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"  Cached red-zone PBP aggregates: {len(result)} rows -> {cache_path}")
    return result
