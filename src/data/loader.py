import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.config import (
    CACHE_DIR,
    SCORING,
    SCORING_HALF_PPR,
    SCORING_PPR,
    SCORING_STANDARD,
    SEASONS,
)
from src.data import nfl_source
from src.data.external_sources import (
    CONTRACT_FEATURE_COLUMNS,
    FF_OPP_FEATURE_COLUMNS,
    QBR_FEATURE_COLUMNS,
    _cached_parquet_has_columns,
    _seasons_cache_signature,
    load_contracts,
    load_ff_opportunity,
    load_qbr_weekly,
)
from src.data.nflcom_loader import schedule_team_code_normalization
from src.data.redzone_pbp import RZ_PBP_FEATURE_COLUMNS, reconstruct_redzone_from_pbp

# Re-export the redzone_pbp feature list as a list (for ``not in df.columns``
# loops below). Single source of truth lives in ``redzone_pbp.RZ_PBP_FEATURE_COLUMNS``.
_REDZONE_COLUMNS = list(RZ_PBP_FEATURE_COLUMNS)

# Legacy (NFL-Data-Exchange) depth-chart schema. The merge in ``load_raw_data``
# and the audit in ``src/scripts/audit_features.py`` both depend on exactly these
# columns, so the 2025+ ESPN feed is normalized back to this shape rather than
# changing every consumer (see ``_normalize_espn_depth``).
_DEPTH_CANONICAL_COLS = ["gsis_id", "season", "week", "formation", "depth_team"]

# Schedule-side relocation map (OAK→LV, SD→LAC, STL→LA) — reused so the ESPN
# depth feed's team codes align with the schedule for the as-of join. The 2025
# ESPN code set already matches the schedule's exactly; this only future-proofs
# against a legacy code reappearing.
_TEAM_CODE_NORMALIZATION = schedule_team_code_normalization()


def _is_espn_offense(pos_grp: pd.Series) -> pd.Series:
    """Boolean mask selecting offensive rows in the 2025+ ESPN depth feed.

    The feed groups players by personnel package in ``pos_grp``; today the only
    offensive value is ``"3WR 1TE"`` while defenses are ``"Base 4-3 D"`` /
    ``"Base 3-4 D"`` and special teams is ``"Special Teams"``. Select offense
    *negatively* (not a defensive front, not special teams) so a future
    offensive label (``"2WR 2TE"``, ``"Empty"``, …) isn't silently dropped.
    """
    g = pos_grp.fillna("").str.strip()
    return ~(g.str.endswith(" D") | g.eq("Special Teams"))


def _normalize_espn_depth(espn: pd.DataFrame, schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """Collapse the 2025+ ESPN depth feed into the legacy 5-column depth shape.

    The ESPN feed is a daily snapshot time-series (``dt`` timestamp, no
    season/week) keyed by ``gsis_id`` with depth rank in ``pos_rank``. For each
    of a team's weekly games we keep the latest snapshot at/before kickoff (an
    as-of join on the schedule's ``gameday`` — leakage-safe since depth charts
    are pre-game public info), then take the player's best (min) offensive rank
    that week, mirroring the legacy ``min(depth_team)`` semantics.

    Returns ``gsis_id, season, week, formation="Offense", depth_team`` (str rank
    to match the legacy dtype; the downstream merge re-coerces it to numeric).
    Returns empty if the schedule lacks the columns the join needs — the
    loader's ``-1`` sentinel + consumer-side impute then cover the gap.
    """
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in _DEPTH_CANONICAL_COLS})
    required = {"season", "week", "game_type", "gameday", "home_team", "away_team"}
    if not required.issubset(schedules.columns):
        return empty

    off = espn[_is_espn_offense(espn["pos_grp"])].copy()
    off = off[off["gsis_id"].notna() & off["gsis_id"].astype(str).str.len().gt(0)]
    if off.empty:
        return empty
    off["snapshot_ts"] = pd.to_datetime(off["dt"], errors="coerce", utc=True).dt.tz_localize(None)
    off = off.dropna(subset=["snapshot_ts"])
    off["team"] = off["team"].replace(_TEAM_CODE_NORMALIZATION)
    # Best (min) rank per player per snapshot — a player can hold multiple slots
    # in one snapshot; min mirrors the legacy ``min(depth_team)`` (order-independent).
    snap = off.groupby(["gsis_id", "team", "snapshot_ts"], as_index=False)["pos_rank"].min()

    sched = schedules[(schedules["game_type"] == "REG") & (schedules["season"] == season)]
    # gameday is date-granularity, so kickoff is that day's midnight: a snapshot
    # taken the morning of the game falls *after* it and maps to the prior day's
    # snapshot instead. Conservative and leakage-safe; coverage stays ~99.8%.
    kickoff = pd.to_datetime(sched["gameday"], errors="coerce")
    cal = pd.concat(
        [
            pd.DataFrame(
                {
                    "team": sched["home_team"].replace(_TEAM_CODE_NORMALIZATION),
                    "week": sched["week"],
                    "kickoff": kickoff,
                }
            ),
            pd.DataFrame(
                {
                    "team": sched["away_team"].replace(_TEAM_CODE_NORMALIZATION),
                    "week": sched["week"],
                    "kickoff": kickoff,
                }
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["kickoff"])
    if cal.empty:
        return empty

    # Pair each player-snapshot with each of their team's weekly games, keep
    # snapshots at/before that game's kickoff, then take the latest such snapshot
    # per (player, week) — the depth chart entering that game. A bye week has no
    # cal row, so the prior snapshot correctly carries to the next played week.
    merged = snap.merge(cal, on="team")
    merged = merged[merged["snapshot_ts"] <= merged["kickoff"]]
    if merged.empty:
        return empty
    idx = merged.groupby(["gsis_id", "week"])["snapshot_ts"].idxmax()
    out = merged.loc[idx, ["gsis_id", "week", "pos_rank"]].rename(
        columns={"pos_rank": "depth_team"}
    )
    out["season"] = season
    out["formation"] = "Offense"
    # Drop rows whose latest snapshot has a NaN rank before the int cast — a NaN
    # ``pos_rank`` (e.g. a player listed without a numeric depth) would otherwise
    # raise IntCastingNaNError and abort the whole ESPN-season normalize. These
    # rows carry no usable rank anyway; dropping them mirrors the missing-rank
    # gap the loader's -1 sentinel already covers. (#924)
    out = out.dropna(subset=["depth_team"])
    out["depth_team"] = out["depth_team"].astype("int64").astype(str)
    return out[_DEPTH_CANONICAL_COLS]


def load_team_week_stats(
    seasons: list[int] | None = None, cache_dir: str = CACHE_DIR
) -> pd.DataFrame:
    """Load NFL team-week stats from nflverse release parquets.

    Returns one row per (team, season, week) with full historical defensive
    and offensive columns (def_tds, def_safeties, def_fumbles_forced,
    fg_blocked, pat_blocked, passing_yards, rushing_yards, ...).

    Cache pattern mirrors ``load_raw_data``: persist the concatenated frame
    to ``{cache_dir}/team_stats_{min}_{max}.parquet`` and short-circuit on
    subsequent calls.
    """
    if seasons is None:
        seasons = SEASONS

    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/team_stats_{_seasons_cache_signature(seasons)}.parquet"

    if os.path.exists(path):
        return pd.read_parquet(path)

    parts = []
    skipped = []
    for s in seasons:
        try:
            parts.append(nfl_source.team_week_stats_release(s))
        except Exception as e:
            # nflverse release for the current season can 404 mid-season;
            # skip so cache build succeeds on older seasons.
            skipped.append(s)
            print(f"WARNING: team_stats fetch failed for {s} ({e}); skipping")

    if not parts:
        # Don't poison the cache with an empty frame — let the next call retry.
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if skipped:
        # Partial coverage: some seasons failed to fetch. Return what we have for
        # THIS call, but do NOT persist a partial frame as the authoritative cache
        # — a cache-hit on the next call would serve the incomplete frame forever,
        # and downstream fillna(0) would silently zero the missing seasons' team
        # stats (the cache-survives-gap failure mode this codebase guards against
        # in redzone_pbp / k.data). (#807)
        print(f"WARNING: team_stats partial coverage (skipped seasons {skipped}); not caching")
        return df
    df.to_parquet(path)
    return df


def load_raw_data(seasons: list[int] | None = None, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load and merge NFL weekly data, rosters, snap counts, and schedules.

    Most network/parquet fetches (weekly, rosters, schedules, snap counts,
    injuries, red-zone PBP) run in a ``ThreadPoolExecutor`` so a cold cache
    populates in parallel. Depth charts run *after* the pool: the 2025+ ESPN
    format needs schedule kickoff dates to map its daily snapshots onto NFL
    weeks (see ``_fetch_depth`` / ``_normalize_espn_depth``). Each fetch is
    HTTP/parquet I/O and so spends most of its time off the GIL. Cache hits
    short-circuit at the start of each helper so warm starts pay only the
    parquet read cost (also fanned out, but trivial).
    """
    if seasons is None:
        seasons = SEASONS

    os.makedirs(cache_dir, exist_ok=True)
    _sig = _seasons_cache_signature(seasons)
    weekly_path = f"{cache_dir}/weekly_{_sig}.parquet"
    rosters_path = f"{cache_dir}/rosters_{_sig}.parquet"
    schedules_path = f"{cache_dir}/schedules_{_sig}.parquet"
    snap_path = f"{cache_dir}/snap_counts_{_sig}.parquet"
    injury_path = f"{cache_dir}/injuries_{_sig}.parquet"
    # ``_v2`` cache-version sentinel: PR #595's REG-only ``week -= 1`` realignment
    # only runs in _fetch_depth's cache-miss branch, so a pre-#595 ``depth_charts_*``
    # parquet (stale-by-1 weeks) would be served verbatim and silently revert the
    # fix. Bumping the filename makes any legacy cache unreachable, forcing a
    # re-fetch through the realignment. (#616)
    depth_path = f"{cache_dir}/depth_charts_v2_{_sig}.parquet"

    def _fetch_weekly():
        # Schema-gate: regenerate if the cache predates the current
        # nfl_source._WEEKLY_RENAME layer — a renamed column absent ⇒ stale schema,
        # and downstream fillna would silently zero it. These four are guaranteed
        # present after the rename. (#428)
        _weekly_required = ("recent_team", "interceptions", "sacks", "sack_yards")
        if os.path.exists(weekly_path) and _cached_parquet_has_columns(
            weekly_path, _weekly_required
        ):
            return pd.read_parquet(weekly_path)
        # All seasons come from nflreadpy's load_player_stats (the modern nflverse
        # stats_player release). Harmonisation of that schema to the legacy weekly
        # column names lives in src.data.nfl_source.weekly_data — one code path for
        # every season (previously ≤2024 used nfl_data_py and ≥2025 read the
        # stats_player parquet directly, with the rename inlined here).
        weekly = nfl_source.weekly_data(seasons)
        weekly.to_parquet(weekly_path)
        return weekly

    def _fetch_rosters():
        if os.path.exists(rosters_path):
            return pd.read_parquet(rosters_path)
        rosters = nfl_source.rosters(seasons)
        # Coerce mixed-type columns that break parquet serialization
        # (e.g. jersey_number is str in older seasons, int in newer)
        for col in rosters.columns:
            if rosters[col].dtype == object:
                rosters[col] = rosters[col].astype(str)
        rosters.to_parquet(rosters_path)
        return rosters

    def _fetch_schedules():
        if os.path.exists(schedules_path):
            return pd.read_parquet(schedules_path)
        schedules = nfl_source.schedules(seasons)
        schedules.to_parquet(schedules_path)
        return schedules

    def _fetch_snap_counts():
        if os.path.exists(snap_path):
            return pd.read_parquet(snap_path)
        snap_seasons = [s for s in seasons if s >= 2012]
        snap_counts = nfl_source.snap_counts(snap_seasons) if snap_seasons else pd.DataFrame()
        snap_counts.to_parquet(snap_path)
        return snap_counts

    def _fetch_injuries():
        if os.path.exists(injury_path):
            return pd.read_parquet(injury_path)
        injuries = nfl_source.injuries(seasons)
        injuries.to_parquet(injury_path)
        return injuries

    def _fetch_depth(schedules):
        if os.path.exists(depth_path):
            return pd.read_parquet(depth_path)
        # nflreadpy's load_depth_charts serves the legacy NFL-Data-Exchange schema
        # for ≤2024 and the new ESPN schema (dt/pos_grp/pos_rank, daily snapshots,
        # no season/week) for ≥2025. Pull legacy via the nfl_source shim; normalize
        # ESPN to the legacy 5-column shape so the downstream Offense/depth_team
        # merge stays untouched.
        old_seasons = [s for s in seasons if s <= 2024]
        new_seasons = [s for s in seasons if s >= 2025]
        parts = []
        if old_seasons:
            # The legacy NFL Data Exchange ``week`` is stale by one game: the chart
            # labeled week W reflects the lineup from week W-1, not the one entering
            # week W (audited 2026-05-30 by src/analysis/audit_depth_alignment.py —
            # rank-1-QB vs actual-starter match 0.873 at the raw label vs 0.913 after
            # ``week -= 1`` (the argmax over shifts ±2); at QB-change weeks 0.227 match
            # the current week vs 0.707 the prior week). Realign so week-W
            # ``depth_chart_rank`` is the chart entering week W's game — matching the
            # >=2025 ESPN as-of-kickoff path (``_normalize_espn_depth``), which removes
            # a train(stale)/test(current) skew. REG-only: playoff ``week`` numbers
            # overlap the regular season, and ``_normalize_espn_depth`` is REG-only too.
            # The labeled week-1 chart (preseason roster) shifts to week 0 and drops out
            # of the (player, season, week) merge; the "extra" week-19 label backfills
            # week 18, so coverage stays ~99%. The residual QB-change-week lag is
            # depth-chart inertia (charts react ~2 weeks late) — a source limit no
            # uniform relabel fixes.
            legacy = nfl_source.depth_charts(old_seasons)
            legacy = legacy[legacy["game_type"] == "REG"].copy()
            legacy["week"] = pd.to_numeric(legacy["week"], errors="coerce") - 1
            legacy = legacy.dropna(subset=["week"])
            legacy["week"] = legacy["week"].astype(int)
            parts.append(legacy[_DEPTH_CANONICAL_COLS])
        for s in new_seasons:
            url = (
                "https://github.com/nflverse/nflverse-data/releases/download/"
                f"depth_charts/depth_charts_{s}.parquet"
            )
            # The nflverse ESPN release for the current season can 404 mid-season;
            # skip that season (mirrors the team_week_stats skip above) so a
            # transient miss doesn't abort the depth-chart load for all six
            # positions. The loader's -1 sentinel + consumer-side impute cover the
            # gap. (#830)
            try:
                parts.append(_normalize_espn_depth(pd.read_parquet(url), schedules, s))
            except Exception as e:
                print(f"WARNING: ESPN depth_charts fetch failed for {s} ({e}); skipping")
        depth = pd.concat(parts, ignore_index=True)
        depth.to_parquet(depth_path)
        return depth

    def _fetch_redzone():
        # Schema-gated cache lives inside reconstruct_redzone_from_pbp; the
        # cache_path is derived from CACHE_DIR + seasons range, so this call
        # short-circuits to a parquet read on warm cache.
        return reconstruct_redzone_from_pbp(seasons, cache_dir=cache_dir)

    def _fetch_ff_opp():
        return load_ff_opportunity(seasons, cache_dir=cache_dir)

    def _fetch_qbr():
        # Merge-ready weekly QBR (ESPN id already bridged to gsis internally).
        return load_qbr_weekly(seasons, cache_dir=cache_dir)

    def _fetch_contracts():
        return load_contracts(seasons, cache_dir=cache_dir)

    with ThreadPoolExecutor(max_workers=10) as pool:
        weekly_f = pool.submit(_fetch_weekly)
        rosters_f = pool.submit(_fetch_rosters)
        schedules_f = pool.submit(_fetch_schedules)
        snap_counts_f = pool.submit(_fetch_snap_counts)
        injuries_f = pool.submit(_fetch_injuries)
        redzone_f = pool.submit(_fetch_redzone)
        ff_opp_f = pool.submit(_fetch_ff_opp)
        qbr_f = pool.submit(_fetch_qbr)
        contracts_f = pool.submit(_fetch_contracts)
        weekly = weekly_f.result()
        rosters = rosters_f.result()
        # _fetch_schedules() persists schedules_{min}_{max}.parquet for downstream
        # consumers (src/k/data.py::load_data, src/shared/weather_features.py::
        # _load_schedules). The frame is also consumed below by _fetch_depth, which
        # needs kickoff dates to map the 2025+ ESPN depth feed (a daily snapshot
        # series with no week column) onto NFL weeks via an as-of join.
        schedules = schedules_f.result()
        snap_counts = snap_counts_f.result()
        injuries = injuries_f.result()
        redzone = redzone_f.result()
        ff_opp = ff_opp_f.result()
        qbr = qbr_f.result()
        contracts = contracts_f.result()

    # Depth charts depend on the schedule (as-of join for the 2025+ ESPN format),
    # so they run after the pool rather than inside it. Cold cache pays one extra
    # HTTP read; the warm path short-circuits on the parquet.
    depth = _fetch_depth(schedules)

    # --- Merge rosters for position override ---
    roster_pos = rosters[["player_id", "season", "position"]].drop_duplicates(
        subset=["player_id", "season"]
    )
    weekly = weekly.merge(
        roster_pos, on=["player_id", "season"], how="left", suffixes=("_weekly", "")
    )
    if "position_weekly" in weekly.columns:
        weekly["position"] = weekly["position"].fillna(weekly["position_weekly"])
        weekly.drop(columns=["position_weekly"], inplace=True)

    # --- Merge snap counts via ID bridge ---
    try:
        ids = nfl_source.player_ids()
        pfr_to_gsis = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()
        # Subset-dedup on pfr_id before merging — the full-row drop_duplicates
        # above only collapses identical (pfr_id, gsis_id) pairs. If a single
        # pfr_id maps to multiple distinct gsis_ids in the ID release (rare
        # ID-system churn or data-entry collisions), the snap-count merge below
        # would fan out and multiply each player-week row. Keep "first" — the
        # downstream consequence of dropping a real cross-reference is one
        # player-week with NaN snap_pct (gracefully filled by the
        # position-week median imputation), strictly better than silently
        # double-counting.
        pfr_to_gsis = pfr_to_gsis.drop_duplicates(subset=["pfr_id"], keep="first")
        assert pfr_to_gsis["pfr_id"].is_unique, (
            "pfr_to_gsis must be unique on pfr_id after subset-dedup"
        )
        snap_counts = snap_counts.merge(
            pfr_to_gsis, left_on="pfr_player_id", right_on="pfr_id", how="left"
        )
        snap_merged = snap_counts[["gsis_id", "season", "week", "offense_pct"]].dropna(
            subset=["gsis_id"]
        )
        weekly = weekly.merge(
            snap_merged,
            left_on=["player_id", "season", "week"],
            right_on=["gsis_id", "season", "week"],
            how="left",
        )
        if "gsis_id" in weekly.columns:
            weekly.drop(columns=["gsis_id"], inplace=True)
        weekly.rename(columns={"offense_pct": "snap_pct"}, inplace=True)
    except Exception as e:
        print(f"WARNING: Snap count merge failed ({e}), snap_pct will be NaN")
        if "snap_pct" not in weekly.columns:
            weekly["snap_pct"] = float("nan")

    # 5. Injury reports
    # Wrapped defensively (parallel to the snap-count merge above): a malformed
    # or empty injuries frame (e.g. a season the source has no injury data for,
    # or an upstream schema rename) would otherwise KeyError on the .map/.groupby
    # below. On failure the practice/game status columns still land with their
    # neutral fill defaults so downstream features see "no injury" rather than
    # crashing.
    try:
        practice_map = {
            "Full Participation in Practice": 2,
            "Limited Participation in Practice": 1,
            "Did Not Participate In Practice": 0,
        }
        injuries["practice_status_num"] = injuries["practice_status"].map(practice_map)

        status_map = {"Questionable": 0.5, "Doubtful": 0.1, "Out": 0.0}
        injuries["game_status_num"] = injuries["report_status"].map(status_map).fillna(1.0)

        # Worst practice/game status per player-week (multiple injuries possible)
        inj_agg = (
            injuries.groupby(["gsis_id", "season", "week"])
            .agg(
                practice_status=("practice_status_num", "min"),
                game_status=("game_status_num", "min"),
            )
            .reset_index()
        )

        weekly = weekly.merge(
            inj_agg,
            left_on=["player_id", "season", "week"],
            right_on=["gsis_id", "season", "week"],
            how="left",
        )
        weekly.drop(columns=["gsis_id"], errors="ignore", inplace=True)
    except Exception as e:
        print(f"WARNING: Injury merge failed ({e}), injury status will be neutral")
    if "practice_status" not in weekly.columns:
        weekly["practice_status"] = float("nan")
    if "game_status" not in weekly.columns:
        weekly["game_status"] = float("nan")
    weekly["practice_status"] = weekly["practice_status"].fillna(2.0)
    weekly["game_status"] = weekly["game_status"].fillna(1.0)

    # 6. Depth charts (Offense formation, most recent entry per player-week)
    depth_off = depth[depth["formation"] == "Offense"].copy()
    depth_off["depth_team"] = pd.to_numeric(depth_off["depth_team"], errors="coerce")
    # One row per player-week. ``min`` picks the best (lowest) rank a player
    # held that week — order-independent (so deterministic across runs) and
    # meaningful when a player is listed at multiple positions in one week
    # (e.g., starter at TE + third-string FB → rank 1, their primary role).
    depth_agg = (
        depth_off.groupby(["gsis_id", "season", "week"])
        .agg(depth_chart_rank=("depth_team", "min"))
        .reset_index()
    )

    weekly = weekly.merge(
        depth_agg,
        left_on=["player_id", "season", "week"],
        right_on=["gsis_id", "season", "week"],
        how="left",
    )
    weekly.drop(columns=["gsis_id"], errors="ignore", inplace=True)
    # Sentinel: depth-chart-absent rows get -1 (clearly out-of-band — legitimate
    # depth ranks are 1+). Previously these were filled with 3 (third string),
    # which conflated "no data" with "third-string player" and biased every
    # downstream feature that uses depth_chart_rank to treat unknown players
    # as buried on the depth chart. Consumers treat -1 as the explicit
    # "unlisted / unavailable depth chart" category. The clip upper bound is loosened to 10 so
    # legitimate deeper ranks survive intact; depth chart rows beyond 10 are
    # exceedingly rare (and almost certainly noise) so they still get capped.
    weekly["depth_chart_rank"] = weekly["depth_chart_rank"].fillna(-1).clip(lower=-1, upper=10)

    # --- Merge red-zone PBP aggregates ---
    # Per-game red-zone touch counts feed the attention NN's history sequence
    # (added to ATTN_HISTORY_STATS per-position). Targets the RB attention-NN
    # weakness on sparse TD heads (rushing_tds +0.080 MAE vs Ridge) where the
    # model has no upstream red-zone-usage signal to learn TD propensity from.
    # Merged for all positions, not just RB — QB/WR/TE can wire these into
    # their own ATTN_HISTORY_STATS as one-line follow-ups.
    # Players with no PBP red-zone activity (or seasons before PBP rows
    # carry the player) get 0, not NaN.
    if not redzone.empty:
        weekly = weekly.merge(
            redzone,
            on=["player_id", "season", "week", "recent_team"],
            how="left",
        )
    for col in _REDZONE_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = 0.0
        weekly[col] = weekly[col].fillna(0.0)

    # --- Merge ff_opportunity expected-stat per-game columns ---
    # Per-game expected stats (modeled from in-game opportunity) feed the
    # attention NN history sequence for QB/RB/WR/TE, and prior-season means of
    # them feed the static branch. ff_opp is gsis-keyed (joins directly) and
    # deduped to one row per player-game, so this left-merge can't fan out.
    # No modeled opportunity that week → 0 (a meaningful low-usage signal).
    if not ff_opp.empty:
        weekly = weekly.merge(ff_opp, on=["player_id", "season", "week"], how="left")
    for col in FF_OPP_FEATURE_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = 0.0
        weekly[col] = weekly[col].fillna(0.0)

    # --- Merge weekly ESPN QBR (already bridged ESPN id → gsis in load_qbr_weekly) ---
    # QB-only per-game signal. Left NaN (not zero-filled): non-QB rows never use
    # it, QB weeks without a QBR are skipped by the prior-season mean and zeroed
    # only inside the attention history tensor (build_game_history_arrays).
    if not qbr.empty:
        weekly = weekly.merge(qbr, on=["player_id", "season", "week"], how="left")
    for col in QBR_FEATURE_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = float("nan")

    # --- Merge active-as-of-season contract attributes ---
    # Player-season state (one row per player-season → no fan-out). Players with
    # no contract data (undrafted/minimum/team-unit DST) → 0.
    if not contracts.empty:
        weekly = weekly.merge(contracts, on=["player_id", "season"], how="left")
    for col in CONTRACT_FEATURE_COLUMNS:
        if col not in weekly.columns:
            weekly[col] = 0.0
        weekly[col] = weekly[col].fillna(0.0)

    return weekly


def compute_fantasy_points(df: pd.DataFrame, scoring: dict | None = None) -> pd.Series:
    """Compute fantasy points from raw stat columns for a given scoring dict."""
    if scoring is None:
        scoring = SCORING

    col_map = {
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "interceptions",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
    }

    fantasy_points = pd.Series(0.0, index=df.index)
    for key, weight in scoring.items():
        if key == "fumbles_lost":
            val = (
                df["sack_fumbles_lost"].fillna(0)
                + df["rushing_fumbles_lost"].fillna(0)
                + df["receiving_fumbles_lost"].fillna(0)
            )
        else:
            val = df[col_map[key]].fillna(0)
        fantasy_points += val * weight
    return fantasy_points


def compute_all_scoring_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fantasy points for standard, half-PPR, and full PPR formats.

    Adds columns: fantasy_points_standard, fantasy_points_half_ppr, fantasy_points
    """
    # Copy before mutating so ad-hoc callers (REPL/notebooks) don't see their
    # input frame silently gain columns — mirrors the copy-first contract in
    # src/data/preprocessing.py and src/data/split.py.
    df = df.copy()
    df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)
    df["fantasy_points"] = compute_fantasy_points(df, SCORING_PPR)
    return df
