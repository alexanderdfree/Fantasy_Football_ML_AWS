import logging

import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    EWMA_SPANS,
    EWMA_STATS,
    OPP_ROLLING_WINDOW,
    ROLL_STATS,
    ROLLING_WINDOWS,
    SEASONS,
    SHARE_WINDOWS,
    TREND_STATS,
)
from src.data.external_sources import EXTERNAL_PRIOR_STATS
from src.shared.weather_features import (
    _TEAM_CODE_NORMALIZATION,
    _load_schedules,
    build_implied_team_total_lookup,
)

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the engineered feature columns from preprocessed data.

    Covers rolling / prior-season / EWMA / trend / share / opponent-defense
    aggregates plus the per-game history columns and the external-source
    (ff_opportunity / QBR / contract) signals merged upstream by
    src.data.loader. Each position's config opts into the subset it consumes.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- Rolling Features (84: 81 mean/std/max + 3 min) ---
    rolling_cols: dict[str, pd.Series] = {}
    for stat in ROLL_STATS:
        grouped = df.groupby(["player_id", "season"])[stat]
        for window in ROLLING_WINDOWS:
            rolling_cols[f"rolling_mean_{stat}_L{window}"] = grouped.transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
            )
            rolling_cols[f"rolling_std_{stat}_L{window}"] = grouped.transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).std()
            )
            rolling_cols[f"rolling_max_{stat}_L{window}"] = grouped.transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).max()
            )
            if stat == "fantasy_points":
                rolling_cols[f"rolling_min_{stat}_L{window}"] = grouped.transform(
                    lambda x, w=window: x.shift(1).rolling(w, min_periods=1).min()
                )
    df = pd.concat([df, pd.DataFrame(rolling_cols, index=df.index)], axis=1)

    # --- Prior-Season Summary Features (27 = 9 stats x 3 aggs) ---
    prior_stats = list(ROLL_STATS)
    prior = df.groupby(["player_id", "season"])[prior_stats].agg(["mean", "std", "max"])
    prior.columns = [f"prior_season_{agg}_{stat}" for stat, agg in prior.columns]
    prior = prior.reset_index()
    prior["season"] = prior["season"] + 1  # align S-1 stats with season S
    df = df.merge(prior, on=["player_id", "season"], how="left")

    # Prior-season total touchdowns: rushing_tds + receiving_tds summed across
    # all the player's season S-1 games, merged onto season S. TD propensity
    # (red-zone usage, scheme role) is a player attribute that's typically
    # consistent year-over-year and complementary to the volume aggregates
    # above — none of which encode score-converting efficiency. Positions
    # that don't use TDs (DST/K) simply omit this from their INCLUDE_FEATURES
    # whitelist; the column is always materialised because rushing_tds and
    # receiving_tds are present on every preprocessed weekly row.
    tds_per_season = df.groupby(["player_id", "season"])[["rushing_tds", "receiving_tds"]].sum()
    prior_tds = (
        (tds_per_season["rushing_tds"] + tds_per_season["receiving_tds"])
        .rename("prior_season_total_touchdowns")
        .reset_index()
    )
    prior_tds["season"] = prior_tds["season"] + 1
    df = df.merge(prior_tds, on=["player_id", "season"], how="left")

    # Three more prior-season aggregates (PR #192) that decompose the
    # fantasy-points signal into orthogonal upstream pieces. PR #191 dropped
    # both prior_season_mean_fantasy_points and prior_season_max_fantasy_points
    # but the post-merge EC2 retrain showed a +0.072 Attention NN MAE
    # regression — the FP aggregates were carrying signal that the
    # decomposed features (catch_rate, YPC, total_touchdowns) didn't fully
    # restore. These three close that gap:
    #
    #  * prior_season_total_yards: sum of (rushing_yards + receiving_yards)
    #    across S-1 games. Yards × 0.1 is ~90% of FP for non-TD-heavy backs;
    #    the *combined* total (rather than separate rushing/receiving sums)
    #    avoids re-introducing the prior_season_*_rushing_yards redundancy
    #    PR #190 found with prior_season_*_carries.
    #  * prior_season_games_played: count of S-1 game rows that *survived
    #    preprocess* — i.e. games the player was active for and accumulated
    #    measurable stats in. Despite the name, this is **not** the schedule's
    #    games-played count: byes, suspensions, IR weeks, and inactive
    #    games are all excluded by the upstream preprocess filter. A 16-game
    #    season with a 3-game IR stint reads as 13, not 16. The semantic is
    #    "games active". Treated as a feature it still gives Ridge / the base
    #    NN the volume normaliser they need (per-game means × games → season
    #    totals), and the discrepancy is small for the cohort that drives
    #    fantasy decisions (top-60 RBs/WRs typically miss 0–3 games). Sourcing
    #    from schedule would require an extra join with debatable benefit;
    #    keeping the active-games proxy and documenting the semantic is the
    #    lower-risk option per audit-318 (W.SHARED-ENG, medium).
    #  * prior_season_mean_fumbles_lost: per-game fumble rate. Negative-FP
    #    component currently absent from every prior_season aggregate. Mean
    #    rather than total because rates compose better with the existing
    #    per-game means; total has many zeros and high variance.
    #
    # ``fumbles_lost`` isn't a preprocessed column directly (the position
    # targets.py modules sum it), so we compose it inline from the three
    # source components present in every weekly row. Synthetic test
    # fixtures sometimes omit one or more components — guard via ``in``
    # checks so adding into a missing column doesn't crash with
    # ``AttributeError: 'int' object has no attribute 'fillna'``.
    def _col_or_zero(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(0)
        return pd.Series(0, index=df.index, dtype=float)

    df_fumbles = (
        _col_or_zero("sack_fumbles_lost")
        + _col_or_zero("rushing_fumbles_lost")
        + _col_or_zero("receiving_fumbles_lost")
    )
    df_total_yards = _col_or_zero("rushing_yards") + _col_or_zero("receiving_yards")
    prior_extra = (
        pd.DataFrame(
            {
                "player_id": df["player_id"],
                "season": df["season"],
                "_total_yards": df_total_yards,
                "_fumbles_lost": df_fumbles,
            }
        )
        .groupby(["player_id", "season"])
        .agg(
            prior_season_total_yards=("_total_yards", "sum"),
            prior_season_games_played=("_total_yards", "size"),
            prior_season_mean_fumbles_lost=("_fumbles_lost", "mean"),
        )
        .reset_index()
    )
    prior_extra["season"] = prior_extra["season"] + 1
    df = df.merge(prior_extra, on=["player_id", "season"], how="left")

    # Prior-season red-zone touch volume + rate. Sourced from per-game PBP
    # aggregates merged onto each weekly row by
    # src.data.redzone_pbp.reconstruct_redzone_from_pbp via src.data.loader.
    # ``prior_season_total_touchdowns`` above encodes score-converting
    # *outcome*; the red-zone aggregates encode the *opportunity* that
    # outcome is consistent with, separating "high TD propensity because the
    # scheme keeps feeding him goal-line work" from "got lucky on rate."
    # Motivation: the RB attention-NN loses to Ridge on sparse TD heads
    # (rushing_tds +0.080 MAE, receiving_tds +0.035) and has no red-zone
    # signal in ATTN_HISTORY_STATS pre-task; the hurdle_poisson architectural
    # fix was rejected for regressing aggregate FP MAE
    # (see TODO.md "[TESTED, REJECTED] RB hurdle_poisson on sparse-count heads").
    rz_carries = _col_or_zero("redzone_carries")
    rz_targets = _col_or_zero("redzone_targets")
    prior_rz = (
        pd.DataFrame(
            {
                "player_id": df["player_id"],
                "season": df["season"],
                "_rz_touches": rz_carries + rz_targets,
            }
        )
        .groupby(["player_id", "season"])
        .agg(prior_season_total_redzone_touches=("_rz_touches", "sum"))
        .reset_index()
    )
    prior_rz["season"] = prior_rz["season"] + 1
    df = df.merge(prior_rz, on=["player_id", "season"], how="left")
    # Per-game rate: derived after the merge so it composes
    # ``prior_season_total_redzone_touches`` with the just-merged
    # ``prior_season_games_played``. Guard the divisor so rookies (0 prior
    # games) get 0, not NaN/inf — matches the loader's fillna(0) convention
    # for missing red-zone columns.
    games = df["prior_season_games_played"].where(df["prior_season_games_played"] > 0)
    df["prior_season_mean_redzone_touches_per_game"] = (
        df["prior_season_total_redzone_touches"] / games
    ).fillna(0.0)

    # --- Prior-season means of external opportunity / QBR signals ---
    # ff_opportunity expected stats and weekly ESPN QBR are merged per-game by
    # src.data.loader (via src.data.external_sources). Their prior-season means
    # form the static-branch "opportunity / talent" prior — the leakage-safe
    # (season + 1 shift), non-cumulative shape the attention static branch
    # accepts (career-cumulative is deliberately kept in the history domain; see
    # tests/test_attn_static_columns.py, which forbids career_* from static).
    # ff_opp is skill-position-only and QBR is QB-only, so guard each column
    # with ``in``. ff_opportunity is zero-filled upstream for no-opportunity
    # weeks; QBR remains NaN for missing QB weeks and non-QB rows, then groupby
    # mean skips those gaps. NaN here is the same rookie-first-season shape as
    # every other prior_season_* feature and is handled by the downstream fill.
    external_prior_stats = [c for c in EXTERNAL_PRIOR_STATS if c in df.columns]
    if external_prior_stats:
        prior_external = (
            df.groupby(["player_id", "season"])[external_prior_stats]
            .mean()
            .rename(columns=lambda c: f"prior_season_mean_{c}")
            .reset_index()
        )
        prior_external["season"] = prior_external["season"] + 1  # align S-1 with S
        df = df.merge(prior_external, on=["player_id", "season"], how="left")

    # --- EWMA Features (14) ---
    ewma_cols: dict[str, pd.Series] = {}
    for stat in EWMA_STATS:
        grouped = df.groupby(["player_id", "season"])[stat]
        for span in EWMA_SPANS:
            ewma_cols[f"ewma_{stat}_L{span}"] = grouped.transform(
                lambda x, s=span: x.shift(1).ewm(span=s, min_periods=1).mean()
            )
    df = pd.concat([df, pd.DataFrame(ewma_cols, index=df.index)], axis=1)

    # --- Trend / Momentum Features (4) ---
    for stat in TREND_STATS:
        short = df.groupby(["player_id", "season"])[stat].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        long = df.groupby(["player_id", "season"])[stat].transform(
            lambda x: x.shift(1).rolling(8, min_periods=1).mean()
        )
        df[f"trend_{stat}"] = short - long

    # --- Share / Usage Features (5) ---
    team_totals = (
        df.groupby(["recent_team", "season", "week"])
        .agg(
            team_targets=("targets", "sum"),
            team_carries=("carries", "sum"),
        )
        .reset_index()
    )
    df = df.merge(team_totals, on=["recent_team", "season", "week"], how="left")

    # Detect team changes for stint-aware grouping
    df = df.sort_values(["player_id", "season", "week"])
    df["team_changed"] = (
        df.groupby(["player_id", "season"])["recent_team"].shift(1) != df["recent_team"]
    ).fillna(False)
    df["stint_id"] = df.groupby(["player_id", "season"])["team_changed"].cumsum()

    share_cols: dict[str, pd.Series] = {}
    stint_g = df.groupby(["player_id", "season", "stint_id"])
    for window in SHARE_WINDOWS:
        player_tgt = stint_g["targets"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        team_tgt = stint_g["team_targets"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        share_cols[f"target_share_L{window}"] = (
            (player_tgt / team_tgt).replace([np.inf, -np.inf], 0).fillna(0)
        )

        player_car = stint_g["carries"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        team_car = stint_g["team_carries"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).sum()
        )
        share_cols[f"carry_share_L{window}"] = (
            (player_car / team_car).replace([np.inf, -np.inf], 0).fillna(0)
        )
    df = pd.concat([df, pd.DataFrame(share_cols, index=df.index)], axis=1)

    # air_yards_share (lagged to prevent data leakage)
    if "receiving_air_yards" in df.columns:
        team_air_yards = df.groupby(["recent_team", "season", "week"])[
            "receiving_air_yards"
        ].transform("sum")
        df["_raw_air_yards_share"] = (
            (df["receiving_air_yards"] / team_air_yards).replace([np.inf, -np.inf], 0).fillna(0)
        )
        df["air_yards_share"] = (
            df.groupby(["player_id", "season"])["_raw_air_yards_share"].shift(1).fillna(0)
        )
        df.drop(columns=["_raw_air_yards_share"], inplace=True)
    else:
        df["air_yards_share"] = 0.0

    # snap_pct gets two derived forms:
    #   - ``snap_pct_raw``: the un-lagged per-game value, consumed by the
    #     attention history sequence. ``build_game_history_arrays`` applies its
    #     own leakage-safe ``shift(1)``, so feeding it the already-lagged
    #     ``snap_pct`` double-lagged the sequence (#788).
    #   - ``snap_pct`` (static feature): prior week's snap %, lagged
    #     ``stint``-aware so a player traded mid-season doesn't inherit the old
    #     team's snap share on his first game with the new team (#677). The
    #     non-stint-aware ``groupby(player_id, season)`` lag carried it over.
    # ``stint_id`` is still present here (dropped just below). The rolling / ewma
    # / trend snap_pct features above were already computed off the raw value
    # (before this block) with their own shift, so they are unaffected.
    if "snap_pct" in df.columns:
        # Raw per-game snap %, missing -> 0 (a missing snap row means the player
        # didn't take snaps, the same 0-fill the old in-place lag applied). This
        # is the history-sequence input; build_game_history_arrays lags it.
        df["snap_pct_raw"] = df["snap_pct"].fillna(0)
        df["snap_pct"] = (
            df.groupby(["player_id", "season", "stint_id"])["snap_pct_raw"].shift(1).fillna(0)
        )

    # Clean up intermediate columns
    drop_cols = ["team_targets", "team_carries", "team_changed", "stint_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Matchup / Opponent Features (4) ---
    df = _build_matchup_features(df)

    # --- Defense Matchup Features (7) ---
    df = _build_defense_matchup_features(df)

    # --- Contextual Features (4) ---
    df = _build_contextual_features(df)

    # --- Position Encoding (4) ---
    pos_cols = {f"pos_{p}": (df["position"] == p).astype(int) for p in ["QB", "RB", "WR", "TE"]}
    df = pd.concat([df, pd.DataFrame(pos_cols, index=df.index)], axis=1)

    return df


# === Game History Extraction (for attention model) ===

# Default per-game stats to include in history vectors.
# These are the raw stats that the rolling features are derived from.
GAME_HISTORY_STATS = [
    "fantasy_points",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "attempts",
    "completions",
    "carries",
    "targets",
    "receptions",
    "snap_pct_raw",
    "interceptions",
]


def build_game_history_arrays(
    df: pd.DataFrame,
    history_stats: list[str] = None,
    max_seq_len: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-player game history as padded arrays for the attention model.

    For each player-week row, gathers that player's prior games within the same
    season (shifted by 1 to prevent leakage — same convention as rolling features).

    Ordering is newest-first: index 0 holds the most-recent prior game, with older
    games at higher indices and zero-padding on the right. This keeps each absolute
    sequence index at a fixed recency ("games ago") regardless of how many prior
    games the player has, so the attention branch's learned positional embedding at
    a given index always means the same thing across players.

    Args:
        df: DataFrame with player_id, season, week, and stat columns.
            Must already be sorted by [player_id, season, week].
        history_stats: list of column names to include per game.
        max_seq_len: maximum history length (pad/truncate to this).

    Returns:
        X_history: [n_samples, max_seq_len, game_dim] float32 array (right-padded;
            index 0 = most-recent prior game)
        history_mask: [n_samples, max_seq_len] bool array (True = real game)
    """
    if history_stats is None:
        history_stats = GAME_HISTORY_STATS
    # Fail loud on missing cols (was: silent filter). Silent-filter let a
    # config addition to attn_history_stats train with the wrong game_dim,
    # surfacing only at smoke-test time as a state_dict shape mismatch
    # (PR #235's redzone cols, RB stable pinned ~24 h).
    missing = [s for s in history_stats if s not in df.columns]
    if missing:
        raise KeyError(
            f"build_game_history_arrays: attn_history_stats columns missing "
            f"from df: {missing}. Likely cause: data/splits/*.parquet was "
            f"generated before these columns were added to a position's "
            f"attn_history_stats — trigger refresh-splits.yml (or regenerate "
            f"via the SETUP.md data-pull snippet and re-upload to S3) so the "
            f"splits include the new columns."
        )
    game_dim = len(history_stats)

    n = len(df)
    X_history = np.zeros((n, max_seq_len, game_dim), dtype=np.float32)
    history_mask = np.zeros((n, max_seq_len), dtype=bool)

    if n == 0:
        return X_history, history_mask

    # Work with positional indices throughout. reset_index(drop=True) makes
    # sort_values(...).index yield caller row *positions* (0..n-1) rather than
    # caller-supplied labels, which fancy-indexes safely even if caller indices
    # are duplicated or non-contiguous.
    df_pos = df.reset_index(drop=True)
    sorted_idx = df_pos.sort_values(["player_id", "season", "week"], kind="stable").index.to_numpy()
    stat_values = df_pos.loc[sorted_idx, history_stats].to_numpy(dtype=np.float32)
    player_ids = df_pos["player_id"].to_numpy()[sorted_idx]
    seasons = df_pos["season"].to_numpy()[sorted_idx]

    # (player_id, season) -> list of sorted-row positions
    group_indices: dict[tuple, list[int]] = {}
    for sorted_pos in range(n):
        key = (player_ids[sorted_pos], seasons[sorted_pos])
        group_indices.setdefault(key, []).append(sorted_pos)

    hist_sorted = np.zeros_like(X_history)
    mask_sorted = np.zeros_like(history_mask)

    for indices in group_indices.values():
        for pos_in_group, row_idx in enumerate(indices):
            if pos_in_group == 0:
                continue  # no history for first game of season

            start = max(0, pos_in_group - max_seq_len)
            # Reverse so the most-recent prior game is first (index 0) and older
            # games follow; padding stays on the right. Fixes each index to a
            # constant recency across players (see function docstring).
            prior_indices = indices[start:pos_in_group][::-1]
            seq_len = len(prior_indices)

            hist_sorted[row_idx, :seq_len] = stat_values[prior_indices]
            mask_sorted[row_idx, :seq_len] = True

    # Scatter sorted results back to caller row order.
    X_history[sorted_idx] = hist_sorted
    history_mask[sorted_idx] = mask_sorted

    # Replace NaN with 0 in history
    np.nan_to_num(X_history, copy=False, nan=0.0)

    return X_history, history_mask


# === Opponent-Defense History Extraction (for second attention branch) ===

# Canonical per-game defensive stats. The attention branch learns the trailing-
# form signal directly from this sequence; keeping these six mirrors the
# existing L5 static features so the NN gets an equivalent opponent-defense
# signal (just unrolled). `def_pts_allowed` is sourced from schedule scores;
# the rest are aggregated from player rows in the training frame.
OPP_DEFENSE_HISTORY_STATS = [
    "def_sacks",
    "def_pass_yds_allowed",
    "def_pass_td_allowed",
    "def_ints",
    "def_rush_yds_allowed",
    "def_pts_allowed",
]


def build_opp_defense_per_game_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(opponent_team, season, week) defensive stats.

    Mirrors the aggregation inside :func:`_build_defense_matchup_features` but
    returns the *unrolled* per-game frame so it can drive an attention sequence
    instead of a pre-computed L5 average. `def_pts_allowed` is merged in from
    the cached schedules parquet so the 6-stat schema matches the L5 feature
    set column-for-column (minus the ``_L5`` suffix).

    The caller is expected to pass an **all-position** DataFrame — the sacks /
    pass-yards / rush-yards aggregates sum over every offensive player, so a
    position-filtered frame would miss contributions from other positions.

    Returns:
        DataFrame with columns ``opponent_team, season, week`` + the six
        defensive stats in :data:`OPP_DEFENSE_HISTORY_STATS`.
    """
    required = {
        "opponent_team",
        "season",
        "week",
        "sacks",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
    }
    missing = required - set(df.columns)
    if missing or df["opponent_team"].isna().all():
        # Return an empty frame with the full schema so callers downstream
        # degrade gracefully to zero-filled history tensors.
        return pd.DataFrame(columns=["opponent_team", "season", "week"] + OPP_DEFENSE_HISTORY_STATS)

    # Normalize the player frame's opponent_team to modern codes before the
    # groupby — mirror of the schedule-side normalization for def_pts_allowed
    # below. A no-op on nflverse weekly data (already modern: LA/LAC/LV); closes
    # the silent-miss gap if a source ever feeds historical codes (OAK/SD/STL).
    df = df.copy()
    df["opponent_team"] = df["opponent_team"].replace(_TEAM_CODE_NORMALIZATION)

    def_stats = (
        df.groupby(["opponent_team", "season", "week"])
        .agg(
            def_sacks=("sacks", "sum"),
            def_pass_yds_allowed=("passing_yards", "sum"),
            def_pass_td_allowed=("passing_tds", "sum"),
            def_ints=("interceptions", "sum"),
            def_rush_yds_allowed=("rushing_yards", "sum"),
        )
        .reset_index()
    )

    # Points allowed — sourced from schedule scores (away_score for the home
    # team and vice versa). Same pattern as ``_build_defense_matchup_features``.
    schedules_path = f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    try:
        schedules = pd.read_parquet(schedules_path)
    except FileNotFoundError:
        # Tests and callers without the schedules cache still get the five
        # player-derived stats; pts_allowed falls back to 0.
        def_stats["def_pts_allowed"] = 0.0
        return def_stats[["opponent_team", "season", "week"] + OPP_DEFENSE_HISTORY_STATS]

    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()
    # Normalize historical team codes (OAK→LV, SD→LAC, STL→LA) so the join
    # finds rows for relocated franchises in their pre-relocation seasons.
    # Mirrors _build_team_schedule_lookup / build_implied_team_total_lookup
    # in src.shared.weather_features. Without this, def_pts_allowed for a
    # team like OAK in 2017–2019 silently fills with 0 because the schedule
    # already uses "LV" while the player frame still carries "OAK".
    schedules_reg["away_team"] = schedules_reg["away_team"].replace(_TEAM_CODE_NORMALIZATION)
    schedules_reg["home_team"] = schedules_reg["home_team"].replace(_TEAM_CODE_NORMALIZATION)
    away_pts = schedules_reg[["season", "week", "away_team", "home_score"]].copy()
    away_pts.columns = ["season", "week", "team", "def_pts_allowed"]
    home_pts = schedules_reg[["season", "week", "home_team", "away_score"]].copy()
    home_pts.columns = ["season", "week", "team", "def_pts_allowed"]
    pts_allowed = pd.concat([away_pts, home_pts], ignore_index=True).drop_duplicates(
        subset=["season", "week", "team"]
    )

    def_stats = def_stats.merge(
        pts_allowed.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"],
        how="left",
    )
    def_stats["def_pts_allowed"] = def_stats["def_pts_allowed"].fillna(0.0)

    return def_stats[["opponent_team", "season", "week"] + OPP_DEFENSE_HISTORY_STATS]


# === Opponent-Offense History Extraction (DST second attention branch) ===
#
# DST faces an *offense*, not a defense, so the parallel attention branch
# for DST attends over the opposing OFFENSE's per-game trailing form (what
# the upcoming opponent has done with the ball over their season). The
# branch reuses `build_opp_defense_history_arrays` unchanged — that function
# is already generic over its per-game frame and stat list. Only the
# per-game builder differs. ``off_pts_scored`` mirrors ``def_pts_allowed``
# in how it sources from the schedules cache.
OPP_OFFENSE_HISTORY_STATS = [
    "off_pass_yards",
    "off_pass_tds",
    "off_rush_yards",
    "off_rush_tds",
    "off_ints",
    "off_fumbles_lost",
    "off_pts_scored",
]


def build_opp_offense_per_game_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(opponent_team, season, week) offensive stats.

    Semantic inverse of :func:`build_opp_defense_per_game_df`. Groups by
    ``recent_team`` (the team whose offense produced the stats) and renames
    that key to ``opponent_team`` so downstream history lookups — which key
    on ``opponent_team`` — find the row for whoever a given DST is facing
    that week. ``off_pts_scored`` is sourced from schedule scores (the
    team's *own* score in that game), matching how ``def_pts_allowed`` is
    sourced for the defense helper.

    The caller is expected to pass an **all-position** DataFrame — the
    passing, rushing, and turnover columns sum over every offensive
    player, so a position-filtered frame (e.g. RB-only) would systematically
    undercount.

    Returns:
        DataFrame with columns ``opponent_team, season, week`` + the seven
        offensive stats in :data:`OPP_OFFENSE_HISTORY_STATS`.
    """
    required = {
        "recent_team",
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "rushing_yards",
        "rushing_tds",
        "interceptions",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    }
    missing = required - set(df.columns)
    if missing:
        # Schema violation — the caller is supposed to pass an all-position
        # weekly frame with these stat columns. Returning an empty frame here
        # silently degrades DST's opponent-offense attention branch to zeros,
        # so surface the missing columns explicitly. Kept as a warning + empty
        # frame rather than KeyError to preserve the test contract in
        # tests/test_opp_offense_history.py::test_missing_columns_returns_empty_frame
        # (file outside this worker bundle).
        logger.warning(
            "build_opp_offense_per_game_df: missing required columns %s — "
            "returning empty frame (downstream attention will see zeros).",
            sorted(missing),
        )
    if missing or df["recent_team"].isna().all():
        return pd.DataFrame(columns=["opponent_team", "season", "week"] + OPP_OFFENSE_HISTORY_STATS)

    # Combined fumbles_lost across sack / rushing / receiving sources — same
    # combination DST's own `fumbles_lost` head uses.
    work = df.copy()
    # Normalize the player frame's recent_team to modern codes before the
    # groupby — mirror of the opponent_team normalization in
    # _build_defense_matchup_features and the schedule-side normalization of
    # off_pts_scored below. nflverse weekly data already emits modern codes
    # (LA/LAC/LV), so this is a no-op in production; it closes the silent-miss
    # gap should any upstream source feed historical codes (OAK/SD/STL).
    work["recent_team"] = work["recent_team"].replace(_TEAM_CODE_NORMALIZATION)
    work["_fumbles_lost"] = (
        work["sack_fumbles_lost"].fillna(0)
        + work["rushing_fumbles_lost"].fillna(0)
        + work["receiving_fumbles_lost"].fillna(0)
    )

    off_stats = (
        work.groupby(["recent_team", "season", "week"])
        .agg(
            off_pass_yards=("passing_yards", "sum"),
            off_pass_tds=("passing_tds", "sum"),
            off_rush_yards=("rushing_yards", "sum"),
            off_rush_tds=("rushing_tds", "sum"),
            off_ints=("interceptions", "sum"),
            off_fumbles_lost=("_fumbles_lost", "sum"),
        )
        .reset_index()
        .rename(columns={"recent_team": "opponent_team"})
    )

    # Points scored — sourced from schedule scores. Each schedule row
    # contributes home_team -> home_score and away_team -> away_score.
    schedules_path = f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    try:
        schedules = pd.read_parquet(schedules_path)
    except FileNotFoundError:
        # Tests and degraded callers still get the six player-derived stats;
        # off_pts_scored falls back to 0. Kept as a warning + fallback rather
        # than re-raising to preserve the test contract in
        # tests/test_opp_offense_history.py::test_missing_schedules_falls_back_to_zero_pts
        # (file outside this worker bundle).
        logger.warning(
            "build_opp_offense_per_game_df: schedules parquet not found at "
            "%s — off_pts_scored falls back to 0.",
            schedules_path,
        )
        off_stats["off_pts_scored"] = 0.0
        return off_stats[["opponent_team", "season", "week"] + OPP_OFFENSE_HISTORY_STATS]

    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()
    # Normalize historical team codes (OAK→LV, SD→LAC, STL→LA) — mirror of the
    # def-side normalization in build_opp_defense_per_game_df. Without it,
    # off_pts_scored for relocated franchises in pre-relocation seasons
    # silently fills with 0 because the schedule already uses the new code.
    schedules_reg["home_team"] = schedules_reg["home_team"].replace(_TEAM_CODE_NORMALIZATION)
    schedules_reg["away_team"] = schedules_reg["away_team"].replace(_TEAM_CODE_NORMALIZATION)
    home_pts = schedules_reg[["season", "week", "home_team", "home_score"]].copy()
    home_pts.columns = ["season", "week", "team", "off_pts_scored"]
    away_pts = schedules_reg[["season", "week", "away_team", "away_score"]].copy()
    away_pts.columns = ["season", "week", "team", "off_pts_scored"]
    pts_scored = pd.concat([home_pts, away_pts], ignore_index=True).drop_duplicates(
        subset=["season", "week", "team"]
    )

    off_stats = off_stats.merge(
        pts_scored.rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"],
        how="left",
    )
    off_stats["off_pts_scored"] = off_stats["off_pts_scored"].fillna(0.0)

    return off_stats[["opponent_team", "season", "week"] + OPP_OFFENSE_HISTORY_STATS]


# Single source of truth for opp-history per-game builders. Both training
# (``src.shared.pipeline``) and serving (``src.serving.app``) look up by
# the same key, eliminating the per-callsite if/else surface flagged in
# CLAUDE.md as a recurring training/inference drift source.
OPP_ATTN_PER_GAME_BUILDERS = {
    "defense": build_opp_defense_per_game_df,
    "offense": build_opp_offense_per_game_df,
}


def build_opp_defense_history_arrays(
    df: pd.DataFrame,
    opp_def_per_game: pd.DataFrame,
    opp_stats: list[str] | None = None,
    max_seq_len: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-sample padded arrays of the opponent defense's prior games.

    Analogue of :func:`build_game_history_arrays` but the sequence being
    gathered is the *opponent defense's* game log instead of the player's own.
    For each ``(opponent_team, season, week)`` in ``df``, looks up that
    defense's games in ``opp_def_per_game`` within the same season and
    strictly before the target week (shift-by-1 equivalent, realised via the
    ``week_opp < week_target`` filter).

    Args:
        df: per-sample DataFrame — must contain ``opponent_team``, ``season``,
            ``week``. Sorted order is not required; results are scattered back
            to caller row order.
        opp_def_per_game: frame from :func:`build_opp_defense_per_game_df`;
            columns ``opponent_team, season, week`` + stat columns.
        opp_stats: subset of columns in ``opp_def_per_game`` to include per
            game; defaults to :data:`OPP_DEFENSE_HISTORY_STATS` intersected
            with the frame's columns.
        max_seq_len: pad/truncate sequences to this length.

    Returns:
        ``(X_opp, opp_mask)`` where ``X_opp`` is ``[n, max_seq_len, opp_dim]``
        float32 (newest → oldest within the sequence: index 0 = most-recent prior
        game, right-padded — mirrors :func:`build_game_history_arrays`) and
        ``opp_mask`` is ``[n, max_seq_len]`` bool (``True`` = real game).
    """
    if opp_stats is None:
        opp_stats = OPP_DEFENSE_HISTORY_STATS
    # Intersect with the lookup frame to stay robust to missing columns — same
    # convention as build_game_history_arrays.
    opp_stats = [s for s in opp_stats if s in opp_def_per_game.columns]
    opp_dim = len(opp_stats)

    n = len(df)
    X_opp = np.zeros((n, max_seq_len, opp_dim), dtype=np.float32)
    opp_mask = np.zeros((n, max_seq_len), dtype=bool)

    if n == 0 or opp_dim == 0 or len(opp_def_per_game) == 0:
        return X_opp, opp_mask

    # Pre-sort the lookup frame so per-(team, season) slices are already in
    # chronological order. One sort avoids an O(n·log n) cost per sample.
    lookup = opp_def_per_game.sort_values(["opponent_team", "season", "week"]).reset_index(
        drop=True
    )
    # (opp_team, season) -> contiguous (start, end) index range into the sorted
    # lookup. The per-game arrays (team/season/week/stat) live as module-local
    # numpy arrays below, not in this dict — values stored here are 2-tuples of
    # ints, so the annotation reflects ``tuple[int, int]``.
    groups: dict[tuple, tuple[int, int]] = {}
    team_arr = lookup["opponent_team"].to_numpy()
    season_arr = lookup["season"].to_numpy()
    week_arr = lookup["week"].to_numpy()
    stat_arr = lookup[opp_stats].to_numpy(dtype=np.float32)
    for idx in range(len(lookup)):
        key = (team_arr[idx], season_arr[idx])
        slot = groups.get(key)
        if slot is None:
            groups[key] = (idx, idx + 1)
        else:
            start, _ = slot
            groups[key] = (start, idx + 1)

    df_opp = df["opponent_team"].to_numpy()
    df_season = df["season"].to_numpy()
    df_week = df["week"].to_numpy()

    for row_idx in range(n):
        key = (df_opp[row_idx], df_season[row_idx])
        slot = groups.get(key)
        if slot is None:
            continue
        start, end = slot
        # Prior games = rows in (start, end) whose week is strictly less than
        # target week. `week_arr` is sorted within the group, so the upper bound
        # is found by a linear walk (seasons cap at ~17 weeks; no need for bisect).
        take_end = start
        target_week = df_week[row_idx]
        while take_end < end and week_arr[take_end] < target_week:
            take_end += 1
        seq_len = take_end - start
        if seq_len <= 0:
            continue
        if seq_len > max_seq_len:
            # Keep the MOST RECENT max_seq_len games (drop oldest), same
            # semantics as build_game_history_arrays' tail-take.
            slice_start = take_end - max_seq_len
            seq_len = max_seq_len
        else:
            slice_start = start
        # Reverse to newest-first (index 0 = most-recent prior game), right-padded —
        # mirrors build_game_history_arrays so the opp-defense attention branch shares
        # the same recency-indexed positional convention.
        X_opp[row_idx, :seq_len] = stat_arr[slice_start:take_end][::-1]
        opp_mask[row_idx, :seq_len] = True

    np.nan_to_num(X_opp, copy=False, nan=0.0)
    return X_opp, opp_mask


def _build_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build opponent/matchup features."""
    # Determine opponent from schedule or opponent_team column. Normalize to
    # modern team codes (mirrors _build_defense_matchup_features and the opp-
    # defense aggregator): inert today since weekly opponent_team is already
    # modern, but keeps the group/merge key from splitting a relocated team
    # across legacy+modern codes if an upstream source ever mixed them (#440/#549).
    if "opponent_team" in df.columns:
        df["opponent"] = df["opponent_team"].replace(_TEAM_CODE_NORMALIZATION)
    else:
        df["opponent"] = None

    # Compute rush and receiving fantasy components
    df["rush_fantasy"] = df["rushing_yards"] * 0.1 + df["rushing_tds"] * 6
    df["recv_fantasy"] = (
        df["receiving_yards"] * 0.1 + df["receiving_tds"] * 6 + df["receptions"] * 1
    )

    if df["opponent"].notna().any():
        def_pts = (
            df.groupby(["opponent", "position", "season", "week"])
            .agg(
                pts_allowed_to_pos=("fantasy_points", "sum"),
                rush_pts_allowed_to_pos=("rush_fantasy", "sum"),
                recv_pts_allowed_to_pos=("recv_fantasy", "sum"),
            )
            .reset_index()
        )

        def_pts = def_pts.sort_values(["opponent", "position", "season", "week"])
        for col in ["pts_allowed_to_pos", "rush_pts_allowed_to_pos", "recv_pts_allowed_to_pos"]:
            def_pts[f"opp_{col}"] = def_pts.groupby(["opponent", "position", "season"])[
                col
            ].transform(lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean())

        def_pts.rename(
            columns={
                "opp_pts_allowed_to_pos": "opp_fantasy_pts_allowed_to_pos",
            },
            inplace=True,
        )

        # Rank: 1 = most points allowed = best matchup
        def_pts["opp_def_rank_vs_pos"] = def_pts.groupby(["position", "season", "week"])[
            "opp_fantasy_pts_allowed_to_pos"
        ].rank(ascending=False, method="min")

        merge_cols = [
            "opponent",
            "position",
            "season",
            "week",
            "opp_fantasy_pts_allowed_to_pos",
            "opp_rush_pts_allowed_to_pos",
            "opp_recv_pts_allowed_to_pos",
            "opp_def_rank_vs_pos",
        ]
        def_pts_merge = def_pts[merge_cols].drop_duplicates()
        n_before = len(df)
        df = df.merge(def_pts_merge, on=["opponent", "position", "season", "week"], how="left")
        if len(df) != n_before:
            df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")
    else:
        for col in [
            "opp_fantasy_pts_allowed_to_pos",
            "opp_rush_pts_allowed_to_pos",
            "opp_recv_pts_allowed_to_pos",
            "opp_def_rank_vs_pos",
        ]:
            df[col] = 0.0

    df.drop(columns=["rush_fantasy", "recv_fantasy", "opponent"], errors="ignore", inplace=True)
    return df


def _build_defense_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build detailed opposing-defense features from team-level aggregations.

    Computes 5 rolling defense stats (sacks, pass yds/TDs allowed, INTs, rush yds allowed),
    1 schedule-derived stat (points allowed), and 1 Vegas feature (implied team total).
    """
    # --- 1. Defense stats derived from opponent offensive data ---
    if "opponent_team" not in df.columns or df["opponent_team"].isna().all():
        for col in [
            "opp_def_sacks_L5",
            "opp_def_pass_yds_allowed_L5",
            "opp_def_pass_td_allowed_L5",
            "opp_def_ints_L5",
            "opp_def_rush_yds_allowed_L5",
            "opp_def_pts_allowed_L5",
            "implied_team_total",
        ]:
            df[col] = 0.0
        return df

    # Normalize the player frame's opponent_team (and recent_team, used by the
    # implied-total merge below) to modern codes BEFORE the groupby/merges. The
    # schedule side is normalized too, so both sides must agree. nflverse weekly
    # data already emits modern codes (LA/LAC/LV) for relocation franchises,
    # making this a no-op in production — but it closes the latent silent-miss
    # gap if any upstream source ever feeds a frame with historical codes
    # (OAK/SD/STL). A one-sided (schedule-only) normalization cannot survive
    # that case. See nflcom_loader.schedule_team_code_normalization.
    df = df.copy()
    df["opponent_team"] = df["opponent_team"].replace(_TEAM_CODE_NORMALIZATION)
    if "recent_team" in df.columns:
        df["recent_team"] = df["recent_team"].replace(_TEAM_CODE_NORMALIZATION)

    # Aggregate offensive stats allowed by each defense per game
    def_stats = (
        df.groupby(["opponent_team", "season", "week"])
        .agg(
            _def_sacks=("sacks", "sum"),
            _def_pass_yds=("passing_yards", "sum"),
            _def_pass_tds=("passing_tds", "sum"),
            _def_ints=("interceptions", "sum"),
            _def_rush_yds=("rushing_yards", "sum"),
        )
        .reset_index()
    )

    def_stats.sort_values(["opponent_team", "season", "week"], inplace=True)

    # L5 rolling averages with shift(1) for leakage prevention
    stat_map = {
        "_def_sacks": "opp_def_sacks_L5",
        "_def_pass_yds": "opp_def_pass_yds_allowed_L5",
        "_def_pass_tds": "opp_def_pass_td_allowed_L5",
        "_def_ints": "opp_def_ints_L5",
        "_def_rush_yds": "opp_def_rush_yds_allowed_L5",
    }
    for raw_col, out_col in stat_map.items():
        def_stats[out_col] = def_stats.groupby(["opponent_team", "season"])[raw_col].transform(
            lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean()
        )

    # Merge onto player rows via opponent_team
    merge_cols = ["opponent_team", "season", "week"] + list(stat_map.values())
    def_merge = def_stats[merge_cols].drop_duplicates()
    n_before = len(df)
    df = df.merge(def_merge, on=["opponent_team", "season", "week"], how="left")
    if len(df) != n_before:
        df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")

    # --- 2. Points allowed from schedule scores ---
    # Route via _load_schedules so the test autouse fixture (which patches
    # _load_schedules when the parquet is absent) catches this read too.
    # _load_schedules already filters to game_type == "REG".
    schedules_reg = _load_schedules().copy()

    # Normalize historical team codes (OAK→LV, SD→LAC, STL→LA) so the join
    # against opponent_team matches for relocated franchises in their
    # pre-relocation seasons. Mirrors build_opp_defense_per_game_df's
    # normalization. Without this, opp_def_pts_allowed_L5 for a team like
    # OAK in 2017–2019 silently fills with 0.
    schedules_reg["away_team"] = schedules_reg["away_team"].replace(_TEAM_CODE_NORMALIZATION)
    schedules_reg["home_team"] = schedules_reg["home_team"].replace(_TEAM_CODE_NORMALIZATION)
    away_pts = schedules_reg[["season", "week", "away_team", "home_score"]].copy()
    away_pts.columns = ["season", "week", "team", "points_allowed"]
    home_pts = schedules_reg[["season", "week", "home_team", "away_score"]].copy()
    home_pts.columns = ["season", "week", "team", "points_allowed"]
    pts_allowed = pd.concat([away_pts, home_pts], ignore_index=True)
    pts_allowed.sort_values(["team", "season", "week"], inplace=True)

    pts_allowed["opp_def_pts_allowed_L5"] = pts_allowed.groupby(["team", "season"])[
        "points_allowed"
    ].transform(lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean())

    pts_merge = pts_allowed[["team", "season", "week", "opp_def_pts_allowed_L5"]].drop_duplicates()
    n_before = len(df)
    df = df.merge(
        pts_merge,
        left_on=["opponent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    )
    if len(df) != n_before:
        df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")
    df.drop(columns=["team"], errors="ignore", inplace=True)

    # --- 3. Implied team total from Vegas lines ---
    # Single source of truth for the formula lives in
    # ``src.shared.weather_features.build_implied_team_total_lookup``; both
    # this path (tests that call ``build_features`` directly) and the
    # production schedule merge (``merge_schedule_features`` → its
    # ``_build_team_schedule_lookup`` inlines the same formula on the
    # enriched lookup) share that helper so the per-team value can't drift
    # between code paths.
    impl_lookup = build_implied_team_total_lookup(schedules_reg)

    if "implied_team_total" in df.columns:
        df.drop(columns=["implied_team_total"], inplace=True)
    n_before = len(df)
    df = df.merge(impl_lookup, on=["season", "week", "recent_team"], how="left")
    if len(df) != n_before:
        df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")

    # Fill NaNs (early-season games with no prior history) for the rolling
    # opp-defense stats. ``implied_team_total`` is deliberately NOT filled here:
    # the production schedule-merge path (``merge_schedule_features``) preserves
    # NaN for unmatched games so downstream code can detect them, and the final
    # feature build (``feature_build.py``'s catch-all
    # ``replace([inf, -inf], nan).fillna(0)``) maps any survivor to 0 uniformly.
    # Force-filling 0 here diverged from that NaN-preserving contract.
    for col in list(stat_map.values()) + ["opp_def_pts_allowed_L5"]:
        df[col] = df[col].fillna(0)

    return df


def _build_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build contextual features: is_home, week, is_returning_from_absence, days_rest."""
    # is_home: when the upstream loader doesn't carry venue info, default to 0
    # (treat as "away"). This is a *biased* fallback — production data always
    # has ``is_home`` populated; only synthetic test fixtures and degraded
    # serving paths hit this branch. Switching the default to 0.5 (neutral)
    # would silently shift training-vs-inference parity if either path were to
    # bypass the schedule join unintentionally, so we keep the explicit-bias
    # default and rely on upstream paths to populate the column.
    if "is_home" not in df.columns:
        df["is_home"] = 0

    # is_returning_from_absence
    df = df.sort_values(["player_id", "season", "week"])
    df["weeks_since_last_game"] = df.groupby(["player_id", "season"])["week"].diff().fillna(1)
    df["is_returning_from_absence"] = (df["weeks_since_last_game"] > 1).astype(int)
    df.drop(columns=["weeks_since_last_game"], inplace=True)

    # days_rest (approximate: 7 days per week gap)
    df["days_rest"] = df.groupby(["player_id", "season"])["week"].diff().fillna(1) * 7
    df["days_rest"] = df["days_rest"].clip(lower=4, upper=21)

    # Injury status (merged in loader; ensure defaults for missing data)
    if "practice_status" not in df.columns:
        df["practice_status"] = 2.0
    if "game_status" not in df.columns:
        df["game_status"] = 1.0

    # Depth chart rank (merged in loader; ensure default).
    # loader.py now fills absent depth ranks with -1 (a clearly out-of-band
    # sentinel; see src/data/loader.py); this 3.0 column-absent fallback only
    # fires for synthetic fixtures lacking the column entirely.
    if "depth_chart_rank" not in df.columns:
        df["depth_chart_rank"] = 3.0

    return df


# ── Whitelist-based feature selection ────────────────────────────────────────
INCLUDE_CATEGORY_ORDER = [
    "rolling",
    "prior_season",
    "ewma",
    "trend",
    "share",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
    "specific",
]


def flatten_include_features(include_dict: dict[str, list[str]]) -> list[str]:
    """Flatten a whitelist feature dictionary into an ordered column list."""
    unknown = set(include_dict) - set(INCLUDE_CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"Unknown feature categories: {unknown}")
    cols = []
    for key in INCLUDE_CATEGORY_ORDER:
        cols.extend(include_dict.get(key, []))
    return cols


def get_attn_static_columns(
    all_feature_cols: list[str],
    static_whitelist: list[str],
) -> list[str]:
    """Return the subset of ``all_feature_cols`` whitelisted for the attention
    NN's static branch, preserving the input column order.

    Each position's config (``attn_static_features`` for QB/RB/WR/TE/K/DST)
    owns the whitelist. The attention branch learns its own temporal
    representation from ``{POS}_ATTN_HISTORY_STATS``, so rolling / EWMA / trend
    / share / specific categories are intentionally excluded upstream at the
    config level — making new-feature inclusion opt-in and eliminating the
    silent leak the previous prefix/suffix blacklist had for ``_L3``/``_L5``
    specific columns.
    """
    whitelist = set(static_whitelist)
    available = set(all_feature_cols)
    dropped = whitelist - available
    if dropped:
        logger.warning(
            "get_attn_static_columns: %d whitelist entries missing from "
            "all_feature_cols and will be silently dropped: %s",
            len(dropped),
            sorted(dropped),
        )
    return [c for c in all_feature_cols if c in whitelist]
