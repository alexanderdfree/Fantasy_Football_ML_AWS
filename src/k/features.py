import numpy as np
import pandas as pd

from src.k.config import POSITION_CONFIG
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the K model."""
    return list(POSITION_CONFIG.all_features)


def compute_features(df: pd.DataFrame) -> None:
    """Compute all kicker-specific features in-place.

    Must be called on the FULL dataset (before splitting) so that rolling
    windows have access to complete within-season history. The shift(1)
    prevents current-week leakage.

    Side-effect contract: per-group rolling aggregates require chronological
    row order, so the function physically reorders ``df`` by
    ``(player_id, season, week)`` via column-wise reassignment (no
    ``sort_values(..., inplace=True)``, which pandas discourages under CoW)
    before computing features. Row labels are realigned so ``df.index``
    reflects the sorted order — same as the previous in-place mutator left
    the frame.
    """
    df_sorted = df.sort_values(["player_id", "season", "week"])
    for col in df_sorted.columns:
        df[col] = df_sorted[col].values
    df.index = df_sorted.index

    # Rolling-feature input: signed fantasy total written by compute_targets.
    df["_k_total_pts"] = df["fantasy_points"]

    # Cross-season grouping (no season reset): kickers have stable multi-year
    # careers and small sample sizes per season, so cross-season windows provide
    # more signal than single-season windows. All other positions reset per-season.
    grp = ["player_id"]

    def _mean(col, window):
        return rolling_agg(df, col, grp, window=window, agg="mean", fill=0)

    def _sum(col, window):
        return rolling_agg(df, col, grp, window=window, fill=0)

    df["fg_attempts_L3"] = _mean("fg_att", 3)

    fg_made_roll = _sum("fg_made", 5)
    fg_att_roll = _sum("fg_att", 5)
    df["fg_accuracy_L5"] = safe_divide(fg_made_roll, fg_att_roll)

    df["pat_volume_L3"] = _mean("pat_att", 3)
    df["total_k_pts_L3"] = _mean("_k_total_pts", 3)

    # long_fg_rate_L3 (40+ yard FG proportion)
    df["_long_fg_att"] = (
        df["fg_made_40_49"].fillna(0)
        + df["fg_missed_40_49"].fillna(0)
        + df["fg_made_50_59"].fillna(0)
        + df["fg_missed_50_59"].fillna(0)
        + df["fg_made_60_"].fillna(0)
        + df["fg_missed_60_"].fillna(0)
    )
    long_roll = _sum("_long_fg_att", 3)
    total_att_roll = _sum("fg_att", 3)
    df["long_fg_rate_L3"] = safe_divide(long_roll, total_att_roll)

    # k_pts_trend (L3 - L8 momentum)
    short = _mean("_k_total_pts", 3)
    long = _mean("_k_total_pts", 8)
    df["k_pts_trend"] = (short - long).fillna(0)

    # k_pts_std_L3 (consistency) — keeps the inline transform: rolling std
    # needs ``min_periods=2`` (single-sample std is undefined), which our
    # helper's default doesn't cover.
    df["k_pts_std_L3"] = (
        df.groupby(grp)["_k_total_pts"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # PBP-derived features (Tier 1 + Tier 2)
    # ---------------------------------------------------------------

    df["avg_fg_distance_L3"] = _mean("avg_fg_distance", 3)
    df["avg_fg_prob_L3"] = _mean("avg_fg_prob", 3)

    long_made_roll = _sum("long_fg_made", 5)
    long_att_roll = _sum("long_fg_att", 5)
    df["fg_pct_40plus_L5"] = safe_divide(long_made_roll, long_att_roll)

    q4_made_roll = _sum("q4_fg_made", 5)
    q4_att_roll = _sum("q4_fg_att", 5)
    df["q4_fg_rate_L5"] = safe_divide(q4_made_roll, q4_att_roll)

    pat_made_roll = _sum("pat_made", 5)
    pat_att_roll = _sum("pat_att", 5)
    df["xp_accuracy_L5"] = safe_divide(pat_made_roll, pat_att_roll)

    # Clean up intermediate columns
    df.drop(
        columns=["_k_total_pts", "_long_fg_att"],
        inplace=True,
        errors="ignore",
    )


def add_specific_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """No-op — kicker features are pre-computed on the full dataset before splitting."""
    return train_df, val_df, test_df


def fill_nans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_feature_cols: list[str],
) -> tuple:
    """Fill NaNs in kicker feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, k_feature_cols)


def build_nested_kick_history(
    weekly_df: pd.DataFrame,
    kicks_df: pd.DataFrame,
    kick_stats: list[str],
    max_games: int = 17,
    max_kicks_per_game: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble nested per-game kick history aligned with weekly rows.

    For each row in `weekly_df`, gathers that kicker's prior-week kicks from
    `kicks_df` (same player_id, same season, `kicks.week < weekly.week`),
    bucketed by prior-game index. Outer dim is game-ordered newest-first
    (most recent game at outer index 0, older games at higher indices,
    right-padded — mirrors :func:`src.features.engineer.build_game_history_arrays`
    so the attention branch's positional embedding is recency-indexed). Within a
    game, kicks are sorted by ``play_id`` (PBP's monotonically increasing per-play
    sequence number) so within-game ordering is deterministic. Truncation keeps the
    most recent `max_games` games and, within each game, the last
    `max_kicks_per_game` kicks under the play_id ordering.

    Returns:
        X_history:  [n, max_games, max_kicks_per_game, kick_dim] float32, zero-padded
        outer_mask: [n, max_games]                                bool, True = real game
        inner_mask: [n, max_games, max_kicks_per_game]            bool, True = real kick
    """
    kick_dim = len(kick_stats)
    n = len(weekly_df)
    X_history = np.zeros((n, max_games, max_kicks_per_game, kick_dim), dtype=np.float32)
    outer_mask = np.zeros((n, max_games), dtype=bool)
    inner_mask = np.zeros((n, max_games, max_kicks_per_game), dtype=bool)

    if n == 0:
        return X_history, outer_mask, inner_mask

    missing = [s for s in kick_stats if s not in kicks_df.columns]
    if missing:
        raise KeyError(f"kicks_df missing columns: {missing}")

    weekly = weekly_df.reset_index(drop=True)

    if len(kicks_df) == 0:
        return X_history, outer_mask, inner_mask

    # play_id is the secondary sort when present (per-attempt ordering within a
    # game); fall back to whatever original row order kicks_df had if absent.
    sort_keys = ["player_id", "season", "week"]
    if "play_id" in kicks_df.columns:
        sort_keys.append("play_id")
    kicks_sorted = kicks_df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    kick_values = kicks_sorted[kick_stats].to_numpy(dtype=np.float32)
    np.nan_to_num(kick_values, copy=False, nan=0.0)

    # Pre-group: (pid, sea) -> (weeks_sorted[W], kick_indices_by_week[W])
    # where W = number of unique kick weeks for that player-season.
    kicks_by_week = kicks_sorted.groupby(["player_id", "season", "week"]).indices
    per_pid_sea: dict[tuple, tuple[np.ndarray, list]] = {}
    for (pid, sea, wk), kick_idx in kicks_by_week.items():
        entry = per_pid_sea.setdefault((pid, sea), ([], []))
        entry[0].append(wk)
        entry[1].append(kick_idx)
    # Sort each player-season's weeks ascending so searchsorted + slicing
    # below yield "all prior kick-weeks" in oldest-first order.
    for key, (weeks_list, idx_list) in per_pid_sea.items():
        order = np.argsort(np.asarray(weeks_list))
        per_pid_sea[key] = (
            np.asarray(weeks_list, dtype=int)[order],
            [idx_list[i] for i in order],
        )

    for (pid, sea), grp in weekly.groupby(["player_id", "season"], sort=False):
        prior_weeks_arr, prior_idx_list = per_pid_sea.get((pid, sea), (np.empty(0, dtype=int), []))
        grp_sorted = grp.sort_values("week", kind="stable")
        for wk, row_pos in zip(
            grp_sorted["week"].to_numpy(), grp_sorted.index.to_numpy(), strict=True
        ):
            cut = int(np.searchsorted(prior_weeks_arr, wk, side="left"))
            if cut == 0:
                continue
            start = max(0, cut - max_games)
            # reversed() so the most-recent prior game lands at outer index 0,
            # older games at higher indices (newest-first, right-padded). Within
            # a game the kick ordering below is unchanged.
            for g_idx, slot in enumerate(reversed(range(start, cut))):
                kick_idx = prior_idx_list[slot]
                if len(kick_idx) > max_kicks_per_game:
                    kick_idx = kick_idx[-max_kicks_per_game:]
                n_kicks = len(kick_idx)
                outer_mask[row_pos, g_idx] = True
                if n_kicks > 0:
                    X_history[row_pos, g_idx, :n_kicks] = kick_values[kick_idx]
                    inner_mask[row_pos, g_idx, :n_kicks] = True

    return X_history, outer_mask, inner_mask
