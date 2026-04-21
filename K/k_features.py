import numpy as np
import pandas as pd

from K.k_config import K_ALL_FEATURES


def get_k_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the K model."""
    return list(K_ALL_FEATURES)


def compute_k_features(df: pd.DataFrame) -> None:
    """Compute all kicker-specific features in-place.

    Must be called on the FULL dataset (before splitting) so that rolling
    windows have access to complete within-season history. The shift(1)
    prevents current-week leakage.
    """
    df.sort_values(["player_id", "season", "week"], inplace=True)

    # Rolling-feature input: signed fantasy total written by compute_k_targets.
    df["_k_total_pts"] = df["fantasy_points"]

    # Cross-season grouping (no season reset): kickers have stable multi-year
    # careers and small sample sizes per season, so cross-season windows provide
    # more signal than single-season windows. All other positions reset per-season.
    grp = ["player_id"]

    # --- Feature 1: fg_attempts_L3 ---
    df["fg_attempts_L3"] = (
        df.groupby(grp)["fg_att"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # --- Feature 2: fg_accuracy_L5 ---
    fg_made_roll = df.groupby(grp)["fg_made"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    fg_att_roll = df.groupby(grp)["fg_att"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    )
    df["fg_accuracy_L5"] = (fg_made_roll / fg_att_roll).fillna(0)
    df.loc[fg_att_roll == 0, "fg_accuracy_L5"] = 0

    # --- Feature 3: pat_volume_L3 ---
    df["pat_volume_L3"] = (
        df.groupby(grp)["pat_att"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # --- Feature 4: total_k_pts_L3 ---
    df["total_k_pts_L3"] = (
        df.groupby(grp)["_k_total_pts"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # --- Feature 6: long_fg_rate_L3 (40+ yard FG proportion) ---
    df["_long_fg_att"] = (
        df["fg_made_40_49"].fillna(0)
        + df["fg_missed_40_49"].fillna(0)
        + df["fg_made_50_59"].fillna(0)
        + df["fg_missed_50_59"].fillna(0)
        + df["fg_made_60_"].fillna(0)
        + df["fg_missed_60_"].fillna(0)
    )
    long_roll = df.groupby(grp)["_long_fg_att"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    total_att_roll = df.groupby(grp)["fg_att"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["long_fg_rate_L3"] = (long_roll / total_att_roll).fillna(0)
    df.loc[total_att_roll == 0, "long_fg_rate_L3"] = 0

    # --- Feature 7: k_pts_trend (L3 - L8 momentum) ---
    short = df.groupby(grp)["_k_total_pts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    long = df.groupby(grp)["_k_total_pts"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["k_pts_trend"] = (short - long).fillna(0)

    # --- Feature 8: k_pts_std_L3 (consistency) ---
    df["k_pts_std_L3"] = (
        df.groupby(grp)["_k_total_pts"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # PBP-derived features (Tier 1 + Tier 2)
    # ---------------------------------------------------------------

    # --- Tier 1: avg_fg_distance_L3 ---
    df["avg_fg_distance_L3"] = (
        df.groupby(grp)["avg_fg_distance"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # --- Tier 1: avg_fg_prob_L3 ---
    df["avg_fg_prob_L3"] = (
        df.groupby(grp)["avg_fg_prob"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # --- Tier 2: fg_pct_40plus_L5 (make% on 40+ yard FGs) ---
    long_made_roll = (
        df.groupby(grp)["long_fg_made"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    long_att_roll = (
        df.groupby(grp)["long_fg_att"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    df["fg_pct_40plus_L5"] = (long_made_roll / long_att_roll).fillna(0)
    df.loc[long_att_roll == 0, "fg_pct_40plus_L5"] = 0

    # --- Tier 2: q4_fg_rate_L5 (make% in 4th quarter + OT) ---
    q4_made_roll = (
        df.groupby(grp)["q4_fg_made"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    q4_att_roll = (
        df.groupby(grp)["q4_fg_att"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    df["q4_fg_rate_L5"] = (q4_made_roll / q4_att_roll).fillna(0)
    df.loc[q4_att_roll == 0, "q4_fg_rate_L5"] = 0

    # --- Tier 2: xp_accuracy_L5 (PAT make%) ---
    pat_made_roll = (
        df.groupby(grp)["pat_made"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    pat_att_roll = (
        df.groupby(grp)["pat_att"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        .fillna(0)
    )
    df["xp_accuracy_L5"] = (pat_made_roll / pat_att_roll).fillna(0)
    df.loc[pat_att_roll == 0, "xp_accuracy_L5"] = 0

    # ---------------------------------------------------------------
    # L1 (shift-1) equivalents — attention NN's static branch consumes
    # these via K_ATTN_STATIC_FEATURES. They live in `df` but are NOT
    # listed in K_ALL_FEATURES, so Ridge and the base NN never see them.
    # Skip L1 versions of k_pts_trend (needs two windows) and k_pts_std
    # (undefined at L1).
    # ---------------------------------------------------------------
    _shift = lambda col: df.groupby(grp)[col].shift(1).fillna(0)  # noqa: E731

    df["fg_attempts_L1"] = _shift("fg_att")
    df["pat_volume_L1"] = _shift("pat_att")
    df["total_k_pts_L1"] = df.groupby(grp)["fantasy_points"].shift(1).fillna(0)
    df["avg_fg_distance_L1"] = _shift("avg_fg_distance")
    df["avg_fg_prob_L1"] = _shift("avg_fg_prob")

    # Ratio-valued L1 features — previous game's made/attempts. Zero-denom
    # guard mirrors the L5 ratios above (k_features.py:43-44 pattern).
    def _ratio_L1(num_col: str, den_col: str) -> pd.Series:
        num = df.groupby(grp)[num_col].shift(1)
        den = df.groupby(grp)[den_col].shift(1)
        out = (num / den).fillna(0)
        out = out.mask(den.fillna(0) == 0, 0)
        return out

    df["fg_accuracy_L1"] = _ratio_L1("fg_made", "fg_att")

    # long_fg_rate_L1: last game's (40+ attempts) / (total attempts).
    _long_fg_att_all = (
        df["fg_made_40_49"].fillna(0)
        + df["fg_missed_40_49"].fillna(0)
        + df["fg_made_50_59"].fillna(0)
        + df["fg_missed_50_59"].fillna(0)
        + df["fg_made_60_"].fillna(0)
        + df["fg_missed_60_"].fillna(0)
    )
    df["_long_fg_att_all"] = _long_fg_att_all
    df["long_fg_rate_L1"] = _ratio_L1("_long_fg_att_all", "fg_att")

    df["fg_pct_40plus_L1"] = _ratio_L1("long_fg_made", "long_fg_att")
    df["q4_fg_rate_L1"] = _ratio_L1("q4_fg_made", "q4_fg_att")
    df["xp_accuracy_L1"] = _ratio_L1("pat_made", "pat_att")

    # Clean up intermediate columns
    df.drop(
        columns=["_k_total_pts", "_long_fg_att", "_long_fg_att_all"],
        inplace=True,
        errors="ignore",
    )


def add_k_specific_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """No-op — kicker features are pre-computed on the full dataset before splitting."""
    return train_df, val_df, test_df


def fill_k_nans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_feature_cols: list[str],
) -> tuple:
    """Fill NaNs in kicker feature columns using training set statistics."""
    for split_df in [train_df, val_df, test_df]:
        split_df[k_feature_cols] = split_df[k_feature_cols].replace([np.inf, -np.inf], np.nan)

    train_means = train_df[k_feature_cols].mean()

    for split_df in [train_df, val_df, test_df]:
        for col in k_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])

    return train_df, val_df, test_df


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
    bucketed by prior-game index. Outer dim is game-ordered oldest-first
    (most recent game in the last real slot); inner dim preserves kick order
    within each game. Truncation keeps the most recent `max_games` games and
    the most recent `max_kicks_per_game` kicks per game.

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

    kicks_sorted = kicks_df.sort_values(["player_id", "season", "week"], kind="stable").reset_index(
        drop=True
    )
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
            for g_idx, slot in enumerate(range(start, cut)):
                kick_idx = prior_idx_list[slot]
                if len(kick_idx) > max_kicks_per_game:
                    kick_idx = kick_idx[-max_kicks_per_game:]
                n_kicks = len(kick_idx)
                outer_mask[row_pos, g_idx] = True
                if n_kicks > 0:
                    X_history[row_pos, g_idx, :n_kicks] = kick_values[kick_idx]
                    inner_mask[row_pos, g_idx, :n_kicks] = True

    return X_history, outer_mask, inner_mask
