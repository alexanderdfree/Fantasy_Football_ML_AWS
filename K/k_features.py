import numpy as np
import pandas as pd
from K.k_config import K_ALL_FEATURES, K_SPECIFIC_FEATURES


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

    # Pre-compute raw kicker total points (for rolling features)
    df["_k_total_pts"] = (
        df["fg_points"] + df["pat_points"] + df["miss_penalty"]
    )

    # Cross-season grouping (no season reset): kickers have stable multi-year
    # careers and small sample sizes per season, so cross-season windows provide
    # more signal than single-season windows. All other positions reset per-season.
    grp = ["player_id"]

    # --- Feature 1: fg_attempts_L3 ---
    df["fg_attempts_L3"] = df.groupby(grp)["fg_att"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

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
    df["pat_volume_L3"] = df.groupby(grp)["pat_att"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 4: total_k_pts_L3 ---
    df["total_k_pts_L3"] = df.groupby(grp)[
        "_k_total_pts"
    ].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 6: long_fg_rate_L3 (40+ yard FG proportion) ---
    df["_long_fg_att"] = (
        df["fg_made_40_49"].fillna(0) + df["fg_missed_40_49"].fillna(0)
        + df["fg_made_50_59"].fillna(0) + df["fg_missed_50_59"].fillna(0)
        + df["fg_made_60_"].fillna(0) + df["fg_missed_60_"].fillna(0)
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
    df["k_pts_std_L3"] = df.groupby(grp)["_k_total_pts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    ).fillna(0)

    # ---------------------------------------------------------------
    # PBP-derived features (Tier 1 + Tier 2)
    # ---------------------------------------------------------------

    # --- Tier 1: avg_fg_distance_L3 ---
    df["avg_fg_distance_L3"] = df.groupby(grp)["avg_fg_distance"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Tier 1: avg_fg_prob_L3 ---
    df["avg_fg_prob_L3"] = df.groupby(grp)["avg_fg_prob"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Tier 2: fg_pct_40plus_L5 (make% on 40+ yard FGs) ---
    long_made_roll = df.groupby(grp)["long_fg_made"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    long_att_roll = df.groupby(grp)["long_fg_att"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df["fg_pct_40plus_L5"] = (long_made_roll / long_att_roll).fillna(0)
    df.loc[long_att_roll == 0, "fg_pct_40plus_L5"] = 0

    # --- Tier 2: q4_fg_rate_L5 (make% in 4th quarter + OT) ---
    q4_made_roll = df.groupby(grp)["q4_fg_made"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    q4_att_roll = df.groupby(grp)["q4_fg_att"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df["q4_fg_rate_L5"] = (q4_made_roll / q4_att_roll).fillna(0)
    df.loc[q4_att_roll == 0, "q4_fg_rate_L5"] = 0

    # --- Tier 2: xp_accuracy_L5 (PAT make%) ---
    pat_made_roll = df.groupby(grp)["pat_made"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    pat_att_roll = df.groupby(grp)["pat_att"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()
    ).fillna(0)
    df["xp_accuracy_L5"] = (pat_made_roll / pat_att_roll).fillna(0)
    df.loc[pat_att_roll == 0, "xp_accuracy_L5"] = 0

    # Clean up intermediate columns
    df.drop(columns=["_k_total_pts", "_long_fg_att"], inplace=True, errors="ignore")


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
        split_df[k_feature_cols] = split_df[k_feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    train_means = train_df[k_feature_cols].mean()

    for split_df in [train_df, val_df, test_df]:
        for col in k_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])

    return train_df, val_df, test_df
