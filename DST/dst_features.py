import numpy as np
import pandas as pd
from DST.dst_config import DST_ALL_FEATURES, DST_SPECIFIC_FEATURES


def get_dst_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the DST model."""
    return list(DST_ALL_FEATURES)


def compute_dst_features(df: pd.DataFrame) -> None:
    """Compute all D/ST features in-place.

    Must be called on the FULL dataset (before splitting) so that rolling
    windows and prior-season features have complete history.
    """
    df.sort_values(["team", "season", "week"], inplace=True)

    # Pre-compute D/ST fantasy points for rolling features
    df["_dst_total_pts"] = (
        df["defensive_scoring"] + df["td_points"] + df["pts_allowed_bonus"]
    )

    # Pre-compute turnovers forced
    df["_turnovers"] = df["def_ints"].fillna(0) + df["def_fumble_rec"].fillna(0)

    # --- Feature 1: sacks_L3 ---
    df["sacks_L3"] = df.groupby(["team", "season"])["def_sacks"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 2: turnovers_L3 ---
    df["turnovers_L3"] = df.groupby(["team", "season"])["_turnovers"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 3: pts_allowed_L3 ---
    df["pts_allowed_L3"] = df.groupby(["team", "season"])["points_allowed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 4: pts_allowed_L5 ---
    df["pts_allowed_L5"] = df.groupby(["team", "season"])["points_allowed"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 5: dst_pts_L3 ---
    df["dst_pts_L3"] = df.groupby(["team", "season"])["_dst_total_pts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 6: dst_pts_L5 ---
    df["dst_pts_L5"] = df.groupby(["team", "season"])["_dst_total_pts"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 7: sack_trend (L3 - L8) ---
    short = df.groupby(["team", "season"])["def_sacks"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    long = df.groupby(["team", "season"])["def_sacks"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["sack_trend"] = (short - long).fillna(0)

    # --- Feature 8: pts_allowed_std_L3 (defensive consistency) ---
    df["pts_allowed_std_L3"] = df.groupby(["team", "season"])[
        "points_allowed"
    ].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    ).fillna(0)

    # --- Prior-season features ---
    prior = df.groupby(["team", "season"]).agg(
        prior_dst_pts=("_dst_total_pts", "mean"),
        prior_pts_allowed=("points_allowed", "mean"),
    ).reset_index()
    prior["season"] = prior["season"] + 1  # Align S-1 stats with season S
    prior.columns = ["team", "season", "prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"]
    df.drop(columns=["prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"],
            errors="ignore", inplace=True)
    df_len = len(df)
    df_reset = df.reset_index(drop=True)
    merged = df_reset.merge(prior, on=["team", "season"], how="left")
    df["prior_season_dst_pts_avg"] = merged["prior_season_dst_pts_avg"].values
    df["prior_season_pts_allowed_avg"] = merged["prior_season_pts_allowed_avg"].values

    # Fill prior-season NaNs with global training means (handled in fill_nans)
    # Clean up
    df.drop(columns=["_dst_total_pts", "_turnovers"], inplace=True, errors="ignore")


def add_dst_specific_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """No-op — D/ST features are pre-computed on the full dataset before splitting."""
    return train_df, val_df, test_df


def fill_dst_nans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dst_feature_cols: list[str],
) -> tuple:
    """Fill NaNs in D/ST feature columns using training set statistics."""
    for split_df in [train_df, val_df, test_df]:
        split_df[dst_feature_cols] = split_df[dst_feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    train_means = train_df[dst_feature_cols].mean()

    for split_df in [train_df, val_df, test_df]:
        for col in dst_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])

    return train_df, val_df, test_df
