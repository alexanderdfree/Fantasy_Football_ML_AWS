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

    grp = df.groupby(["team", "season"])

    # --- Feature 1: sacks_L3 ---
    df["sacks_L3"] = grp["def_sacks"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # turnovers_L3 removed — exactly ints_L3 + fumble_rec_L3 (perfect linear dependency)

    # --- Feature 2: ints_L3 (INTs separated — secondary quality signal) ---
    df["ints_L3"] = grp["def_ints"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 4: fumble_rec_L3 (fumble recoveries — more stochastic) ---
    df["fumble_rec_L3"] = grp["def_fumble_rec"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 5: pts_allowed_L3 ---
    df["pts_allowed_L3"] = grp["points_allowed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 6: pts_allowed_L5 ---
    df["pts_allowed_L5"] = grp["points_allowed"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 7: dst_pts_L3 ---
    df["dst_pts_L3"] = grp["_dst_total_pts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 8: dst_pts_L5 ---
    df["dst_pts_L5"] = grp["_dst_total_pts"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 9: dst_pts_L8 (longer stability window) ---
    df["dst_pts_L8"] = grp["_dst_total_pts"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 10: sack_trend (L3 - L8) ---
    sack_short = grp["def_sacks"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    sack_long = grp["def_sacks"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["sack_trend"] = (sack_short - sack_long).fillna(0)

    # --- Feature 11: turnover_trend (L3 - L8) ---
    to_short = grp["_turnovers"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    to_long = grp["_turnovers"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["turnover_trend"] = (to_short - to_long).fillna(0)

    # --- Feature 12: pts_allowed_trend (L3 - L8, negative = improving defense) ---
    pa_short = grp["points_allowed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    pa_long = grp["points_allowed"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["pts_allowed_trend"] = (pa_short - pa_long).fillna(0)

    # --- Feature 13: pts_allowed_std_L3 (defensive consistency) ---
    df["pts_allowed_std_L3"] = grp["points_allowed"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    ).fillna(0)

    # --- Feature 14: dst_scoring_std_L3 (base scoring consistency) ---
    df["dst_scoring_std_L3"] = grp["defensive_scoring"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    ).fillna(0)

    # --- Feature 15: sacks_L5 (longer sack window for stability) ---
    df["sacks_L5"] = grp["def_sacks"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 16: pts_allowed_ewma (exponential weighting, faster adaptation) ---
    df["pts_allowed_ewma"] = grp["points_allowed"].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 17: dst_pts_ewma (total D/ST scoring, exponential) ---
    df["dst_pts_ewma"] = grp["_dst_total_pts"].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    ).fillna(0)

    # --- Feature 18: opp_scoring_L3 (short-window opponent quality) ---
    # NOTE: opp_scoring_L3 is computed in dst_data.py via opponent merge
    # to ensure correct opponent alignment. Placeholder column created here
    # if not already present.

    # --- Prior-season features (index-safe merge) ---
    prior = df.groupby(["team", "season"]).agg(
        prior_dst_pts=("_dst_total_pts", "mean"),
        prior_pts_allowed=("points_allowed", "mean"),
    ).reset_index()
    prior["season"] = prior["season"] + 1  # Align S-1 stats with season S
    prior.columns = ["team", "season", "prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"]
    df.drop(columns=["prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"],
            errors="ignore", inplace=True)
    # Merge preserving original index (avoids fragile .values assignment)
    orig_idx = df.index
    merged = df.reset_index().merge(prior, on=["team", "season"], how="left").set_index("index")
    df["prior_season_dst_pts_avg"] = merged.loc[orig_idx, "prior_season_dst_pts_avg"]
    df["prior_season_pts_allowed_avg"] = merged.loc[orig_idx, "prior_season_pts_allowed_avg"]

    # Clean up temp columns
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
