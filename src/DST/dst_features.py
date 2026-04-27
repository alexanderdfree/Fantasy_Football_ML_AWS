import pandas as pd

from src.DST.dst_config import DST_ALL_FEATURES
from src.shared.feature_build import fill_nans_with_train_means, rolling_agg


def get_dst_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the DST model."""
    return list(DST_ALL_FEATURES)


def compute_dst_features(df: pd.DataFrame) -> None:
    """Compute all D/ST features in-place.

    Must be called on the FULL dataset (before splitting) so that rolling
    windows and prior-season features have complete history.
    """
    df.sort_values(["team", "season", "week"], inplace=True)

    # Pre-compute D/ST fantasy points for rolling features.  We use the
    # tier-mapped ``fantasy_points`` column produced by compute_dst_targets
    # so the rolling window matches what the model is actually predicting.
    df["_dst_total_pts"] = df["fantasy_points"]

    # Pre-compute turnovers forced (INTs + fumble recoveries)
    df["_turnovers"] = df["def_ints"].fillna(0) + df["def_fumble_rec"].fillna(0)

    grp = ["team", "season"]

    def _mean(col, window):
        return rolling_agg(df, col, grp, window=window, agg="mean", fill=0)

    df["sacks_L3"] = _mean("def_sacks", 3)
    df["ints_L3"] = _mean("def_ints", 3)
    df["fumble_rec_L3"] = _mean("def_fumble_rec", 3)
    df["pts_allowed_L3"] = _mean("points_allowed", 3)
    df["pts_allowed_L5"] = _mean("points_allowed", 5)
    df["dst_pts_L3"] = _mean("_dst_total_pts", 3)
    df["dst_pts_L5"] = _mean("_dst_total_pts", 5)
    df["dst_pts_L8"] = _mean("_dst_total_pts", 8)

    df["sack_trend"] = (_mean("def_sacks", 3) - _mean("def_sacks", 8)).fillna(0)
    df["turnover_trend"] = (_mean("_turnovers", 3) - _mean("_turnovers", 8)).fillna(0)
    df["pts_allowed_trend"] = (_mean("points_allowed", 3) - _mean("points_allowed", 8)).fillna(0)

    # Rolling std for consistency — keeps the inline transform: the helper's
    # ``min_periods=1`` default would produce NaN for single-sample std,
    # while these features require ``min_periods=2``.
    df["pts_allowed_std_L3"] = (
        df.groupby(grp)["points_allowed"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
        .fillna(0)
    )

    # dst_scoring_std_L3 — derived from raw stats (post raw-target migration),
    # same linear combo ``defensive_production`` used to represent.
    df["_dst_defensive_production_tmp"] = (
        df["def_sacks"].fillna(0)
        + df["def_ints"].fillna(0) * 2
        + df["def_fumble_rec"].fillna(0) * 2
        + df["def_fumbles_forced"].fillna(0)
        + df["def_safeties"].fillna(0) * 2
    )
    df["dst_scoring_std_L3"] = (
        df.groupby(grp)["_dst_defensive_production_tmp"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
        .fillna(0)
    )
    df.drop(columns=["_dst_defensive_production_tmp"], inplace=True)

    df["sacks_L5"] = _mean("def_sacks", 5)

    # EWMA features — helper only covers rolling, not exponential smoothing,
    # so these stay as inline transforms.
    df["pts_allowed_ewma"] = (
        df.groupby(grp)["points_allowed"]
        .transform(lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
        .fillna(0)
    )
    df["dst_pts_ewma"] = (
        df.groupby(grp)["_dst_total_pts"]
        .transform(lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
        .fillna(0)
    )

    df["forced_fumbles_L3"] = _mean("def_fumbles_forced", 3)
    df["blocked_kicks_L5"] = _mean("def_blocked_kicks", 5)
    df["yards_allowed_L3"] = _mean("yards_allowed", 3)
    df["yards_allowed_L5"] = _mean("yards_allowed", 5)

    df["yards_allowed_ewma"] = (
        df.groupby(grp)["yards_allowed"]
        .transform(lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
        .fillna(0)
    )

    # opp_scoring_L3 is computed in dst_data.py via opponent merge to ensure
    # correct opponent alignment; just make sure it survives this pass.

    # --- Prior-season features (index-safe merge) ---
    prior = (
        df.groupby(["team", "season"])
        .agg(
            prior_dst_pts=("_dst_total_pts", "mean"),
            prior_pts_allowed=("points_allowed", "mean"),
        )
        .reset_index()
    )
    prior["season"] = prior["season"] + 1  # Align S-1 stats with season S
    prior.columns = ["team", "season", "prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"]
    df.drop(
        columns=["prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"],
        errors="ignore",
        inplace=True,
    )
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
    return fill_nans_with_train_means(train_df, val_df, test_df, dst_feature_cols)
