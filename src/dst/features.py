import pandas as pd

from src.dst.config import POSITION_CONFIG
from src.shared.feature_build import fill_nans_with_train_means, rolling_agg


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the DST model."""
    return list(POSITION_CONFIG.all_features)


def compute_features(df: pd.DataFrame) -> None:
    """Compute all D/ST features in-place.

    Must be called on the FULL dataset (before splitting) so that rolling
    windows and prior-season features have complete history.

    Side-effect contract: per-group rolling aggregates require chronological
    row order, so the function physically reorders ``df`` by
    ``(team, season, week)`` via column-wise reassignment (no
    ``sort_values(..., inplace=True)``, which pandas discourages under CoW)
    before computing features. Row labels are realigned so ``df.index``
    reflects the sorted order — same as the previous in-place mutator left
    the frame.
    """
    # Guard with an explicit raise (not ``assert``, which ``python -O``
    # strips): ``_dst_total_pts`` below reads ``fantasy_points``, which is
    # written by compute_targets. Missing it means scoring hasn't run, and a
    # bare KeyError on the assignment would be opaque.
    if "fantasy_points" not in df.columns:
        raise KeyError(
            "DST compute_features requires compute_targets to have run first "
            "(missing 'fantasy_points' column)."
        )
    df_sorted = df.sort_values(["team", "season", "week"])
    for col in df_sorted.columns:
        df[col] = df_sorted[col].values
    df.index = df_sorted.index

    # Pre-compute D/ST fantasy points for rolling features.  We use the
    # tier-mapped ``fantasy_points`` column produced by compute_targets
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
    # same linear combo ``defensive_production`` used to represent. Point
    # weights mirror the DST scoring in
    # src.shared.aggregate_targets.predictions_to_fantasy_points: sacks×1,
    # INT×2, fumble_rec×2, FF×1, safety×2, def_TD×6, blocked_kick×2, ST_TD×6.
    # def_tds / def_blocked_kicks / special_teams_tds were omitted originally,
    # so the volatility feature understated variance in any 3-game window where
    # a defensive/ST TD or blocked kick scored. (#437)
    df["_dst_defensive_production_tmp"] = (
        df["def_sacks"].fillna(0)
        + df["def_ints"].fillna(0) * 2
        + df["def_fumble_rec"].fillna(0) * 2
        + df["def_fumbles_forced"].fillna(0)
        + df["def_safeties"].fillna(0) * 2
        + df["def_tds"].fillna(0) * 6
        + df["def_blocked_kicks"].fillna(0) * 2
        + df["special_teams_tds"].fillna(0) * 6
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

    # opp_scoring_L3 is computed in src/dst/data.py via opponent merge to ensure
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


def add_specific_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_train: pd.DataFrame | None = None,
) -> tuple:
    """No-op — D/ST features are pre-computed on the full dataset before splitting.

    ``full_train`` is accepted for the shared ``add_features_fn`` contract
    (#574/#531) but unused.
    """
    return train_df, val_df, test_df


def fill_nans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dst_feature_cols: list[str],
) -> tuple:
    """Fill NaNs in D/ST feature columns using training set statistics."""
    # prior_season_* are contextual features (in all_features, so the model uses
    # them) but absent from _SPECIFIC_FEATURES — the column set the pipeline passes
    # here — so without this they skip the leak-safe train-mean fill and fall through
    # to build_position_features' catch-all .fillna(0). That 0 is a structurally-
    # impossible "0 fantasy pts / 0 PA over a whole season" for relocated teams
    # (LV 2020 / LAC 2017 / LAR 2016 have no prior-season row under their new code)
    # and every 2012 row (no 2011 history) — a strong negative z-score post-scaler.
    # Give them the same train-only-mean treatment as the rest. (#856)
    prior_cols = [
        c
        for c in ("prior_season_dst_pts_avg", "prior_season_pts_allowed_avg")
        if c in train_df.columns and c not in dst_feature_cols
    ]
    return fill_nans_with_train_means(train_df, val_df, test_df, [*dst_feature_cols, *prior_cols])
