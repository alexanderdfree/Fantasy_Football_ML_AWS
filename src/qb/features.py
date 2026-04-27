import pandas as pd

from src.features.engineer import flatten_include_features
from src.qb.config import QB_INCLUDE_FEATURES
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)


def get_qb_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the QB model."""
    return flatten_include_features(QB_INCLUDE_FEATURES)


def add_qb_specific_features(train_df, val_df, test_df):
    """Add QB-specific engineered features (see QB_SPECIFIC_FEATURES) to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_qb_features(df)
    return train_df, val_df, test_df


def _compute_qb_features(df: pd.DataFrame) -> None:
    """Compute all QB-specific features (see QB_SPECIFIC_FEATURES) in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    grp = ["player_id", "season"]

    def _sum(col):
        return rolling_agg(df, col, grp, window=3)

    completions_roll = _sum("completions")
    attempts_roll = _sum("attempts")
    pass_yds_roll = _sum("passing_yards")
    pass_tds_roll = _sum("passing_tds")
    ints_roll = _sum("interceptions")
    sacks_roll = _sum("sacks")
    rush_yds_roll = _sum("rushing_yards")
    pass_epa_roll = _sum("passing_epa")
    air_yds_roll = _sum("passing_air_yards")
    carries_roll = _sum("carries")
    pass_first_downs_roll = _sum("passing_first_downs")
    rush_first_downs_roll = _sum("rushing_first_downs")
    rush_epa_roll = _sum("rushing_epa")
    pass_yac_roll = _sum("passing_yards_after_catch")
    sack_yds_roll = _sum("sack_yards")

    dropbacks = attempts_roll + sacks_roll

    df["completion_pct_L3"] = safe_divide(completions_roll, attempts_roll)
    df["yards_per_attempt_L3"] = safe_divide(pass_yds_roll, attempts_roll)
    df["td_rate_L3"] = safe_divide(pass_tds_roll, attempts_roll)
    df["int_rate_L3"] = safe_divide(ints_roll, attempts_roll)
    df["sack_rate_L3"] = safe_divide(sacks_roll, dropbacks)

    # Dual-threat indicator — share of total yards that come from rushing.
    total_yds = pass_yds_roll + rush_yds_roll
    df["qb_rushing_share_L3"] = safe_divide(rush_yds_roll, total_yds)

    df["passing_epa_per_dropback_L3"] = safe_divide(pass_epa_roll, dropbacks)
    df["deep_ball_rate_L3"] = safe_divide(air_yds_roll, attempts_roll)
    df["pass_first_down_rate_L3"] = safe_divide(pass_first_downs_roll, attempts_roll)
    df["rushing_epa_per_carry_L3"] = safe_divide(rush_epa_roll, carries_roll)
    df["rush_first_down_rate_L3"] = safe_divide(rush_first_downs_roll, carries_roll)
    df["yac_rate_L3"] = safe_divide(pass_yac_roll, pass_yds_roll)
    df["sack_damage_per_dropback_L3"] = safe_divide(sack_yds_roll, dropbacks)


def fill_qb_nans(train_df, val_df, test_df, qb_feature_cols):
    """Fill NaNs in QB-specific feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, qb_feature_cols)
