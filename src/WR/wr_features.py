import pandas as pd

from src.features.engineer import flatten_include_features
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)
from src.WR.wr_config import WR_INCLUDE_FEATURES
from src.WR.wr_data import compute_team_wr_totals


def get_wr_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the WR model."""
    return flatten_include_features(WR_INCLUDE_FEATURES)


def add_wr_specific_features(train_df, val_df, test_df):
    """Add 8 WR-specific engineered features to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_wr_features(df)
    return train_df, val_df, test_df


def _compute_wr_features(df: pd.DataFrame) -> None:
    """Compute all 8 WR-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    grp = ["player_id", "season"]

    def _sum(col):
        return rolling_agg(df, col, grp, window=3)

    recv_yds_roll = _sum("receiving_yards")
    rec_roll = _sum("receptions")
    tgt_roll = _sum("targets")
    air_yds_roll = _sum("receiving_air_yards")
    yac_roll = _sum("receiving_yards_after_catch")
    recv_epa_roll = _sum("receiving_epa")
    recv_fd_roll = _sum("receiving_first_downs")

    df["yards_per_reception_L3"] = safe_divide(recv_yds_roll, rec_roll)
    df["yards_per_target_L3"] = safe_divide(recv_yds_roll, tgt_roll)
    df["reception_rate_L3"] = safe_divide(rec_roll, tgt_roll)
    df["air_yards_per_target_L3"] = safe_divide(air_yds_roll, tgt_roll)
    df["yac_per_reception_L3"] = safe_divide(yac_roll, rec_roll)

    team_wr_totals = compute_team_wr_totals(df)
    df_merged = df.merge(team_wr_totals, on=["recent_team", "season", "week"], how="left")
    player_tgt_roll = rolling_agg(df_merged, "targets", grp, window=3)
    team_wr_tgt_roll = rolling_agg(df_merged, "team_wr_targets", grp, window=3)
    df["team_wr_target_share_L3"] = safe_divide(player_tgt_roll, team_wr_tgt_roll).values

    df["receiving_epa_per_target_L3"] = safe_divide(recv_epa_roll, tgt_roll)
    df["receiving_first_down_rate_L3"] = safe_divide(recv_fd_roll, rec_roll)


def fill_wr_nans(train_df, val_df, test_df, wr_feature_cols):
    """Fill NaNs in WR-specific feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, wr_feature_cols)
