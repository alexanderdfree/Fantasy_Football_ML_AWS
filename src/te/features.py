import pandas as pd

from src.features.engineer import flatten_include_features
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)
from src.te.config import INCLUDE_FEATURES
from src.te.data import compute_team_te_totals


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the TE model."""
    return flatten_include_features(INCLUDE_FEATURES)


def add_specific_features(train_df, val_df, test_df):
    """Add 8 TE-specific engineered features to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_features(df)
    return train_df, val_df, test_df


def _compute_features(df: pd.DataFrame) -> None:
    """Compute all 8 TE-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    grp = ["player_id", "season"]

    def _sum(col):
        return rolling_agg(df, col, grp, window=3)

    recv_yds_roll = _sum("receiving_yards")
    rec_roll = _sum("receptions")
    tgt_roll = _sum("targets")
    yac_roll = _sum("receiving_yards_after_catch")
    recv_epa_roll = _sum("receiving_epa")
    recv_fd_roll = _sum("receiving_first_downs")
    air_yds_roll = _sum("receiving_air_yards")
    recv_tds_roll = _sum("receiving_tds")

    df["yards_per_reception_L3"] = safe_divide(recv_yds_roll, rec_roll)
    df["reception_rate_L3"] = safe_divide(rec_roll, tgt_roll)
    df["yac_per_reception_L3"] = safe_divide(yac_roll, rec_roll)

    team_te_totals = compute_team_te_totals(df)
    df_merged = df.merge(team_te_totals, on=["recent_team", "season", "week"], how="left")
    player_tgt_roll = rolling_agg(df_merged, "targets", grp, window=3)
    team_te_tgt_roll = rolling_agg(df_merged, "team_te_targets", grp, window=3)
    df["team_te_target_share_L3"] = safe_divide(player_tgt_roll, team_te_tgt_roll).values

    df["receiving_epa_per_target_L3"] = safe_divide(recv_epa_roll, tgt_roll)
    df["receiving_first_down_rate_L3"] = safe_divide(recv_fd_roll, rec_roll)
    df["air_yards_per_target_L3"] = safe_divide(air_yds_roll, tgt_roll)
    df["td_rate_per_target_L3"] = safe_divide(recv_tds_roll, tgt_roll)


def fill_nans(train_df, val_df, test_df, te_feature_cols):
    """Fill NaNs in TE-specific feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, te_feature_cols)
