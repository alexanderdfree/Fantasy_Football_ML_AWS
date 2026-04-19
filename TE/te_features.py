import numpy as np
import pandas as pd

from src.features.engineer import flatten_include_features
from TE.te_config import TE_INCLUDE_FEATURES
from TE.te_data import compute_team_te_totals


def get_te_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the TE model."""
    return flatten_include_features(TE_INCLUDE_FEATURES)


def add_te_specific_features(train_df, val_df, test_df):
    """Add 8 TE-specific engineered features to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_te_features(df)
    return train_df, val_df, test_df


def _compute_te_features(df: pd.DataFrame) -> None:
    """Compute all 8 TE-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    def _roll_sum(col):
        return df.groupby(["player_id", "season"])[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        )

    recv_yds_roll = _roll_sum("receiving_yards")
    rec_roll = _roll_sum("receptions")
    tgt_roll = _roll_sum("targets")
    yac_roll = _roll_sum("receiving_yards_after_catch")
    recv_epa_roll = _roll_sum("receiving_epa")
    recv_fd_roll = _roll_sum("receiving_first_downs")
    air_yds_roll = _roll_sum("receiving_air_yards")
    recv_tds_roll = _roll_sum("receiving_tds")

    # 1. yards_per_reception_L3
    df["yards_per_reception_L3"] = (recv_yds_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "yards_per_reception_L3"] = 0

    # 2. reception_rate_L3
    df["reception_rate_L3"] = (rec_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "reception_rate_L3"] = 0

    # 3. yac_per_reception_L3
    df["yac_per_reception_L3"] = (yac_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "yac_per_reception_L3"] = 0

    # 4. team_te_target_share_L3
    team_te_totals = compute_team_te_totals(df)
    df_merged = df.merge(team_te_totals, on=["recent_team", "season", "week"], how="left")
    player_tgt_roll = df_merged.groupby(["player_id", "season"])["targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    team_te_tgt_roll = df_merged.groupby(["player_id", "season"])["team_te_targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    share = (player_tgt_roll / team_te_tgt_roll).fillna(0)
    share[team_te_tgt_roll.values == 0] = 0
    df["team_te_target_share_L3"] = share.values

    # 5. receiving_epa_per_target_L3
    df["receiving_epa_per_target_L3"] = (recv_epa_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "receiving_epa_per_target_L3"] = 0

    # 6. receiving_first_down_rate_L3
    df["receiving_first_down_rate_L3"] = (recv_fd_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "receiving_first_down_rate_L3"] = 0

    # 7. air_yards_per_target_L3
    df["air_yards_per_target_L3"] = (air_yds_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "air_yards_per_target_L3"] = 0

    # 8. td_rate_per_target_L3 (TE TD dependency)
    df["td_rate_per_target_L3"] = (recv_tds_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "td_rate_per_target_L3"] = 0


def fill_te_nans(train_df, val_df, test_df, te_feature_cols):
    """Fill NaNs in TE-specific feature columns using training set statistics."""
    for split_df in [train_df, val_df, test_df]:
        split_df[te_feature_cols] = split_df[te_feature_cols].replace([np.inf, -np.inf], np.nan)
    train_means = train_df[te_feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in te_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])
    return train_df, val_df, test_df
