import numpy as np
import pandas as pd
from src.features.engineer import get_feature_columns
from WR.wr_config import WR_DROP_FEATURES, WR_SPECIFIC_FEATURES
from WR.wr_data import compute_team_wr_totals


def get_wr_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the WR model."""
    general_cols = get_feature_columns()
    wr_cols = [c for c in general_cols if c not in WR_DROP_FEATURES]
    wr_cols.extend(WR_SPECIFIC_FEATURES)
    return wr_cols


def add_wr_specific_features(train_df, val_df, test_df):
    """Add 8 WR-specific engineered features to each split."""
    for df in [train_df, val_df, test_df]:
        _compute_wr_features(df)
    return train_df, val_df, test_df


def _compute_wr_features(df: pd.DataFrame) -> None:
    """Compute all 8 WR-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    def _roll_sum(col):
        return df.groupby(["player_id", "season"])[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        )

    recv_yds_roll = _roll_sum("receiving_yards")
    rec_roll = _roll_sum("receptions")
    tgt_roll = _roll_sum("targets")
    air_yds_roll = _roll_sum("receiving_air_yards")
    yac_roll = _roll_sum("receiving_yards_after_catch")
    recv_epa_roll = _roll_sum("receiving_epa")
    recv_fd_roll = _roll_sum("receiving_first_downs")

    # 1. yards_per_reception_L3
    df["yards_per_reception_L3"] = (recv_yds_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "yards_per_reception_L3"] = 0

    # 2. yards_per_target_L3
    df["yards_per_target_L3"] = (recv_yds_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "yards_per_target_L3"] = 0

    # 3. reception_rate_L3
    df["reception_rate_L3"] = (rec_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "reception_rate_L3"] = 0

    # 4. air_yards_per_target_L3
    df["air_yards_per_target_L3"] = (air_yds_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "air_yards_per_target_L3"] = 0

    # 5. yac_per_reception_L3
    df["yac_per_reception_L3"] = (yac_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "yac_per_reception_L3"] = 0

    # 6. team_wr_target_share_L3
    team_wr_totals = compute_team_wr_totals(df)
    df_merged = df.merge(team_wr_totals, on=["recent_team", "season", "week"], how="left")
    player_tgt_roll = df_merged.groupby(["player_id", "season"])["targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    team_wr_tgt_roll = df_merged.groupby(["player_id", "season"])["team_wr_targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    share = (player_tgt_roll / team_wr_tgt_roll).fillna(0)
    share[team_wr_tgt_roll.values == 0] = 0
    df["team_wr_target_share_L3"] = share.values

    # 7. receiving_epa_per_target_L3
    df["receiving_epa_per_target_L3"] = (recv_epa_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "receiving_epa_per_target_L3"] = 0

    # 8. receiving_first_down_rate_L3
    df["receiving_first_down_rate_L3"] = (recv_fd_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "receiving_first_down_rate_L3"] = 0


def fill_wr_nans(train_df, val_df, test_df, wr_feature_cols):
    """Fill NaNs in WR-specific feature columns using training set statistics."""
    for split_df in [train_df, val_df, test_df]:
        split_df[wr_feature_cols] = split_df[wr_feature_cols].replace([np.inf, -np.inf], np.nan)
    train_means = train_df[wr_feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in wr_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])
    return train_df, val_df, test_df
