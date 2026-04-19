import numpy as np
import pandas as pd

from RB.rb_config import RB_INCLUDE_FEATURES
from RB.rb_data import compute_team_rb_totals
from src.features.engineer import flatten_include_features


def get_rb_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the RB model."""
    return flatten_include_features(RB_INCLUDE_FEATURES)


def add_rb_specific_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """Add 14 RB-specific engineered features to each split.

    Features are computed independently per split to prevent leakage.
    Team RB totals are computed from each split's own data.
    """
    # Career carries: cumulative across all seasons, shifted to prevent leakage.
    # Must combine splits to capture cross-season history (e.g., 2024 val needs 2018-2023 carries).
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "week"])
    combined["career_carries"] = (
        combined.groupby("player_id")["carries"]
        .transform(lambda x: x.fillna(0).cumsum().shift(1))
        .fillna(0)
    )
    lookup = combined.groupby(["player_id", "season", "week"])["career_carries"].first()
    for df in [train_df, val_df, test_df]:
        keys = pd.MultiIndex.from_arrays([df["player_id"], df["season"], df["week"]])
        df["career_carries"] = lookup.reindex(keys).values

    for df in [train_df, val_df, test_df]:
        _compute_rb_features(df)
    return train_df, val_df, test_df


def _compute_rb_features(df: pd.DataFrame) -> None:
    """Compute all 14 RB-specific features in-place."""
    df.sort_values(["player_id", "season", "week"], inplace=True)

    # --- Feature 1: yards_per_carry_L3 ---
    rush_yds_roll = df.groupby(["player_id", "season"])["rushing_yards"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    carries_roll = df.groupby(["player_id", "season"])["carries"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["yards_per_carry_L3"] = (rush_yds_roll / carries_roll).fillna(0)
    df.loc[carries_roll == 0, "yards_per_carry_L3"] = 0

    # --- Feature 2: reception_rate_L3 ---
    rec_roll = df.groupby(["player_id", "season"])["receptions"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    tgt_roll = df.groupby(["player_id", "season"])["targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["reception_rate_L3"] = (rec_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "reception_rate_L3"] = 0

    # --- Feature 3: weighted_opportunities_L3 ---
    df["_raw_weighted_opps"] = df["carries"].fillna(0) + 2 * df["targets"].fillna(0)
    df["weighted_opportunities_L3"] = df.groupby(["player_id", "season"])[
        "_raw_weighted_opps"
    ].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df.drop(columns=["_raw_weighted_opps"], inplace=True)

    # --- Features 4 & 5: team_rb_carry_share_L3 and team_rb_target_share_L3 ---
    team_rb_totals = compute_team_rb_totals(df)
    df_merged = df.merge(team_rb_totals, on=["recent_team", "season", "week"], how="left")

    # Carry share
    player_carries_roll = df_merged.groupby(["player_id", "season"])["carries"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    team_rb_carries_roll = df_merged.groupby(["player_id", "season"])["team_rb_carries"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    carry_share = (player_carries_roll / team_rb_carries_roll).fillna(0)
    carry_share[team_rb_carries_roll.values == 0] = 0
    df["team_rb_carry_share_L3"] = carry_share.values

    # Target share
    player_targets_roll = df_merged.groupby(["player_id", "season"])["targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    team_rb_targets_roll = df_merged.groupby(["player_id", "season"])["team_rb_targets"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    target_share = (player_targets_roll / team_rb_targets_roll).fillna(0)
    target_share[team_rb_targets_roll.values == 0] = 0
    df["team_rb_target_share_L3"] = target_share.values

    # --- Game-level shares (for attention history) ---
    _carries = df_merged["carries"].fillna(0).values
    _targets = df_merged["targets"].fillna(0).values
    _team_carries = df_merged["team_rb_carries"].values
    _team_targets = df_merged["team_rb_targets"].values

    df["game_carry_share"] = np.where(_team_carries > 0, _carries / _team_carries, 0.0)
    df["game_target_share"] = np.where(_team_targets > 0, _targets / _team_targets, 0.0)

    # --- Game-level HHI (team carry/target concentration) ---
    df["game_carry_hhi"] = df.groupby(["recent_team", "season", "week"])[
        "game_carry_share"
    ].transform(lambda x: (x**2).sum())
    df["game_target_hhi"] = df.groupby(["recent_team", "season", "week"])[
        "game_target_share"
    ].transform(lambda x: (x**2).sum())

    # --- Features 12 & 13: rolling HHI (L3) for Ridge ---
    df["team_rb_carry_hhi_L3"] = df.groupby(["player_id", "season"])["game_carry_hhi"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["team_rb_target_hhi_L3"] = df.groupby(["player_id", "season"])["game_target_hhi"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # --- Feature 14: opportunity_index_L3 (weighted opp share) ---
    _team_w_opps = _team_carries + 2 * _team_targets
    _player_w_opps = _carries + 2 * _targets
    df["_game_opp_idx"] = np.where(_team_w_opps > 0, _player_w_opps / _team_w_opps, 0.0)
    df["opportunity_index_L3"] = df.groupby(["player_id", "season"])["_game_opp_idx"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df.drop(columns=["_game_opp_idx"], inplace=True)

    # --- Feature 6: rushing_epa_per_attempt_L3 ---
    rushing_epa_roll = df.groupby(["player_id", "season"])["rushing_epa"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    # Reuse carries_roll from feature 1
    carries_roll_epa = df.groupby(["player_id", "season"])["carries"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["rushing_epa_per_attempt_L3"] = (rushing_epa_roll / carries_roll_epa).fillna(0)
    df.loc[carries_roll_epa == 0, "rushing_epa_per_attempt_L3"] = 0

    # --- Features 7 & 8: split first-down rates (replaces combined first_down_rate_L3
    #     to avoid collinearity — combined rate ≈ weighted avg of the two) ---
    rush_fd_roll = df.groupby(["player_id", "season"])["rushing_first_downs"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["rushing_first_down_rate_L3"] = (rush_fd_roll / carries_roll).fillna(0)
    df.loc[carries_roll == 0, "rushing_first_down_rate_L3"] = 0

    recv_fd_roll = df.groupby(["player_id", "season"])["receiving_first_downs"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["receiving_first_down_rate_L3"] = (recv_fd_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "receiving_first_down_rate_L3"] = 0

    # --- Feature 9: yac_per_reception_L3 ---
    yac_roll = df.groupby(["player_id", "season"])["receiving_yards_after_catch"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["yac_per_reception_L3"] = (yac_roll / rec_roll).fillna(0)
    df.loc[rec_roll == 0, "yac_per_reception_L3"] = 0

    # --- Feature 10: receiving_epa_per_target_L3 ---
    recv_epa_roll = df.groupby(["player_id", "season"])["receiving_epa"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["receiving_epa_per_target_L3"] = (recv_epa_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "receiving_epa_per_target_L3"] = 0

    # --- Feature 11: air_yards_per_target_L3 ---
    # Distinct from yac_per_reception (air yards = intended depth, YAC = post-catch;
    # low collinearity since backs can have high YAC with low air yards or vice versa)
    air_yds_roll = df.groupby(["player_id", "season"])["receiving_air_yards"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )
    df["air_yards_per_target_L3"] = (air_yds_roll / tgt_roll).fillna(0)
    df.loc[tgt_roll == 0, "air_yards_per_target_L3"] = 0


def fill_rb_nans(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rb_feature_cols: list[str],
) -> tuple:
    """Fill NaNs in RB-specific feature columns using training set statistics.

    Called AFTER temporal_split() and AFTER add_rb_specific_features().
    Uses ONLY training set statistics to prevent leakage.
    """
    # Replace inf with NaN
    for split_df in [train_df, val_df, test_df]:
        split_df[rb_feature_cols] = split_df[rb_feature_cols].replace([np.inf, -np.inf], np.nan)

    # Compute training set means
    train_means = train_df[rb_feature_cols].mean()

    for split_df in [train_df, val_df, test_df]:
        for col in rb_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])

    return train_df, val_df, test_df
