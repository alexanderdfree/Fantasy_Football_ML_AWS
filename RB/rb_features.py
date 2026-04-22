import numpy as np
import pandas as pd

from RB.rb_config import RB_INCLUDE_FEATURES
from RB.rb_data import compute_team_rb_totals
from shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)
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

    grp = ["player_id", "season"]

    def _sum(frame, col):
        return rolling_agg(frame, col, grp, window=3)

    rush_yds_roll = _sum(df, "rushing_yards")
    carries_roll = _sum(df, "carries")
    df["yards_per_carry_L3"] = safe_divide(rush_yds_roll, carries_roll)

    rec_roll = _sum(df, "receptions")
    tgt_roll = _sum(df, "targets")
    df["reception_rate_L3"] = safe_divide(rec_roll, tgt_roll)

    df["_raw_weighted_opps"] = df["carries"].fillna(0) + 2 * df["targets"].fillna(0)
    df["weighted_opportunities_L3"] = rolling_agg(
        df, "_raw_weighted_opps", grp, window=3, agg="mean"
    )
    df.drop(columns=["_raw_weighted_opps"], inplace=True)

    team_rb_totals = compute_team_rb_totals(df)
    df_merged = df.merge(team_rb_totals, on=["recent_team", "season", "week"], how="left")

    player_carries_roll = _sum(df_merged, "carries")
    team_rb_carries_roll = _sum(df_merged, "team_rb_carries")
    df["team_rb_carry_share_L3"] = safe_divide(player_carries_roll, team_rb_carries_roll).values

    player_targets_roll = _sum(df_merged, "targets")
    team_rb_targets_roll = _sum(df_merged, "team_rb_targets")
    df["team_rb_target_share_L3"] = safe_divide(player_targets_roll, team_rb_targets_roll).values

    # Game-level shares (for attention history). ``np.divide(..., where=...)``
    # is used directly rather than ``safe_divide`` because we need to suppress
    # the divide-by-zero warning entirely — the quotient is never computed on
    # zero-denom rows here.
    _carries = df_merged["carries"].fillna(0).values
    _targets = df_merged["targets"].fillna(0).values
    _team_carries = df_merged["team_rb_carries"].values
    _team_targets = df_merged["team_rb_targets"].values

    df["game_carry_share"] = np.divide(
        _carries, _team_carries, out=np.zeros_like(_carries, dtype=float), where=_team_carries > 0
    )
    df["game_target_share"] = np.divide(
        _targets, _team_targets, out=np.zeros_like(_targets, dtype=float), where=_team_targets > 0
    )

    df["game_carry_hhi"] = df.groupby(["recent_team", "season", "week"])[
        "game_carry_share"
    ].transform(lambda x: (x**2).sum())
    df["game_target_hhi"] = df.groupby(["recent_team", "season", "week"])[
        "game_target_share"
    ].transform(lambda x: (x**2).sum())

    df["team_rb_carry_hhi_L3"] = rolling_agg(df, "game_carry_hhi", grp, window=3, agg="mean")
    df["team_rb_target_hhi_L3"] = rolling_agg(df, "game_target_hhi", grp, window=3, agg="mean")

    # opportunity_index_L3 (weighted opp share). Same ``np.divide(..., where=)``
    # pattern as the game-level shares above for the same reason.
    _team_w_opps = _team_carries + 2 * _team_targets
    _player_w_opps = _carries + 2 * _targets
    df["_game_opp_idx"] = np.divide(
        _player_w_opps,
        _team_w_opps,
        out=np.zeros_like(_player_w_opps, dtype=float),
        where=_team_w_opps > 0,
    )
    df["opportunity_index_L3"] = rolling_agg(df, "_game_opp_idx", grp, window=3, agg="mean")
    df.drop(columns=["_game_opp_idx"], inplace=True)

    rushing_epa_roll = _sum(df, "rushing_epa")
    df["rushing_epa_per_attempt_L3"] = safe_divide(rushing_epa_roll, carries_roll)

    # Split first-down rates (replaces combined first_down_rate_L3 to avoid
    # collinearity — combined rate ≈ weighted avg of the two).
    rush_fd_roll = _sum(df, "rushing_first_downs")
    df["rushing_first_down_rate_L3"] = safe_divide(rush_fd_roll, carries_roll)

    recv_fd_roll = _sum(df, "receiving_first_downs")
    df["receiving_first_down_rate_L3"] = safe_divide(recv_fd_roll, rec_roll)

    yac_roll = _sum(df, "receiving_yards_after_catch")
    df["yac_per_reception_L3"] = safe_divide(yac_roll, rec_roll)

    recv_epa_roll = _sum(df, "receiving_epa")
    df["receiving_epa_per_target_L3"] = safe_divide(recv_epa_roll, tgt_roll)

    # air_yards_per_target_L3: distinct from yac_per_reception (air yards =
    # intended depth, YAC = post-catch; low collinearity since backs can have
    # high YAC with low air yards or vice versa).
    air_yds_roll = _sum(df, "receiving_air_yards")
    df["air_yards_per_target_L3"] = safe_divide(air_yds_roll, tgt_roll)


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
    return fill_nans_with_train_means(train_df, val_df, test_df, rb_feature_cols)
