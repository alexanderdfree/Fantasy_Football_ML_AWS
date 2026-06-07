import pandas as pd

from src.features.engineer import flatten_include_features
from src.qb.config import POSITION_CONFIG
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the QB model."""
    return flatten_include_features(POSITION_CONFIG.include_features)


def add_specific_features(train_df, val_df, test_df, full_train=None):
    """Add QB-specific engineered features (see ``POSITION_CONFIG.specific_features``) to each split.

    ``full_train`` is accepted for the shared ``add_features_fn`` contract
    (#574/#531) but unused — QB has no team-level per-game share/HHI features
    whose denominators the MIN_GAMES train filter would undercount.
    """
    dfs = [train_df, val_df, test_df]
    for df in dfs:
        _compute_features(df)
    _add_rookie_phase_features(dfs)
    return train_df, val_df, test_df


def _add_rookie_phase_features(dfs: list[pd.DataFrame], early_games: int = 3) -> None:
    """Add split-aware rookie phase indicators in-place.

    ``_compute_features`` receives one split at a time, but rookie status needs
    the player's first season across the train/val/test set. The first season
    available in the whole dataset is deliberately treated as unknown/veteran:
    the data starts mid-career for many players, so labeling 2012 rows as
    rookies would inject noise into training.
    """
    required = {"player_id", "season", "week"}
    if any(not required.issubset(df.columns) for df in dfs):
        for df in dfs:
            df["is_rookie"] = 0.0
            df["rookie_early"] = 0.0
        return

    parts = []
    for split_idx, df in enumerate(dfs):
        part = df[["player_id", "season", "week"]].copy()
        part["_split_idx"] = split_idx
        part["_row_pos"] = range(len(df))
        parts.append(part)

    all_rows = pd.concat(parts, ignore_index=True)
    debut_season = all_rows.groupby("player_id")["season"].min()
    first_data_season = all_rows["season"].min()

    ordered = all_rows.sort_values(["player_id", "season", "week"], kind="stable").copy()
    ordered["_game_idx"] = ordered.groupby(["player_id", "season"]).cumcount()
    ordered["is_rookie"] = (
        ordered["season"].eq(ordered["player_id"].map(debut_season))
        & ordered["season"].gt(first_data_season)
    ).astype(float)
    ordered["rookie_early"] = (
        ordered["is_rookie"].eq(1.0) & ordered["_game_idx"].lt(early_games)
    ).astype(float)

    restored = ordered.sort_values(["_split_idx", "_row_pos"], kind="stable")
    for split_idx, df in enumerate(dfs):
        values = restored[restored["_split_idx"].eq(split_idx)]
        df["is_rookie"] = values["is_rookie"].to_numpy(dtype=float)
        df["rookie_early"] = values["rookie_early"].to_numpy(dtype=float)


def _compute_features(df: pd.DataFrame) -> None:
    """Compute all QB-specific features (see ``POSITION_CONFIG.specific_features``) in-place.

    Side-effect contract: per-group rolling aggregates require chronological
    row order, so the function physically reorders ``df`` by
    ``(player_id, season, week)`` via column-wise reassignment (no
    ``sort_values(..., inplace=True)``, which pandas discourages under CoW)
    before computing features. Row labels are realigned so ``df.index``
    reflects the sorted order — same as the previous in-place mutator left
    the frame. Callers that needed the original row order should pass a copy.
    """
    df_sorted = df.sort_values(["player_id", "season", "week"])
    for col in df_sorted.columns:
        df[col] = df_sorted[col].values
    df.index = df_sorted.index

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


def fill_nans(train_df, val_df, test_df, qb_feature_cols):
    """Fill NaNs in QB-specific feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, qb_feature_cols)
