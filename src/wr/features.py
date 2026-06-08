import numpy as np
import pandas as pd

from src.features.engineer import flatten_include_features
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)
from src.wr.config import POSITION_CONFIG
from src.wr.data import compute_team_wr_totals


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the WR model."""
    return flatten_include_features(POSITION_CONFIG.include_features)


def add_specific_features(train_df, val_df, test_df, full_train=None):
    """Add the WR-specific engineered features to each split.

    ``team_wr_target_share_L3`` divides player targets by the team's WR-target
    total per game. When ``full_train`` (the pre-min-games-filter train) is
    supplied, those totals are computed over it — not the filtered ``train_df`` —
    so dropped low-volume WRs don't undercount the denominator (#531). Only the
    min-games-filtered rows are returned; ``fill_nans`` + the StandardScaler
    still fit on those filtered rows in ``build_position_features`` (#569).
    """
    base_train = train_df if full_train is None else full_train
    for df in [base_train, val_df, test_df]:
        _compute_features(df)
    if full_train is None:
        return base_train, val_df, test_df
    train_out = base_train[base_train.index.isin(train_df.index)]
    return train_out, val_df, test_df


def _compute_features(df: pd.DataFrame) -> None:
    """Compute the WR-specific engineered features in-place.

    The 8 rate/share features (``*_L3``, ``team_wr_target_share_L3``) plus the
    red-zone / opportunity boom-tier block (``game_target_share`` / ``_hhi`` /
    ``game_opportunity_index`` → attention history; ``opportunity_index_L3`` /
    ``redzone_targets_L3`` / ``redzone_target_share_L3`` / ``prior_season_mean_catch_rate``
    → whitelist). The boom block was validated in the 6-seed WR A/B
    (``src/tuning/ab_boom_signals_wr.py``, ``+all`` arm) and mirrors ``src/rb/features.py``.

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
    # Stint-aware grouping for the team-WR-target-share rolling: a WR traded
    # mid-season would otherwise have their 3-week rolling team_wr_targets
    # denominator concatenate the OLD team's WR-target volume with the NEW
    # team's for ~3 weeks post-trade, mixing two teams' totals into one share.
    # Build stint_id locally (the engineer.py one is dropped before we run) by
    # flagging each in-season team change and cumsum-ing it — mirrors
    # src.features.engineer's target_share_L{w} (#674). df is already sorted by
    # (player_id, season, week) above, so the merge (1-to-1 on recent_team,
    # season, week) preserves order and stint_id carries onto df_merged.
    df_merged["_team_changed"] = (
        df_merged.groupby(["player_id", "season"])["recent_team"].shift(1)
        != df_merged["recent_team"]
    ).fillna(False)
    df_merged["stint_id"] = df_merged.groupby(["player_id", "season"])["_team_changed"].cumsum()
    stint_grp = ["player_id", "season", "stint_id"]
    player_tgt_roll = rolling_agg(df_merged, "targets", stint_grp, window=3)
    team_wr_tgt_roll = rolling_agg(df_merged, "team_wr_targets", stint_grp, window=3)
    df["team_wr_target_share_L3"] = safe_divide(player_tgt_roll, team_wr_tgt_roll).values

    df["receiving_epa_per_target_L3"] = safe_divide(recv_epa_roll, tgt_roll)
    df["receiving_first_down_rate_L3"] = safe_divide(recv_fd_roll, rec_roll)

    # --- Red-zone receiving + opportunity (boom-tier signal) ---
    # Validated in the 6-seed WR boom A/B (src/tuning/ab_boom_signals_wr.py, +all arm:
    # direction-robust modest gain on the Q4 / receiving-TD subgroup across Ridge, LightGBM
    # and the attention NN). Mirrors src/rb/features.py. Team totals are WR-scoped (sum over
    # this team-game's WR rows — transform("sum"), matching the validated A/B injector);
    # redzone_targets/_share come from the splits (engineer via redzone_pbp).
    team_g = df.groupby(["recent_team", "season", "week"])
    team_tgt = team_g["targets"].transform("sum").to_numpy(dtype=float)
    team_car = team_g["carries"].transform("sum").to_numpy(dtype=float)
    p_tgt = df["targets"].fillna(0).to_numpy(dtype=float)
    p_car = df["carries"].fillna(0).to_numpy(dtype=float)
    # Per-game shares + weighted-opportunity index — raw current-week values fed ONLY to the
    # attention history (build_game_history_arrays applies its own prior-games shift, so they
    # never leak into a static feature). np.divide(..., where=) maps 0/0 (no WR targets that
    # game) to 0 without a warning.
    df["game_target_share"] = np.divide(
        p_tgt, team_tgt, out=np.zeros_like(p_tgt), where=team_tgt > 0
    )
    df["game_target_hhi"] = df.groupby(["recent_team", "season", "week"])[
        "game_target_share"
    ].transform(lambda x: (x**2).sum())
    team_w = team_car + 2.0 * team_tgt
    player_w = p_car + 2.0 * p_tgt
    df["game_opportunity_index"] = np.divide(
        player_w, team_w, out=np.zeros_like(player_w), where=team_w > 0
    )
    # Leakage-safe rolling forms (rolling_agg shift=1) → whitelist (Ridge/LGBM/NN-static).
    df["opportunity_index_L3"] = rolling_agg(
        df, "game_opportunity_index", grp, window=3, agg="mean"
    )
    df["redzone_targets_L3"] = rolling_agg(df, "redzone_targets", grp, window=3, agg="mean")
    df["redzone_target_share_L3"] = rolling_agg(
        df, "redzone_target_share", grp, window=3, agg="mean"
    )
    # Prior-season catch rate (S-1 → S), low-volume-guarded like src/rb/features.py.
    if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(df.columns):
        catch_rate = safe_divide(
            df["prior_season_mean_receptions"], df["prior_season_mean_targets"]
        )
        df["prior_season_mean_catch_rate"] = catch_rate.where(
            df["prior_season_mean_targets"] >= 0.5
        )


def fill_nans(train_df, val_df, test_df, wr_feature_cols):
    """Fill NaNs in WR-specific feature columns using training set statistics."""
    return fill_nans_with_train_means(train_df, val_df, test_df, wr_feature_cols)
