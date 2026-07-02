import numpy as np
import pandas as pd

from src.features.engineer import flatten_include_features
from src.shared.feature_build import (
    fill_nans_with_train_means,
    rolling_agg,
    safe_divide,
)
from src.te.config import POSITION_CONFIG
from src.te.data import compute_team_te_totals


def get_feature_columns() -> list[str]:
    """Return the complete ordered list of feature columns for the TE model."""
    return flatten_include_features(POSITION_CONFIG.include_features)


def add_specific_features(train_df, val_df, test_df, full_train=None):
    """Add the TE-specific engineered features to each split.

    ``full_train`` is accepted for the shared ``add_features_fn`` contract
    (#574/#531) but unused: TE runs at ``min_games_per_season=1`` (no train
    filter), so the within-frame team-TE-target denominators in the red-zone /
    opportunity block are already complete (no undercount to correct).
    """
    for df in [train_df, val_df, test_df]:
        _compute_features(df)
    return train_df, val_df, test_df


def _compute_features(df: pd.DataFrame) -> None:
    """Compute the TE-specific features in-place.

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
    # Stint-aware grouping for the team-TE-target-share rolling (#1192): a TE
    # traded mid-season would otherwise have the 3-week rolling team_te_targets
    # denominator concatenate the OLD team's TE-target volume with the NEW
    # team's for ~3 weeks post-trade, mixing two teams' totals into one share.
    # Build stint_id locally (the engineer.py one is dropped before we run) by
    # flagging each in-season team change and cumsum-ing it — mirrors
    # src/wr/features.py (#674). df is already sorted by (player_id, season,
    # week) above, so the merge (1-to-1 on recent_team, season, week) preserves
    # order and stint_id carries onto df_merged.
    df_merged["_team_changed"] = (
        df_merged.groupby(["player_id", "season"])["recent_team"].shift(1)
        != df_merged["recent_team"]
    ).fillna(False)
    df_merged["stint_id"] = df_merged.groupby(["player_id", "season"])["_team_changed"].cumsum()
    stint_grp = ["player_id", "season", "stint_id"]
    player_tgt_roll = rolling_agg(df_merged, "targets", stint_grp, window=3)
    team_te_tgt_roll = rolling_agg(df_merged, "team_te_targets", stint_grp, window=3)
    df["team_te_target_share_L3"] = safe_divide(player_tgt_roll, team_te_tgt_roll).values

    df["receiving_epa_per_target_L3"] = safe_divide(recv_epa_roll, tgt_roll)
    df["receiving_first_down_rate_L3"] = safe_divide(recv_fd_roll, rec_roll)
    df["air_yards_per_target_L3"] = safe_divide(air_yds_roll, tgt_roll)
    df["td_rate_per_target_L3"] = safe_divide(recv_tds_roll, tgt_roll)

    # --- Red-zone receiving + opportunity (boom-tier signal) ---
    # Parity with RB/WR (#1061 boom block). TEs are heavy red-zone targets; mirrors
    # src/wr/features.py. Team totals are TE-scoped (sum over this team-game's TE rows —
    # transform("sum")); redzone_targets/_share come from the splits (engineer via
    # redzone_pbp). NOTE: the 8-seed TE A/B (src/tuning/ab_boom_signals_te.py) found this
    # block MAE-neutral with no robust boom-subgroup gain on TE (unlike WR #1061) — shipped
    # for RB/WR feature-surface parity, not as a measured mover.
    team_g = df.groupby(["recent_team", "season", "week"])
    team_tgt = team_g["targets"].transform("sum").to_numpy(dtype=float)
    team_car = team_g["carries"].transform("sum").to_numpy(dtype=float)
    p_tgt = df["targets"].fillna(0).to_numpy(dtype=float)
    p_car = df["carries"].fillna(0).to_numpy(dtype=float)
    # Per-game shares + weighted-opportunity index — raw current-week values fed ONLY to
    # the attention history (build_game_history_arrays applies its own prior-games shift, so
    # they never leak into a static feature).
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
    # Prior-season catch rate (S-1 → S), low-volume-guarded like src/wr/features.py.
    if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(df.columns):
        catch_rate = safe_divide(
            df["prior_season_mean_receptions"], df["prior_season_mean_targets"]
        )
        df["prior_season_mean_catch_rate"] = catch_rate.where(
            df["prior_season_mean_targets"] >= 0.5
        )


def fill_nans(train_df, val_df, test_df, te_feature_cols):
    """Fill NaNs in TE-specific feature columns using training set statistics."""
    # prior_season_mean_catch_rate is in INCLUDE_FEATURES["prior_season"] (so
    # the model uses it) but absent from _SPECIFIC_FEATURES — the column set
    # the pipeline passes here as te_feature_cols — so without this it skips
    # the leak-safe train-mean fill and falls through to
    # build_position_features' catch-all .fillna(0). For a rookie /
    # no-prior-season TE (and the ≥0.5/game volume guard in _compute_features)
    # that 0 is far from the league-average catch rate and maps to a strong
    # negative z-score post-scaler. Mirrors the RB fix (#390); TE gap #1290.
    prior_cols = [
        c
        for c in ("prior_season_mean_catch_rate",)
        if c in train_df.columns and c not in te_feature_cols
    ]
    return fill_nans_with_train_means(train_df, val_df, test_df, [*te_feature_cols, *prior_cols])
