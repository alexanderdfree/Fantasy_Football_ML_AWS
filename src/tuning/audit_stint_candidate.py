"""Experiment-only WR/TE feature functions copied from PR #1607.

Source: d27a876cc80dde55a1d33f65b04664171fb3380f.
Only function names differ to keep both isolated implementations in one module.
"""

import numpy as np
import pandas as pd

from src.shared.feature_build import rolling_agg, safe_divide
from src.te.data import compute_team_te_totals
from src.wr.data import compute_team_wr_totals


def wr_compute_features(df: pd.DataFrame) -> None:
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
    # Team-relative shares restart when the player joins a different team.
    # Raw personal red-zone target volume retains its player-season window.
    for source, target in (
        ("game_opportunity_index", "opportunity_index_L3"),
        ("redzone_target_share", "redzone_target_share_L3"),
    ):
        df_merged[source] = df[source].to_numpy()
        df[target] = rolling_agg(df_merged, source, stint_grp, window=3, agg="mean").to_numpy()
    df["redzone_targets_L3"] = rolling_agg(df, "redzone_targets", grp, window=3, agg="mean")
    # Prior-season catch rate (S-1 → S), low-volume-guarded like src/rb/features.py.
    if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(df.columns):
        catch_rate = safe_divide(
            df["prior_season_mean_receptions"], df["prior_season_mean_targets"]
        )
        df["prior_season_mean_catch_rate"] = catch_rate.where(
            df["prior_season_mean_targets"] >= 0.5
        )


def te_compute_features(df: pd.DataFrame) -> None:
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
    # Team-relative shares restart when the player joins a different team.
    # Raw personal red-zone target volume retains its player-season window.
    for source, target in (
        ("game_opportunity_index", "opportunity_index_L3"),
        ("redzone_target_share", "redzone_target_share_L3"),
    ):
        df_merged[source] = df[source].to_numpy()
        df[target] = rolling_agg(df_merged, source, stint_grp, window=3, agg="mean").to_numpy()
    df["redzone_targets_L3"] = rolling_agg(df, "redzone_targets", grp, window=3, agg="mean")
    # Prior-season catch rate (S-1 → S), low-volume-guarded like src/wr/features.py.
    if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(df.columns):
        catch_rate = safe_divide(
            df["prior_season_mean_receptions"], df["prior_season_mean_targets"]
        )
        df["prior_season_mean_catch_rate"] = catch_rate.where(
            df["prior_season_mean_targets"] >= 0.5
        )
