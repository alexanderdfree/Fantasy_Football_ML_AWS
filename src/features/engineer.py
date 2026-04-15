import pandas as pd
import numpy as np
from src.config import (
    ROLLING_WINDOWS, ROLL_STATS, ROLL_AGGS, EWMA_STATS, EWMA_SPANS,
    TREND_STATS, SHARE_WINDOWS, OPP_ROLLING_WINDOW,
    CACHE_DIR, SEASONS,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 144 engineered features from preprocessed data."""
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- Rolling Features (93: 90 mean/std/max + 3 min) ---
    for stat in ROLL_STATS:
        for window in ROLLING_WINDOWS:
            df[f"rolling_mean_{stat}_L{window}"] = df.groupby(
                ["player_id", "season"]
            )[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"rolling_std_{stat}_L{window}"] = df.groupby(
                ["player_id", "season"]
            )[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            df[f"rolling_max_{stat}_L{window}"] = df.groupby(
                ["player_id", "season"]
            )[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )
            if stat == "fantasy_points":
                df[f"rolling_min_{stat}_L{window}"] = df.groupby(
                    ["player_id", "season"]
                )[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).min()
                )

    # --- Prior-Season Summary Features (24) ---
    prior_stats = [s for s in ROLL_STATS]
    prior = df.groupby(["player_id", "season"])[prior_stats].agg(["mean", "std", "max"])
    prior.columns = [f"prior_season_{agg}_{stat}" for stat, agg in prior.columns]
    prior = prior.reset_index()
    prior["season"] = prior["season"] + 1  # align S-1 stats with season S
    df = df.merge(prior, on=["player_id", "season"], how="left")

    # --- EWMA Features (14) ---
    for stat in EWMA_STATS:
        for span in EWMA_SPANS:
            df[f"ewma_{stat}_L{span}"] = df.groupby(
                ["player_id", "season"]
            )[stat].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
            )

    # --- Trend / Momentum Features (4) ---
    for stat in TREND_STATS:
        short = df.groupby(["player_id", "season"])[stat].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        long = df.groupby(["player_id", "season"])[stat].transform(
            lambda x: x.shift(1).rolling(8, min_periods=1).mean()
        )
        df[f"trend_{stat}"] = short - long

    # --- Share / Usage Features (6) ---
    team_totals = df.groupby(["recent_team", "season", "week"]).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
    ).reset_index()
    df = df.merge(team_totals, on=["recent_team", "season", "week"], how="left")

    # Detect team changes for stint-aware grouping
    df = df.sort_values(["player_id", "season", "week"])
    df["team_changed"] = (
        df.groupby(["player_id", "season"])["recent_team"].shift(1) != df["recent_team"]
    ).fillna(False)
    df["stint_id"] = df.groupby(["player_id", "season"])["team_changed"].cumsum()

    for window in SHARE_WINDOWS:
        df[f"player_targets_roll_L{window}"] = df.groupby(
            ["player_id", "season", "stint_id"]
        )["targets"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"team_targets_roll_L{window}"] = df.groupby(
            ["player_id", "season", "stint_id"]
        )["team_targets"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"target_share_L{window}"] = (
            df[f"player_targets_roll_L{window}"]
            / df[f"team_targets_roll_L{window}"]
        ).fillna(0)

        df[f"player_carries_roll_L{window}"] = df.groupby(
            ["player_id", "season", "stint_id"]
        )["carries"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"team_carries_roll_L{window}"] = df.groupby(
            ["player_id", "season", "stint_id"]
        )["team_carries"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).sum()
        )
        df[f"carry_share_L{window}"] = (
            df[f"player_carries_roll_L{window}"]
            / df[f"team_carries_roll_L{window}"]
        ).fillna(0)

    # air_yards_share (lagged to prevent data leakage)
    if "receiving_air_yards" in df.columns:
        team_air_yards = df.groupby(["recent_team", "season", "week"])[
            "receiving_air_yards"
        ].transform("sum")
        df["_raw_air_yards_share"] = (df["receiving_air_yards"] / team_air_yards).fillna(0)
        df["air_yards_share"] = df.groupby(
            ["player_id", "season"]
        )["_raw_air_yards_share"].shift(1).fillna(0)
        df.drop(columns=["_raw_air_yards_share"], inplace=True)
    else:
        df["air_yards_share"] = 0.0

    # Lag snap_pct to prevent data leakage (use prior week's snap percentage)
    if "snap_pct" in df.columns:
        df["snap_pct"] = df.groupby(["player_id", "season"])["snap_pct"].shift(1).fillna(0)

    # Clean up intermediate columns
    drop_cols = [c for c in df.columns if c.startswith((
        "player_targets_roll", "team_targets_roll",
        "player_carries_roll", "team_carries_roll",
    ))]
    drop_cols += ["team_targets", "team_carries", "team_changed", "stint_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Matchup / Opponent Features (4) ---
    df = _build_matchup_features(df)

    # --- Defense Matchup Features (7) ---
    df = _build_defense_matchup_features(df)

    # --- Contextual Features (4) ---
    df = _build_contextual_features(df)

    # --- Position Encoding (4) ---
    for pos in ["QB", "RB", "WR", "TE"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)

    return df


# === Game History Extraction (for attention model) ===

# Default per-game stats to include in history vectors.
# These are the raw stats that the rolling features are derived from.
GAME_HISTORY_STATS = [
    "fantasy_points", "fantasy_points_floor",
    "passing_yards", "rushing_yards", "receiving_yards",
    "passing_tds", "rushing_tds", "receiving_tds",
    "attempts", "completions", "carries",
    "targets", "receptions", "snap_pct",
    "interceptions",
]


def build_game_history_arrays(
    df: pd.DataFrame,
    history_stats: list[str] = None,
    max_seq_len: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-player game history as padded arrays for the attention model.

    For each player-week row, gathers that player's prior games within the same
    season (shifted by 1 to prevent leakage — same convention as rolling features).

    Args:
        df: DataFrame with player_id, season, week, and stat columns.
            Must already be sorted by [player_id, season, week].
        history_stats: list of column names to include per game.
        max_seq_len: maximum history length (pad/truncate to this).

    Returns:
        X_history: [n_samples, max_seq_len, game_dim] float32 array (zero-padded)
        history_mask: [n_samples, max_seq_len] bool array (True = real game)
    """
    if history_stats is None:
        history_stats = GAME_HISTORY_STATS
    # Only use stats that exist in the DataFrame
    history_stats = [s for s in history_stats if s in df.columns]
    game_dim = len(history_stats)

    n = len(df)
    X_history = np.zeros((n, max_seq_len, game_dim), dtype=np.float32)
    history_mask = np.zeros((n, max_seq_len), dtype=bool)

    # Group by player-season for efficient lookback
    df_sorted = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    stat_values = df_sorted[history_stats].values.astype(np.float32)

    # Build index mapping: (player_id, season) -> list of row indices in order
    group_indices = {}
    for idx, (pid, season) in enumerate(zip(df_sorted["player_id"], df_sorted["season"])):
        key = (pid, season)
        if key not in group_indices:
            group_indices[key] = []
        group_indices[key].append(idx)

    # For each row, look back at prior games in the same player-season
    for key, indices in group_indices.items():
        for pos_in_group, row_idx in enumerate(indices):
            # Games before this one (shift=1 equivalent)
            n_prior = pos_in_group  # games 0..pos_in_group-1
            if n_prior == 0:
                continue  # no history for first game of season

            # Take most recent games, up to max_seq_len
            start = max(0, n_prior - max_seq_len)
            prior_indices = indices[start:pos_in_group]
            seq_len = len(prior_indices)

            # Fill from the end so most recent game is last
            X_history[row_idx, :seq_len] = stat_values[prior_indices]
            history_mask[row_idx, :seq_len] = True

    # Replace NaN with 0 in history
    np.nan_to_num(X_history, copy=False, nan=0.0)

    return X_history, history_mask


def _build_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build opponent/matchup features."""
    # Determine opponent from schedule or opponent_team column
    if "opponent_team" in df.columns:
        df["opponent"] = df["opponent_team"]
    else:
        df["opponent"] = None

    # Compute rush and receiving fantasy components
    df["rush_fantasy"] = df["rushing_yards"] * 0.1 + df["rushing_tds"] * 6
    df["recv_fantasy"] = (
        df["receiving_yards"] * 0.1 + df["receiving_tds"] * 6 + df["receptions"] * 1
    )

    if df["opponent"].notna().any():
        def_pts = df.groupby(["opponent", "position", "season", "week"]).agg(
            pts_allowed_to_pos=("fantasy_points", "sum"),
            rush_pts_allowed_to_pos=("rush_fantasy", "sum"),
            recv_pts_allowed_to_pos=("recv_fantasy", "sum"),
        ).reset_index()

        def_pts = def_pts.sort_values(["opponent", "position", "season", "week"])
        for col in ["pts_allowed_to_pos", "rush_pts_allowed_to_pos", "recv_pts_allowed_to_pos"]:
            def_pts[f"opp_{col}"] = def_pts.groupby(
                ["opponent", "position"]
            )[col].transform(
                lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean()
            )

        def_pts.rename(columns={
            "opp_pts_allowed_to_pos": "opp_fantasy_pts_allowed_to_pos",
            "opp_rush_pts_allowed_to_pos": "opp_rush_pts_allowed_to_pos",
            "opp_recv_pts_allowed_to_pos": "opp_recv_pts_allowed_to_pos",
        }, inplace=True)

        # Rank: 1 = most points allowed = best matchup
        def_pts["opp_def_rank_vs_pos"] = def_pts.groupby(
            ["position", "season", "week"]
        )["opp_fantasy_pts_allowed_to_pos"].rank(ascending=False, method="min")

        merge_cols = [
            "opponent", "position", "season", "week",
            "opp_fantasy_pts_allowed_to_pos", "opp_rush_pts_allowed_to_pos",
            "opp_recv_pts_allowed_to_pos", "opp_def_rank_vs_pos",
        ]
        def_pts_merge = def_pts[merge_cols].drop_duplicates()
        df = df.merge(def_pts_merge, on=["opponent", "position", "season", "week"], how="left")
    else:
        for col in ["opp_fantasy_pts_allowed_to_pos", "opp_rush_pts_allowed_to_pos",
                     "opp_recv_pts_allowed_to_pos", "opp_def_rank_vs_pos"]:
            df[col] = 0.0

    df.drop(columns=["rush_fantasy", "recv_fantasy", "opponent"], errors="ignore", inplace=True)
    return df


def _build_defense_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build detailed opposing-defense features from team-level aggregations.

    Computes 5 rolling defense stats (sacks, pass yds/TDs allowed, INTs, rush yds allowed),
    1 schedule-derived stat (points allowed), and 1 Vegas feature (implied team total).
    """
    # --- 1. Defense stats derived from opponent offensive data ---
    if "opponent_team" not in df.columns or df["opponent_team"].isna().all():
        for col in [
            "opp_def_sacks_L5", "opp_def_pass_yds_allowed_L5",
            "opp_def_pass_td_allowed_L5", "opp_def_ints_L5",
            "opp_def_rush_yds_allowed_L5", "opp_def_pts_allowed_L5",
            "implied_team_total",
        ]:
            df[col] = 0.0
        return df

    # Aggregate offensive stats allowed by each defense per game
    def_stats = df.groupby(["opponent_team", "season", "week"]).agg(
        _def_sacks=("sacks", "sum"),
        _def_pass_yds=("passing_yards", "sum"),
        _def_pass_tds=("passing_tds", "sum"),
        _def_ints=("interceptions", "sum"),
        _def_rush_yds=("rushing_yards", "sum"),
    ).reset_index()

    def_stats.sort_values(["opponent_team", "season", "week"], inplace=True)

    # L5 rolling averages with shift(1) for leakage prevention
    stat_map = {
        "_def_sacks": "opp_def_sacks_L5",
        "_def_pass_yds": "opp_def_pass_yds_allowed_L5",
        "_def_pass_tds": "opp_def_pass_td_allowed_L5",
        "_def_ints": "opp_def_ints_L5",
        "_def_rush_yds": "opp_def_rush_yds_allowed_L5",
    }
    for raw_col, out_col in stat_map.items():
        def_stats[out_col] = def_stats.groupby(
            ["opponent_team", "season"]
        )[raw_col].transform(
            lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean()
        )

    # Merge onto player rows via opponent_team
    merge_cols = ["opponent_team", "season", "week"] + list(stat_map.values())
    def_merge = def_stats[merge_cols].drop_duplicates()
    df = df.merge(def_merge, on=["opponent_team", "season", "week"], how="left")

    # --- 2. Points allowed from schedule scores ---
    schedules = pd.read_parquet(f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")
    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()

    away_pts = schedules_reg[["season", "week", "away_team", "home_score"]].copy()
    away_pts.columns = ["season", "week", "team", "points_allowed"]
    home_pts = schedules_reg[["season", "week", "home_team", "away_score"]].copy()
    home_pts.columns = ["season", "week", "team", "points_allowed"]
    pts_allowed = pd.concat([away_pts, home_pts], ignore_index=True)
    pts_allowed.sort_values(["team", "season", "week"], inplace=True)

    pts_allowed["opp_def_pts_allowed_L5"] = pts_allowed.groupby(
        ["team", "season"]
    )["points_allowed"].transform(
        lambda x: x.shift(1).rolling(OPP_ROLLING_WINDOW, min_periods=1).mean()
    )

    pts_merge = pts_allowed[["team", "season", "week", "opp_def_pts_allowed_L5"]].drop_duplicates()
    df = df.merge(
        pts_merge,
        left_on=["opponent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    )
    df.drop(columns=["team"], errors="ignore", inplace=True)

    # --- 3. Implied team total from Vegas lines ---
    home_impl = schedules_reg[["season", "week", "home_team", "spread_line", "total_line"]].copy()
    home_impl["implied_team_total"] = (home_impl["total_line"] - home_impl["spread_line"]) / 2
    home_impl = home_impl[["season", "week", "home_team", "implied_team_total"]]
    home_impl.columns = ["season", "week", "recent_team", "implied_team_total"]

    away_impl = schedules_reg[["season", "week", "away_team", "spread_line", "total_line"]].copy()
    away_impl["implied_team_total"] = (away_impl["total_line"] + away_impl["spread_line"]) / 2
    away_impl = away_impl[["season", "week", "away_team", "implied_team_total"]]
    away_impl.columns = ["season", "week", "recent_team", "implied_team_total"]

    impl_lookup = pd.concat([home_impl, away_impl], ignore_index=True).drop_duplicates(
        subset=["season", "week", "recent_team"]
    )

    if "implied_team_total" in df.columns:
        df.drop(columns=["implied_team_total"], inplace=True)
    df = df.merge(impl_lookup, on=["season", "week", "recent_team"], how="left")

    # Fill NaNs (early-season games with no prior history)
    for col in list(stat_map.values()) + ["opp_def_pts_allowed_L5", "implied_team_total"]:
        df[col] = df[col].fillna(0)

    return df


def _build_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build contextual features: is_home, week, is_returning_from_absence, days_rest."""
    # is_home (default 0 if can't determine)
    if "is_home" not in df.columns:
        df["is_home"] = 0

    # is_returning_from_absence
    df = df.sort_values(["player_id", "season", "week"])
    df["weeks_since_last_game"] = df.groupby(
        ["player_id", "season"]
    )["week"].diff().fillna(1)
    df["is_returning_from_absence"] = (df["weeks_since_last_game"] > 1).astype(int)
    df.drop(columns=["weeks_since_last_game"], inplace=True)

    # days_rest (approximate: 7 days per week gap)
    df["days_rest"] = df.groupby(
        ["player_id", "season"]
    )["week"].diff().fillna(1) * 7
    df["days_rest"] = df["days_rest"].clip(lower=4, upper=21)

    # Injury status (merged in loader; ensure defaults for missing data)
    if "practice_status" not in df.columns:
        df["practice_status"] = 2.0
    if "game_status" not in df.columns:
        df["game_status"] = 1.0

    # Depth chart rank (merged in loader; ensure default)
    if "depth_chart_rank" not in df.columns:
        df["depth_chart_rank"] = 1.0

    return df


def get_feature_columns() -> list[str]:
    """Dynamically generate the ordered list of all feature column names."""
    cols = []

    # Rolling features
    for stat in ROLL_STATS:
        for window in ROLLING_WINDOWS:
            for agg in ROLL_AGGS:
                cols.append(f"rolling_{agg}_{stat}_L{window}")
            if stat == "fantasy_points":
                cols.append(f"rolling_min_{stat}_L{window}")

    # Prior-season summary
    for stat in ROLL_STATS:
        for agg in ROLL_AGGS:
            cols.append(f"prior_season_{agg}_{stat}")

    # EWMA
    for stat in EWMA_STATS:
        for span in EWMA_SPANS:
            cols.append(f"ewma_{stat}_L{span}")

    # Trend
    for stat in TREND_STATS:
        cols.append(f"trend_{stat}")

    # Share
    for window in SHARE_WINDOWS:
        cols.append(f"target_share_L{window}")
        cols.append(f"carry_share_L{window}")
    cols += ["snap_pct", "air_yards_share"]

    # Matchup
    cols += [
        "opp_fantasy_pts_allowed_to_pos", "opp_rush_pts_allowed_to_pos",
        "opp_recv_pts_allowed_to_pos", "opp_def_rank_vs_pos",
    ]

    # Defense matchup (detailed) — Vegas implied_team_total excluded;
    # it belongs in the Weather NN feature set only (weather_features.py).
    cols += [
        "opp_def_sacks_L5", "opp_def_pass_yds_allowed_L5",
        "opp_def_pass_td_allowed_L5", "opp_def_ints_L5",
        "opp_def_rush_yds_allowed_L5", "opp_def_pts_allowed_L5",
    ]

    # Contextual
    cols += ["is_home", "week", "is_returning_from_absence", "days_rest",
             "practice_status", "game_status", "depth_chart_rank"]

    # Position encoding
    cols += ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]

    return cols


def get_attn_static_columns(all_feature_cols: list[str]) -> list[str]:
    """Return feature columns for the attention NN, excluding temporal aggregations.

    The attention mechanism learns its own temporal representation from raw game
    history, so rolling/EWMA/trend/share features are redundant (and collinear).
    Prior-season features are kept — they cover a different season than the
    attention's within-season lookback.
    """
    exclude_prefixes = ("rolling_", "ewma_", "trend_")
    exclude_exact = set()
    for w in SHARE_WINDOWS:
        exclude_exact.update([f"target_share_L{w}", f"carry_share_L{w}"])
    return [c for c in all_feature_cols
            if not c.startswith(exclude_prefixes) and c not in exclude_exact]


def fill_nans_safe(train_df, val_df, test_df, feature_cols) -> tuple:
    """Fill NaNs using ONLY training set statistics. Called AFTER temporal_split()."""
    # Step 1: Player-level means from training set
    player_feature_means = train_df.groupby("player_id")[feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in feature_cols:
            mask = split_df[col].isna()
            if mask.any():
                split_df.loc[mask, col] = split_df.loc[mask, "player_id"].map(
                    player_feature_means[col]
                )

    # Step 2: Position-level mean from TRAINING SET ONLY
    pos_means = train_df.groupby("position")[feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in feature_cols:
            for pos in ["QB", "RB", "WR", "TE"]:
                mask = (split_df[col].isna()) & (split_df["position"] == pos)
                if mask.any() and pos in pos_means.index:
                    split_df.loc[mask, col] = pos_means.loc[pos, col]

    # Step 3: Zero fill remaining
    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = split_df[feature_cols].fillna(0)

    return train_df, val_df, test_df
