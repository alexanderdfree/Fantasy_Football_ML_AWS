import pandas as pd
import numpy as np
from src.config import (
    ROLLING_WINDOWS, ROLL_STATS, ROLL_AGGS, EWMA_STATS, EWMA_SPANS,
    TREND_STATS, SHARE_WINDOWS, OPP_ROLLING_WINDOW, MIN_GAMES_PER_SEASON,
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
    # Team totals (computed BEFORE min-games filter)
    team_totals = df.groupby(["recent_team", "season", "week"]).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
    ).reset_index()
    df = df.merge(team_totals, on=["recent_team", "season", "week"], how="left")

    # Min-games filter
    games_per_season = df.groupby(["player_id", "season"])["week"].transform("count")
    df = df[games_per_season >= MIN_GAMES_PER_SEASON].copy()

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

    # air_yards_share
    if "receiving_air_yards" in df.columns:
        team_air_yards = df.groupby(["recent_team", "season", "week"])[
            "receiving_air_yards"
        ].transform("sum")
        df["air_yards_share"] = (df["receiving_air_yards"] / team_air_yards).fillna(0)
    else:
        df["air_yards_share"] = 0.0

    # Clean up intermediate columns
    drop_cols = [c for c in df.columns if c.startswith((
        "player_targets_roll", "team_targets_roll",
        "player_carries_roll", "team_carries_roll",
    ))]
    drop_cols += ["team_targets", "team_carries", "team_changed", "stint_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- Matchup / Opponent Features (4) ---
    df = _build_matchup_features(df)

    # --- Contextual Features (4) ---
    df = _build_contextual_features(df)

    # --- Position Encoding (4) ---
    for pos in ["QB", "RB", "WR", "TE"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)

    return df


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

    # Contextual
    cols += ["is_home", "week", "is_returning_from_absence", "days_rest"]

    # Position encoding
    cols += ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]

    return cols


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
