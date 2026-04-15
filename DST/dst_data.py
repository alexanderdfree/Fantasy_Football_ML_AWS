import pandas as pd
import numpy as np
from src.config import CACHE_DIR, SEASONS


def build_dst_data() -> pd.DataFrame:
    """Build team-level D/ST data from schedule scores and player stats.

    Strategy:
      - Points allowed: from schedule game scores (complete for all teams/seasons)
      - Sacks forced: derived from opponent offensive sacks suffered (complete)
      - INTs forced: derived from opponent offensive INTs thrown (complete)
      - Fumble recoveries: derived from opponent fumbles lost (complete)
      - Defensive TDs: from individual defensive player stats (partial; fill 0)
      - Safeties: from individual defensive player stats (partial; fill 0)
      - Special teams TDs: from player stats per team (mostly complete)
    """
    weekly = pd.read_parquet(f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet")
    schedules = pd.read_parquet(f"{CACHE_DIR}/schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet")
    schedules_reg = schedules[schedules["game_type"] == "REG"].copy()

    # --- 1. Points allowed from schedule scores ---
    away_pts = schedules_reg[["season", "week", "away_team", "home_score"]].copy()
    away_pts.columns = ["season", "week", "team", "points_allowed"]
    home_pts = schedules_reg[["season", "week", "home_team", "away_score"]].copy()
    home_pts.columns = ["season", "week", "team", "points_allowed"]
    pts_allowed = pd.concat([away_pts, home_pts], ignore_index=True)

    # --- 2. Home/away indicator + Vegas lines from schedule ---
    home_info = schedules_reg[
        ["season", "week", "home_team", "spread_line", "total_line"]
    ].copy()
    home_info.columns = ["season", "week", "team", "spread_line", "total_line"]
    home_info["is_home"] = 1

    away_info = schedules_reg[
        ["season", "week", "away_team", "spread_line", "total_line"]
    ].copy()
    away_info.columns = ["season", "week", "team", "spread_line", "total_line"]
    away_info["is_home"] = 0

    context = pd.concat([home_info, away_info], ignore_index=True)

    # --- 3. Opponent team mapping ---
    home_opp = schedules_reg[["season", "week", "home_team", "away_team"]].copy()
    home_opp.columns = ["season", "week", "team", "opponent_team"]
    away_opp = schedules_reg[["season", "week", "away_team", "home_team"]].copy()
    away_opp.columns = ["season", "week", "team", "opponent_team"]
    opp_map = pd.concat([home_opp, away_opp], ignore_index=True)

    # --- 4. Defensive stats derived from opponent's offensive data ---
    # Sacks forced = opponent QBs/players sacks suffered
    # INTs forced = opponent QBs interceptions thrown
    # Fumble recoveries = opponent fumbles lost
    weekly_copy = weekly.copy()
    weekly_copy["_total_fumbles_lost"] = (
        weekly_copy["sack_fumbles_lost"].fillna(0)
        + weekly_copy["rushing_fumbles_lost"].fillna(0)
        + weekly_copy["receiving_fumbles_lost"].fillna(0)
    )

    def_from_offense = weekly_copy.groupby(["opponent_team", "season", "week"]).agg(
        def_sacks=("sacks", "sum"),
        def_ints=("interceptions", "sum"),
        def_fumble_rec=("_total_fumbles_lost", "sum"),
    ).reset_index()
    def_from_offense.columns = ["team", "season", "week", "def_sacks", "def_ints", "def_fumble_rec"]

    # --- 5. Defensive TDs + safeties from individual defensive player stats ---
    def_positions = [
        "LB", "CB", "DE", "SAF", "DT", "OLB", "FS", "MLB",
        "NT", "ILB", "DB", "DL", "SS", "S",
    ]
    def_players = weekly[weekly["position"].isin(def_positions)].copy()
    def_td_safety = def_players.groupby(["recent_team", "season", "week"]).agg(
        def_tds=("def_tds", "sum"),
        def_safeties=("def_safeties", "sum"),
    ).reset_index()
    def_td_safety.columns = ["team", "season", "week", "def_tds", "def_safeties"]

    # --- 6. Special teams TDs from all players on the team ---
    st_tds = weekly.groupby(["recent_team", "season", "week"]).agg(
        special_teams_tds=("special_teams_tds", "sum"),
    ).reset_index()
    st_tds.columns = ["team", "season", "week", "special_teams_tds"]

    # --- 7. Team offensive scoring (for opponent strength features) ---
    home_scoring = schedules_reg[["season", "week", "home_team", "home_score"]].copy()
    home_scoring.columns = ["season", "week", "team", "team_score"]
    away_scoring = schedules_reg[["season", "week", "away_team", "away_score"]].copy()
    away_scoring.columns = ["season", "week", "team", "team_score"]
    team_scoring = pd.concat([home_scoring, away_scoring], ignore_index=True)

    # --- Merge everything into a single team-week DataFrame ---
    dst_df = pts_allowed.copy()
    dst_df = dst_df.merge(context, on=["season", "week", "team"], how="left")
    dst_df = dst_df.merge(opp_map, on=["season", "week", "team"], how="left")
    dst_df = dst_df.merge(def_from_offense, on=["team", "season", "week"], how="left")
    dst_df = dst_df.merge(def_td_safety, on=["team", "season", "week"], how="left")
    dst_df = dst_df.merge(st_tds, on=["team", "season", "week"], how="left")

    # Merge opponent scoring history
    team_scoring_sorted = team_scoring.sort_values(["team", "season", "week"])
    team_scoring_sorted["scoring_L5"] = team_scoring_sorted.groupby(
        ["team", "season"]
    )["team_score"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    opp_scoring = team_scoring_sorted[["team", "season", "week", "scoring_L5"]].copy()
    opp_scoring.columns = ["opponent_team", "season", "week", "opp_scoring_L5"]
    dst_df = dst_df.merge(opp_scoring, on=["opponent_team", "season", "week"], how="left")

    # Fill missing values
    for col in ["def_sacks", "def_ints", "def_fumble_rec", "def_tds",
                 "def_safeties", "special_teams_tds"]:
        dst_df[col] = dst_df[col].fillna(0)
    for col in ["spread_line", "total_line"]:
        dst_df[col] = dst_df[col].fillna(dst_df[col].median())
    dst_df["is_home"] = dst_df["is_home"].fillna(0)
    dst_df["opp_scoring_L5"] = dst_df["opp_scoring_L5"].fillna(
        dst_df["points_allowed"].mean()
    )

    # Add pipeline-compatible columns
    dst_df["player_id"] = dst_df["team"]
    dst_df["player_display_name"] = dst_df["team"] + " D/ST"
    dst_df["player_name"] = dst_df["team"]
    dst_df["recent_team"] = dst_df["team"]
    dst_df["position"] = "DST"
    dst_df["headshot_url"] = ""

    dst_df = dst_df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    return dst_df


def filter_to_dst(df: pd.DataFrame) -> pd.DataFrame:
    """Identity filter — D/ST data is pre-built at team level."""
    return df.copy()
