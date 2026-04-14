import pandas as pd


def filter_to_wr(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to WR rows only."""
    wr_df = df[df["position"] == "WR"].copy()
    pos_cols = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    wr_df.drop(columns=[c for c in pos_cols if c in wr_df.columns], inplace=True)
    return wr_df


def compute_team_wr_totals(full_wr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level WR totals for share features."""
    team_wr_totals = full_wr_df.groupby(["recent_team", "season", "week"]).agg(
        team_wr_targets=("targets", "sum"),
    ).reset_index()
    return team_wr_totals
