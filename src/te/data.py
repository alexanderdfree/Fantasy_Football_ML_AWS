import pandas as pd


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to TE rows only."""
    te_df = df[df["position"] == "TE"].copy()
    pos_cols = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    te_df.drop(columns=[c for c in pos_cols if c in te_df.columns], inplace=True)
    return te_df


def compute_team_te_totals(full_te_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level TE totals for share features."""
    team_te_totals = (
        full_te_df.groupby(["recent_team", "season", "week"])
        .agg(
            team_te_targets=("targets", "sum"),
        )
        .reset_index()
    )
    return team_te_totals
