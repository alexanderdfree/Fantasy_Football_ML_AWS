import pandas as pd


def filter_to_rb(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to RB rows only.

    Must be called AFTER build_features() and AFTER temporal_split()
    so all team-level and opponent-level features are correctly computed
    from the full-position dataset.
    """
    rb_df = df[df["position"] == "RB"].copy()

    # Drop position encoding columns (all RB, no variance)
    pos_cols = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    rb_df.drop(columns=[c for c in pos_cols if c in rb_df.columns], inplace=True)

    return rb_df


def compute_team_rb_totals(full_rb_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level RB totals for share features.

    Args:
        full_rb_df: All RB rows (before min-games filter), from the general pipeline.
    """
    team_rb_totals = (
        full_rb_df.groupby(["recent_team", "season", "week"])
        .agg(
            team_rb_carries=("carries", "sum"),
            team_rb_targets=("targets", "sum"),
        )
        .reset_index()
    )
    return team_rb_totals
