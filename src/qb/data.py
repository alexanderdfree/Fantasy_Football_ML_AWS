import pandas as pd


def filter_to_position(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to QB rows only."""
    qb_df = df[df["position"] == "QB"].copy()
    pos_cols = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    qb_df.drop(columns=[c for c in pos_cols if c in qb_df.columns], inplace=True)
    return qb_df
