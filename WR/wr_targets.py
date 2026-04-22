import pandas as pd

from shared.aggregate_targets import predictions_to_fantasy_points


def compute_wr_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 raw-stat prediction targets for WR rows.

    Targets (raw NFL stats; fantasy points are aggregated downstream via
    ``shared.aggregate_targets.predictions_to_fantasy_points``):

      - receiving_tds: raw receiving TD count
      - receiving_yards: raw receiving yards
      - receptions: raw reception count
      - fumbles_lost: sack_fumbles_lost + rushing_fumbles_lost +
        receiving_fumbles_lost

    Rushing targets are intentionally dropped — WR rushing stats are too
    sparse to carry reliable signal; noise outweighs gain.
    """
    df = df.copy()

    df["receiving_tds"] = df["receiving_tds"].fillna(0)
    df["receiving_yards"] = df["receiving_yards"].fillna(0)
    df["receptions"] = df["receptions"].fillna(0)
    df["fumbles_lost"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )

    # Sanity check: aggregator-driven fantasy points plus the omitted
    # rushing component must equal the upstream fantasy_points column.
    if "fantasy_points" in df.columns:
        preds = {
            "receiving_tds": df["receiving_tds"].values,
            "receiving_yards": df["receiving_yards"].values,
            "receptions": df["receptions"].values,
            "fumbles_lost": df["fumbles_lost"].values,
        }
        wr_component = predictions_to_fantasy_points("WR", preds, "ppr")
        rushing_component = df["rushing_yards"].fillna(0) * 0.1 + df["rushing_tds"].fillna(0) * 6
        passing_component = (
            df["passing_yards"].fillna(0) * 0.04
            + df["passing_tds"].fillna(0) * 4
            + df["interceptions"].fillna(0) * -2
        )
        discrepancy = (
            df["fantasy_points"] - wr_component - rushing_component - passing_component
        ).abs()
        if (discrepancy > 0.01).any():
            n_bad = (discrepancy > 0.01).sum()
            print(f"WARNING: {n_bad} WR rows have target decomposition discrepancy > 0.01 pts")

    return df
