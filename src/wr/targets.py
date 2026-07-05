import pandas as pd

from src.config import SCORING
from src.shared.aggregate_targets import predictions_to_fantasy_points


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 raw-stat prediction targets for WR rows.

    Targets (raw NFL stats; fantasy points are aggregated downstream via
    ``src.shared.aggregate_targets.predictions_to_fantasy_points``):

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
        # Decomposition uses src.config.SCORING so a single source of
        # truth changes here when scoring constants are tweaked.
        rushing_component = (
            df["rushing_yards"].fillna(0) * SCORING["rushing_yards"]
            + df["rushing_tds"].fillna(0) * SCORING["rushing_tds"]
        )
        passing_component = (
            df["passing_yards"].fillna(0) * SCORING["passing_yards"]
            + df["passing_tds"].fillna(0) * SCORING["passing_tds"]
            + df["interceptions"].fillna(0) * SCORING["interceptions"]
        )
        # 2pt conversions (flat 2 pts each, no SCORING key) land in the
        # upstream fantasy_points column but are NOT model targets — back
        # them out so this *diagnostic* check doesn't spuriously WARN
        # (mirrors src/rb/targets.py). Column-guarded: preprocessing
        # zero-fills only the rushing/receiving 2pt columns. This touches
        # only the discrepancy check, never the targets.
        two_pt_component = pd.Series(0.0, index=df.index)
        for col in (
            "passing_2pt_conversions",
            "rushing_2pt_conversions",
            "receiving_2pt_conversions",
        ):
            if col in df.columns:
                two_pt_component = two_pt_component + df[col].fillna(0) * 2
        discrepancy = (
            df["fantasy_points"]
            - wr_component
            - rushing_component
            - passing_component
            - two_pt_component
        ).abs()
        if (discrepancy > 0.01).any():
            n_bad = (discrepancy > 0.01).sum()
            print(f"WARNING: {n_bad} WR rows have target decomposition discrepancy > 0.01 pts")

    if "fantasy_points_ppr" in df.columns:
        nfl_discrepancy = (df["fantasy_points"] - df["fantasy_points_ppr"]).abs()
        n_nfl_mismatch = int((nfl_discrepancy > 0.5).sum())
        if n_nfl_mismatch > 0:
            print(
                f"INFO: {n_nfl_mismatch} rows differ from nflverse fantasy_points_ppr by > 0.5 pts"
            )

    return df
