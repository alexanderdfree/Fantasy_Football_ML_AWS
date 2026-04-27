"""QB prediction targets (raw-stat migration).

After migration the model predicts six raw NFL stats and fantasy points are
aggregated after prediction via ``src.shared.aggregate_targets``. The only non-
identity target is ``fumbles_lost``, which sums all three fumble categories
(``sack_fumbles_lost`` + ``rushing_fumbles_lost`` + ``receiving_fumbles_lost``)
so a single -2 pts penalty applies regardless of how the fumble occurred.
"""

import pandas as pd

from src.QB.qb_config import QB_TARGETS
from src.shared.aggregate_targets import predictions_to_fantasy_points


def compute_qb_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Emit the 6 raw-stat prediction targets for QB rows.

    Targets (identity unless noted):
      - ``passing_yards``
      - ``rushing_yards``
      - ``passing_tds``
      - ``rushing_tds``
      - ``interceptions``
      - ``fumbles_lost`` = ``sack_fumbles_lost`` + ``rushing_fumbles_lost`` +
        ``receiving_fumbles_lost``

    A sanity check compares the aggregator output against ``fantasy_points`` on
    the same rows. QB scope excludes receiving yards/TDs/receptions (trick-play
    signal only); those contributions show up as a positive-scoring residual.
    """
    df = df.copy()

    for col in ("passing_yards", "rushing_yards", "passing_tds", "rushing_tds", "interceptions"):
        df[col] = df[col].fillna(0)
    df["fumbles_lost"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )

    preds_truth = {t: df[t].to_numpy() for t in QB_TARGETS}
    fantasy_points_check = predictions_to_fantasy_points("QB", preds_truth, "ppr")
    receiving_component = (
        df["receptions"].fillna(0) * 1.0
        + df["receiving_yards"].fillna(0) * 0.1
        + df["receiving_tds"].fillna(0) * 6
    )
    discrepancy = (df["fantasy_points"] - fantasy_points_check - receiving_component).abs()
    if (discrepancy > 0.01).any():
        n_bad = (discrepancy > 0.01).sum()
        print(f"WARNING: {n_bad} QB rows have target decomposition discrepancy > 0.01 pts")

    return df
