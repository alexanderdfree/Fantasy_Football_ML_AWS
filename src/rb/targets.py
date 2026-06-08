"""Raw-stat RB prediction targets.

After the target migration, RB models predict raw NFL stats and
``src.shared.aggregate_targets.predictions_to_fantasy_points`` converts them to
fantasy points post-prediction. No rolling adjustments — fumbles_lost is now
a direct target.
"""

import pandas as pd

from src.config import SCORING
from src.rb.config import POSITION_CONFIG
from src.shared.aggregate_targets import predictions_to_fantasy_points


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Emit the 6 raw-stat RB prediction targets.

    Columns:
      - rushing_tds
      - receiving_tds
      - rushing_yards
      - receiving_yards
      - receptions
      - fumbles_lost = sack_fumbles_lost + rushing_fumbles_lost +
        receiving_fumbles_lost

    Args:
        df: DataFrame filtered to RB rows only, with raw stat columns available.
    """
    df = df.copy()

    for col in ("rushing_tds", "receiving_tds", "rushing_yards", "receiving_yards", "receptions"):
        df[col] = df[col].fillna(0)
    df["fumbles_lost"] = (
        df["sack_fumbles_lost"].fillna(0)
        + df["rushing_fumbles_lost"].fillna(0)
        + df["receiving_fumbles_lost"].fillna(0)
    )

    # Sanity check: aggregator reproduces the RB slice of fantasy_points (PPR).
    # The full fantasy_points column carries passing terms that RBs don't
    # predict, so we back those out before comparing. The aggregator output
    # stays as a local variable (matching QB's compute_targets) rather than
    # a phantom DataFrame column — downstream code only consumes the named
    # targets above.
    preds = {t: df[t].to_numpy() for t in POSITION_CONFIG.targets}
    fantasy_points_check = predictions_to_fantasy_points("RB", preds, "ppr")

    # Decomposition uses src.config.SCORING so a single source of truth
    # changes here when scoring constants are tweaked (mirrors WR/QB/TE).
    passing_component = (
        df["passing_yards"].fillna(0) * SCORING["passing_yards"]
        + df["passing_tds"].fillna(0) * SCORING["passing_tds"]
        + df["interceptions"].fillna(0) * SCORING["interceptions"]
    )
    # 2pt conversions (flat 2 pts each, no SCORING key) land in the upstream
    # fantasy_points column but are NOT model targets — RB genuinely doesn't
    # predict them. Add them to the expected-points side so this *diagnostic*
    # check doesn't spuriously WARN on the 1-2 rows/season with a 2pt
    # conversion. This touches only the discrepancy check, never the targets.
    two_pt_component = (
        df["rushing_2pt_conversions"].fillna(0) * 2 + df["receiving_2pt_conversions"].fillna(0) * 2
    )
    discrepancy = (
        df["fantasy_points"] - fantasy_points_check - passing_component - two_pt_component
    ).abs()
    if (discrepancy > 0.01).any():
        n_bad = int((discrepancy > 0.01).sum())
        print(f"WARNING: {n_bad} RB rows have target decomposition discrepancy > 0.01 pts")

    if "fantasy_points_ppr" in df.columns:
        nfl_discrepancy = (df["fantasy_points"] - df["fantasy_points_ppr"]).abs()
        n_nfl_mismatch = int((nfl_discrepancy > 0.5).sum())
        if n_nfl_mismatch > 0:
            print(
                f"INFO: {n_nfl_mismatch} rows differ from nflverse fantasy_points_ppr by > 0.5 pts"
            )

    return df
