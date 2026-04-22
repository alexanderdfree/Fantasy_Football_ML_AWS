import pandas as pd

from shared.aggregate_targets import predictions_to_fantasy_points

_TE_RAW_TARGETS = ("receiving_tds", "receiving_yards", "receptions", "fumbles_lost")


def compute_te_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 4 raw-stat prediction targets for TE rows.

    Targets:
      - receiving_tds
      - receiving_yards
      - receptions
      - fumbles_lost (sack_fumbles_lost + rushing_fumbles_lost +
        receiving_fumbles_lost)

    Fantasy points are aggregated post-prediction via
    ``predictions_to_fantasy_points("TE", ...)``. Rushing targets are dropped
    because TE rushing stats are near-zero (noise > signal).
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

    # Sanity check: aggregated TE fantasy points plus any residual passing /
    # rushing contribution (near-zero on proper TE rows) should match the
    # pre-computed fantasy_points column. Flag rows where they diverge to
    # catch upstream data corruption.
    if "fantasy_points" in df.columns:
        preds = {t: df[t].values for t in _TE_RAW_TARGETS}
        te_points = predictions_to_fantasy_points("TE", preds, "ppr")
        residual = df["fantasy_points"] - te_points
        for col, weight in (
            ("passing_yards", 0.04),
            ("passing_tds", 4),
            ("interceptions", -2),
            ("rushing_yards", 0.1),
            ("rushing_tds", 6),
        ):
            if col in df.columns:
                residual = residual - df[col].fillna(0) * weight
        n_bad = (residual.abs() > 0.01).sum()
        if n_bad:
            print(f"WARNING: {n_bad} TE rows have target decomposition discrepancy > 0.01 pts")

    return df
