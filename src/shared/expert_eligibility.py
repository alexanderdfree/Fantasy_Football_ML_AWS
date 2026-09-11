"""Source eligibility for evaluation, independently of forecast cache vintage.

Keep raw archives readable for provenance diagnostics, but never grade known
backfilled NFL.com offense as a forecast. Apply this at projection and metric
boundaries so injected or previously cached totals cannot bypass the rule.
"""

import pandas as pd

NFLCOM_OFFENSE_MIN_SEASON = 2024
NFLCOM_ELIGIBILITY_NOTE = (
    "NFL.com offensive forecasts before 2024 are excluded: the archive backfilled actuals."
)
_OFFENSE = frozenset({"QB", "RB", "WR", "TE"})


def eligible_forecast_rows(frame: pd.DataFrame, source: str, position: str) -> pd.Series:
    """Mask usable forecast rows; unknown NFL.com seasons fail closed."""
    if source.lower() == "nflcom" and position.upper() in _OFFENSE:
        if "season" not in frame:
            return pd.Series(False, index=frame.index)
        return pd.to_numeric(frame["season"], errors="coerce").ge(NFLCOM_OFFENSE_MIN_SEASON)
    return pd.Series(True, index=frame.index)


def filter_eligible_forecasts(frame: pd.DataFrame, source: str, position: str) -> pd.DataFrame:
    """Return usable rows without modifying the raw archive."""
    return frame.loc[eligible_forecast_rows(frame, source, position)].copy()
