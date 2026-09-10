"""Interpret the two source eras of nflverse weekly roster availability."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def weekly_roster_status(rosters: pd.DataFrame) -> pd.Series:
    """Return week-specific ACT/INA/RES/etc., or UNKNOWN where unavailable.

    Before 2016 nflverse builds weekly rosters from NFL Data Exchange, then
    overwrites ``status`` with a season-level Shield lookup joined only by player.
    Its preserved weekly descriptor has A01 (active) / I01 (inactive). Since
    2016 ``status`` comes from the weekly NGS roster and is authoritative: the
    descriptor can be backfilled from player metadata and must not override it.
    See https://github.com/nflverse/nflverse-rosters/blob/main/R/rosters.R and
    R/rosters_ngs.R. Unknown legacy descriptors never fall back to season status.
    """
    result = rosters["status"].fillna("UNKNOWN").astype(str).copy()
    legacy = pd.to_numeric(rosters["season"], errors="coerce").lt(2016)
    if legacy.any():
        if "status_description_abbr" in rosters:
            descriptor = rosters.loc[legacy, "status_description_abbr"]
            result.loc[legacy] = descriptor.map({"A01": "ACT", "I01": "INA"}).fillna("UNKNOWN")
        else:
            result.loc[legacy] = "UNKNOWN"
        unknown = legacy & result.eq("UNKNOWN")
        if unknown.any():
            logger.warning(
                "inheritance: %d legacy roster rows lack a supported weekly descriptor; "
                "availability is unknown, not inferred from season-level status",
                int(unknown.sum()),
            )
    return result
