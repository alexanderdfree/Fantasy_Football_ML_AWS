"""Current-season contract features for live projection builds.

Practice reports are handled by src.serving.practice_reports, which verifies
per-team coverage against the official NFL report.
"""

from __future__ import annotations

import pandas as pd

from src.data import nfl_source
from src.data.external_sources import CONTRACT_FEATURE_COLUMNS, derive_active_contracts


def fetch_contract_features(season: int) -> pd.DataFrame:
    """Current-season ``contract_*`` features indexed by ``player_id``.

    Reuses the training deriver (``derive_active_contracts``) over the live OTC
    feed (``nfl_source.contracts()``), so 2026 rows get the active-as-of-2026
    contract (advancing ``contract_age``/``contract_years_remaining`` vs a stale
    carry-forward). Empty frame (right columns) on failure. Never raises.
    """
    cols = list(CONTRACT_FEATURE_COLUMNS)
    try:
        derived = derive_active_contracts(nfl_source.contracts(), [season])
    except Exception as e:  # noqa: BLE001 - network/data boundary
        print(f"[live_sources] contracts derive failed: {e!r}")
        return pd.DataFrame(columns=cols)
    if derived is None or derived.empty:
        return pd.DataFrame(columns=cols)
    return derived.set_index("player_id")[cols]
