"""Current-season contract feature tests; practice reports have their own suite."""

import pandas as pd
import pytest

from src.serving import live_sources


@pytest.mark.unit
def test_fetch_contract_features_derives_current_season(monkeypatch):
    # One-row-per-contract OTC shape; derive_active_contracts (real) collapses it.
    contracts = pd.DataFrame(
        {
            "gsis_id": ["A", "B"],
            "year_signed": [2023, 2020],
            "years": [4, 3],
            "guaranteed": [100.0, 50.0],
            "apy_cap_pct": [0.15, 0.08],
        }
    )
    monkeypatch.setattr(live_sources.nfl_source, "contracts", lambda: contracts)
    cf = live_sources.fetch_contract_features(2026)
    assert cf.index.name == "player_id"
    assert set(live_sources.CONTRACT_FEATURE_COLUMNS).issubset(cf.columns)
    # A signed 2023 (effective 2024) → active for 2026; age = 2026-2023 = 3.
    assert cf.loc["A", "contract_age"] == 3
    assert cf.loc["A", "contract_apy_cap_pct"] == 0.15


@pytest.mark.unit
def test_fetch_contract_features_empty_on_failure(monkeypatch):
    def _boom():
        raise RuntimeError("nflverse down")

    monkeypatch.setattr(live_sources.nfl_source, "contracts", _boom)
    cf = live_sources.fetch_contract_features(2026)
    assert cf.empty and list(cf.columns) == list(live_sources.CONTRACT_FEATURE_COLUMNS)
