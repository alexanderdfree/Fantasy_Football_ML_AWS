"""Unit tests for the pure transforms in ``src.data.external_sources``.

Covers the two non-trivial bits without network: the contract
"active-as-of-season" ``merge_asof`` derivation and the ESPN-id → gsis QBR
bridge. The cached fetch wrappers (``load_*``) are exercised via mocks in
``tests/test_data_loader.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.external_sources import (
    CONTRACT_FEATURE_COLUMNS,
    QBR_FEATURE_COLUMNS,
    bridge_qbr_to_gsis,
    derive_active_contracts,
)


@pytest.mark.unit
def test_derive_active_contracts_picks_latest_signed_on_or_before_season():
    """Each (player, season) takes the contract with the largest
    ``year_signed <= season``; age + years_remaining derive from it."""
    contracts = pd.DataFrame(
        {
            "gsis_id": ["00-A", "00-A", "00-B", None],
            "year_signed": [2020, 2023, 2021, 2022],
            "years": [4.0, 3.0, 2.0, 5.0],
            "guaranteed": [20.0, 40.0, 8.0, 99.0],
            "apy_cap_pct": [0.05, 0.10, 0.03, 0.5],
        }
    )
    out = derive_active_contracts(contracts, [2022, 2023, 2024])

    assert list(out.columns) == ["player_id", "season", *CONTRACT_FEATURE_COLUMNS]
    # Null-gsis contract is dropped entirely.
    assert set(out["player_id"]) == {"00-A", "00-B"}

    def row(pid, season):
        return out[(out["player_id"] == pid) & (out["season"] == season)].iloc[0]

    # A@2022 → the 2020 contract (2023 not yet signed): age 2, 4-2 remaining.
    a22 = row("00-A", 2022)
    assert a22["contract_apy_cap_pct"] == 0.05
    assert a22["contract_age"] == 2
    assert a22["contract_years_remaining"] == 2
    # A@2023 → the freshly-signed 2023 contract: age 0, full 3 remaining.
    a23 = row("00-A", 2023)
    assert a23["contract_apy_cap_pct"] == 0.10
    assert a23["contract_age"] == 0
    assert a23["contract_years_remaining"] == 3
    # B@2024 → 2021 contract long expired: years_remaining clamps at 0 (not -1).
    b24 = row("00-B", 2024)
    assert b24["contract_years_remaining"] == 0


@pytest.mark.unit
def test_derive_active_contracts_absent_before_first_signing():
    """A season before the player's earliest contract yields no row (the
    downstream left-merge + fillna(0) covers it)."""
    contracts = pd.DataFrame(
        {
            "gsis_id": ["00-A"],
            "year_signed": [2023],
            "years": [3.0],
            "guaranteed": [10.0],
            "apy_cap_pct": [0.08],
        }
    )
    out = derive_active_contracts(contracts, [2021, 2022, 2023])
    assert set(out["season"]) == {2023}  # 2021/2022 absent (no contract yet)


@pytest.mark.unit
def test_derive_active_contracts_missing_columns_returns_empty():
    out = derive_active_contracts(pd.DataFrame({"gsis_id": ["00-A"]}), [2023])
    assert out.empty
    assert list(out.columns) == ["player_id", "season", *CONTRACT_FEATURE_COLUMNS]


@pytest.mark.unit
def test_bridge_qbr_to_gsis_maps_espn_id_and_filters_regular_season():
    """ESPN player_id → gsis via the crosswalk; playoff rows and unmatched
    ids are dropped; week comes from week_num; output is merge-ready."""
    qbr = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "season_type": ["Regular", "Playoffs", "Regular"],
            "game_week": [1, 1, 2],
            "week_num": [1, 1, 2],
            "player_id": [100, 100, 999],  # ESPN ids; 999 has no crosswalk entry
            "qbr_total": [70.0, 88.0, 50.0],
            "pts_added": [5.0, 9.0, 1.0],
        }
    )
    ids = pd.DataFrame({"espn_id": [100.0], "gsis_id": ["00-A"]})

    out = bridge_qbr_to_gsis(qbr, ids)

    assert list(out.columns) == ["player_id", "season", "week", *QBR_FEATURE_COLUMNS]
    # Only the matched, regular-season row survives.
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "00-A"
    assert out.iloc[0]["week"] == 1
    assert out.iloc[0]["qbr_total"] == 70.0
    assert "00-A" not in {  # playoff row excluded (would have been week 1 too)
        r for r in out[out["qbr_total"] == 88.0]["player_id"]
    }


@pytest.mark.unit
def test_bridge_qbr_to_gsis_empty_inputs_return_empty():
    cols = ["player_id", "season", "week", *QBR_FEATURE_COLUMNS]
    assert list(bridge_qbr_to_gsis(pd.DataFrame(), pd.DataFrame()).columns) == cols
    # ids without espn_id → empty (can't bridge).
    qbr = pd.DataFrame(
        {
            "season": [2023],
            "season_type": ["Regular"],
            "week_num": [1],
            "player_id": [100],
            "qbr_total": [70.0],
            "pts_added": [5.0],
        }
    )
    assert bridge_qbr_to_gsis(qbr, pd.DataFrame({"gsis_id": ["00-A"]})).empty


@pytest.mark.unit
def test_bridge_qbr_to_gsis_no_fanout_on_duplicate_espn_ids():
    """A crosswalk with duplicate espn_id rows must not multiply QBR rows."""
    qbr = pd.DataFrame(
        {
            "season": [2023],
            "season_type": ["Regular"],
            "week_num": [3],
            "player_id": [100],
            "qbr_total": [60.0],
            "pts_added": [2.0],
        }
    )
    ids = pd.DataFrame({"espn_id": [100.0, 100.0], "gsis_id": ["00-A", "00-A"]})
    out = bridge_qbr_to_gsis(qbr, ids)
    assert len(out) == 1
