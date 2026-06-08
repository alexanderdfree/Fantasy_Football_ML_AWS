"""Unit tests for the pure transforms in ``src.data.external_sources``.

Covers the two non-trivial bits without network: the contract
"active-as-of-season" ``merge_asof`` derivation and the ESPN-id → gsis QBR
bridge, plus the two cache-key/schema-gate helpers. The cached fetch wrappers
(``load_*``) are exercised via mocks in ``tests/test_data_loader.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.external_sources import (
    CONTRACT_FEATURE_COLUMNS,
    QBR_FEATURE_COLUMNS,
    _cached_parquet_has_columns,
    _seasons_cache_signature,
    bridge_qbr_to_gsis,
    derive_active_contracts,
)


@pytest.mark.unit
def test_derive_active_contracts_picks_latest_signed_strictly_before_season():
    """#645 (leak-safe): each (player, season) takes the contract with the
    largest ``year_signed < season`` (effective the season *after* signing,
    since the integer year can't tell an offseason deal from a mid-season
    signing); age + years_remaining derive from the true ``year_signed``."""
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

    # A@2022 → the 2020 contract (2023 signed in-season, not yet effective):
    # age 2, 4-2 remaining.
    a22 = row("00-A", 2022)
    assert a22["contract_apy_cap_pct"] == 0.05
    assert a22["contract_age"] == 2
    assert a22["contract_years_remaining"] == 2
    # A@2023 → STILL the 2020 contract: the 2023 signing isn't known before the
    # early weeks of 2023, so it only becomes effective in 2024. age 3, 4-3=1.
    a23 = row("00-A", 2023)
    assert a23["contract_apy_cap_pct"] == 0.05
    assert a23["contract_age"] == 3
    assert a23["contract_years_remaining"] == 1
    # A@2024 → now the 2023 contract is effective (signed strictly before 2024):
    # age 1, 3-1=2 remaining.
    a24 = row("00-A", 2024)
    assert a24["contract_apy_cap_pct"] == 0.10
    assert a24["contract_age"] == 1
    assert a24["contract_years_remaining"] == 2
    # B@2024 → 2021 contract long expired: years_remaining clamps at 0 (not -1).
    b24 = row("00-B", 2024)
    assert b24["contract_years_remaining"] == 0


@pytest.mark.unit
def test_derive_active_contracts_drops_year_signed_zero():
    """OTC uses ``year_signed == 0`` as a missing-year placeholder (~1.1k mostly
    retired / practice-squad players). Left in, ``contract_age = season - 0 ≈
    season`` — a wildly out-of-distribution value the live serving override
    (``src.serving.live_sources.fetch_contract_features``) would feed straight to
    inference for any such rostered player. It must be dropped (treated as no
    contract), like a null ``gsis_id`` / ``year_signed``."""
    contracts = pd.DataFrame(
        {
            "gsis_id": ["00-REAL", "00-ZERO"],
            "year_signed": [2022, 0],
            "years": [4.0, 4.0],
            "guaranteed": [50.0, 99.0],
            "apy_cap_pct": [0.10, 0.20],
        }
    )
    out = derive_active_contracts(contracts, [2024])
    assert set(out["player_id"]) == {"00-REAL"}  # year_signed==0 row dropped
    assert (out["contract_age"] < 100).all()  # no season-as-age garbage


@pytest.mark.unit
def test_derive_active_contracts_absent_until_season_after_signing():
    """#645: a season on/before the player's earliest signing yields no row —
    a 2023 signing is effective from 2024 (not known before the early weeks of
    2023). The downstream left-merge + fillna(0) covers the absent seasons; the
    contract still appears once its effective season arrives (not dropped)."""
    contracts = pd.DataFrame(
        {
            "gsis_id": ["00-A"],
            "year_signed": [2023],
            "years": [3.0],
            "guaranteed": [10.0],
            "apy_cap_pct": [0.08],
        }
    )
    out = derive_active_contracts(contracts, [2021, 2022, 2023, 2024])
    # 2021/2022/2023 absent (the 2023 signing isn't effective until 2024).
    assert set(out["season"]) == {2024}
    a24 = out[out["season"] == 2024].iloc[0]
    assert a24["contract_apy_cap_pct"] == 0.08
    assert a24["contract_age"] == 1


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


@pytest.mark.unit
def test_seasons_cache_signature_contiguous_is_inert():
    """A contiguous range renders the legacy ``{min}_{max}`` key verbatim.

    This is the inertness linchpin for the season-cache rekey (#810/#488/#487):
    the production full-range filename (``*_2012_2025.parquet``) and every
    contiguous sub-range stay byte-identical, so load-bearing cache-path
    literals in model_sync / CI / docs are unchanged.
    """
    # Production range — the exact literal the cache filenames depend on.
    assert _seasons_cache_signature(list(range(2012, 2026))) == "2012_2025"
    # Contiguous sub-ranges and single seasons are unchanged too. A single
    # season renders ``{y}_{y}`` — identical to the legacy
    # ``seasons[0]_seasons[-1]`` key, which also doubled the year.
    assert _seasons_cache_signature([2020, 2021, 2022]) == "2020_2022"
    assert _seasons_cache_signature([2023]) == "2023_2023"
    # Order- and duplicate-insensitive (still contiguous as a set).
    assert _seasons_cache_signature([2022, 2020, 2021, 2020]) == "2020_2022"
    # Empty selection has its own sentinel rather than crashing on min/max.
    assert _seasons_cache_signature([]) == "none"


@pytest.mark.unit
def test_seasons_cache_signature_sparse_disambiguates_shared_min_max():
    """Two different non-contiguous selections sharing min/max must not collide.

    The legacy ``seasons[0]_seasons[-1]`` key gave both ``{2018,2020}`` and
    ``{2018,2019,2020}`` the file ``..._2018_2020.parquet`` — one would serve
    the other's cached data. The signature appends ``_{len}_{8hex}`` for sparse
    sets so the two render distinctly.
    """
    a = _seasons_cache_signature([2018, 2020])
    b = _seasons_cache_signature([2018, 2019, 2020])  # contiguous → plain
    assert a != b
    assert a.startswith("2018_2020_") and a != "2018_2020"
    assert b == "2018_2020"
    # Deterministic and set-stable (order/dupes don't change the hash).
    assert a == _seasons_cache_signature([2020, 2018, 2018])


@pytest.mark.unit
def test_cached_parquet_has_columns_gate(tmp_path):
    """The schema-gate returns True only when every required column is present.

    A parquet predating a feature-column add (#428/#548) must be rejected so the
    caller regenerates rather than left-merging + ``fillna(0)``-ing the missing
    column to all-zeros.
    """
    required = ("player_id", "season", "feat_a", "feat_b")

    complete = tmp_path / "complete.parquet"
    pd.DataFrame(
        {"player_id": ["00-A"], "season": [2024], "feat_a": [1.0], "feat_b": [2.0]}
    ).to_parquet(complete)
    assert _cached_parquet_has_columns(str(complete), required) is True

    # Missing ``feat_b`` (added after this cache was written) → regenerate.
    stale = tmp_path / "stale.parquet"
    pd.DataFrame({"player_id": ["00-A"], "season": [2024], "feat_a": [1.0]}).to_parquet(stale)
    assert _cached_parquet_has_columns(str(stale), required) is False
