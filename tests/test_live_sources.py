"""Unit tests for the non-ESPN live signals (src/serving/live_sources.py):
practice_status (nflverse primary + Sleeper fallback) and current-season
contracts. Pure parsers + monkeypatched fetchers — no network.
"""

import pandas as pd
import pytest

from src.serving import live_sources


@pytest.mark.unit
def test_nflverse_practice_map_encodes_worst_per_player_and_filters_week(monkeypatch):
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2024],
            "week": [1, 1, 2, 1],
            "gsis_id": ["A", "A", "B", "C"],
            "practice_status": [
                "Limited Participation in Practice",  # A wk1 -> 1
                "Did Not Participate In Practice",  # A wk1 -> 0 (worst wins)
                "Full Participation in Practice",  # B wk2 (filtered out)
                "Did Not Participate In Practice",  # C 2024 (filtered out)
            ],
        }
    )
    monkeypatch.setattr(live_sources.nfl_source, "injuries", lambda seasons: df)
    m = live_sources._nflverse_practice_map(2025, 1)
    assert m == {"A": 0.0}  # worst-per-player; wrong week/season excluded


@pytest.mark.unit
def test_parse_sleeper_practice_maps_known_skips_unknown_and_missing_gsis():
    payload = {
        "1": {"gsis_id": "A", "practice_participation": "Limited"},
        "2": {"gsis_id": "B", "practice_participation": "Did Not Participate"},
        "3": {"gsis_id": "E", "practice_participation": "Full"},
        "4": {"gsis_id": "C", "practice_participation": None},  # null -> skip
        "5": {"gsis_id": None, "practice_participation": "Full"},  # no gsis -> skip
        "6": {"gsis_id": "D", "practice_participation": "weird"},  # unknown -> skip
    }
    assert live_sources._parse_sleeper_practice(payload) == {"A": 1.0, "B": 0.0, "E": 2.0}


@pytest.mark.unit
def test_sleeper_practice_to_num_vocabulary():
    f = live_sources._sleeper_practice_to_num
    assert f("Full") == 2.0 and f("FP") == 2.0
    assert f("Limited") == 1.0 and f("LP") == 1.0
    assert f("DNP") == 0.0 and f("Did Not Participate") == 0.0
    assert f(None) is None and f("") is None and f("questionable") is None


@pytest.mark.unit
def test_fetch_practice_prefers_nflverse_and_skips_sleeper(monkeypatch):
    monkeypatch.setattr(live_sources, "_nflverse_practice_map", lambda s, w: {"X": 1.0})

    def _boom():
        raise AssertionError("Sleeper fallback must not run when nflverse has data")

    monkeypatch.setattr(live_sources, "_sleeper_practice_map", _boom)
    assert live_sources.fetch_practice_status_map(2025, 1) == {"X": 1.0}


@pytest.mark.unit
def test_fetch_practice_falls_back_to_sleeper_in_season(monkeypatch):
    monkeypatch.setattr(live_sources, "_nflverse_practice_map", lambda s, w: {})
    monkeypatch.setattr(live_sources, "_IN_SEASON_MONTHS", frozenset(range(1, 13)))
    monkeypatch.setattr(live_sources, "_sleeper_practice_map", lambda: {"Y": 0.0})
    assert live_sources.fetch_practice_status_map(2025, 1) == {"Y": 0.0}


@pytest.mark.unit
def test_fetch_practice_offseason_skips_sleeper(monkeypatch):
    monkeypatch.setattr(live_sources, "_nflverse_practice_map", lambda s, w: {})
    monkeypatch.setattr(live_sources, "_IN_SEASON_MONTHS", frozenset())  # never in-season

    def _boom():
        raise AssertionError("Sleeper must not run out of season")

    monkeypatch.setattr(live_sources, "_sleeper_practice_map", _boom)
    assert live_sources.fetch_practice_status_map(2025, 1) == {}


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
