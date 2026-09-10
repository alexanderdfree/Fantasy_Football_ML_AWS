"""Expert identity joins cannot treat missing IDs as successful matches."""

import numpy as np
import pandas as pd
import pytest

from src.data.identity import valid_player_ids
from src.data.nflcom_loader import _build_roster_lookup

pytestmark = pytest.mark.unit
INVALID_IDS = [None, np.nan, pd.NA, "", "  ", "None", "nan", "null", "<NA>"]


def _rosters(count=2):
    return pd.DataFrame(
        [
            {
                "player_id": f"00-{i}",
                "player_name": f"Player {i}",
                "position": "QB",
                "team": "KC",
                "season": 2024,
            }
            for i in range(count)
        ]
    )


@pytest.fixture(params=["nflcom", "fftoday"])
def provider(request, monkeypatch, tmp_path):
    if request.param == "nflcom":
        from src.data import nflcom_loader as mod
    else:
        from src.analysis import fftoday_loader as mod
    state = {"rosters": _rosters(), "count": 2, "loads": 0, "roster_loads": 0}

    def projections(*args, **kwargs):
        state["loads"] += 1
        frame = pd.DataFrame(
            [
                {
                    "player_name": f"Player {i}",
                    "position": "QB",
                    "season": 2024,
                    "week": 1,
                    "team": "KC",
                    "opponent": "BUF",
                }
                for i in range(state["count"])
            ]
        )
        frame.attrs[mod._FETCH_COMPLETE_ATTR] = True
        return frame

    def rosters(seasons):
        state["roster_loads"] += 1
        return state["rosters"].copy()

    monkeypatch.setattr(mod, f"load_{request.param}_projections", projections)
    monkeypatch.setattr(mod.nfl_source, "rosters", rosters)
    load = getattr(mod, f"load_{request.param}_with_gsis_id")

    def run(threshold=1.0, **kwargs):
        return load([2024], cache_dir=str(tmp_path), min_match_rate=threshold, **kwargs)

    return request.param, run, state, tmp_path


@pytest.mark.parametrize("invalid", INVALID_IDS)
def test_lookup_drops_missing_ids_before_deduplication(invalid):
    good = _rosters()
    bad = good.iloc[[0]].assign(player_id=invalid)
    source = pd.concat([bad, good], ignore_index=True)
    before = source.copy()
    result = _build_roster_lookup(source)
    assert result.player_id.tolist() == ["00-0", "00-1"]
    assert valid_player_ids(result.player_id).all()
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("invalid", INVALID_IDS)
def test_cold_join_missing_identity_cannot_pass_strict_threshold(provider, invalid):
    _, run, state, _ = provider
    state["rosters"].loc[0, "player_id"] = invalid
    with pytest.raises(RuntimeError, match="match rate"):
        run()


def test_allowed_unmatched_rows_are_never_literal_missing_ids(provider):
    name, run, state, _ = provider
    state["rosters"].loc[0, "player_id"] = "None"
    result = run(0.5)
    assert not (result.player_id.notna() & ~valid_player_ids(result.player_id)).any()
    if name == "nflcom":
        assert len(result) == 2 and result.player_id.isna().sum() == 1
    else:
        assert result.player_id.tolist() == ["00-1"]  # matched-only API is preserved


@pytest.mark.parametrize("corruption", ["None", "nan", "missing_column", "actual_null"])
def test_warm_join_rebuilds_invalid_identity_cache(provider, corruption):
    _, run, state, root = provider
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    cached = pd.read_parquet(path)
    if corruption == "missing_column":
        cached = cached.drop(columns="player_id")
    else:
        cached.loc[0, "player_id"] = None if corruption == "actual_null" else corruption
    cached.to_parquet(path)
    actual = run()
    assert state["loads"] == state["roster_loads"] == 2
    pd.testing.assert_frame_equal(actual, expected)
    assert valid_player_ids(actual.player_id).all()


def test_healthy_warm_join_preserves_ids_without_loading_sources(provider):
    _, run, state, root = provider
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    before = path.read_bytes()
    actual = run()
    assert state["loads"] == state["roster_loads"] == 1
    assert path.read_bytes() == before
    pd.testing.assert_frame_equal(actual, expected)


def test_exact_threshold_uses_original_projection_denominator_on_warm_join(provider):
    name, run, state, _ = provider
    state["count"] = 10
    state["rosters"] = _rosters(9)
    first = run(0.9)
    assert len(first) == (10 if name == "nflcom" else 9)
    pd.testing.assert_frame_equal(run(0.9), first)
    assert state["loads"] == 1
    # Both thresholds map to the existing mr90 cache filename. FFToday's
    # matched-only output must not turn the original9/10 coverage into9/9.
    with pytest.raises(RuntimeError, match="match rate"):
        run(0.9001)
    assert state["loads"] == 2


@pytest.mark.parametrize("metadata", ["missing", None, 1, "2", True])
def test_fftoday_legacy_or_invalid_denominator_metadata_must_rebuild(provider, metadata):
    name, run, state, root = provider
    if name != "fftoday":
        pytest.skip("Only FFToday stores matched-only cache rows")
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    cached = pd.read_parquet(path)
    key = "fftoday_join_source_rows_v1"
    if metadata == "missing":
        cached.attrs.pop(key, None)
    else:
        cached.attrs[key] = metadata
    cached.to_parquet(path)
    actual = run()
    assert state["loads"] == 2
    pd.testing.assert_frame_equal(actual, expected)
