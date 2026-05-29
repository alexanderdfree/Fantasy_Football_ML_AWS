"""Unit + smoke tests for src/analysis/analysis_nflreadpy_diff.py.

The script's ``main()`` pulls nflverse data over the network from both
packages (slow, data-dependent) — out of scope for unit tests. These cover
the pure comparison core (:func:`compare_frames` and the verdict taxonomy)
against synthetic frames, so a future change to the diff logic or thresholds
is caught by CI. Also an import smoke so a signature break fails the unit
shard rather than only surfacing at PR-review time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import analysis_nflreadpy_diff as mod

pytestmark = pytest.mark.unit


def _f(**cols: object) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_module_imports_cleanly() -> None:
    """No network/file IO at import time; public API present."""
    assert hasattr(mod, "compare_frames")
    assert hasattr(mod, "SourceDiff")
    assert hasattr(mod, "main")
    # The weekly rename map must mirror src/data/loader.py — a regression here
    # would mean the harness compares mis-aligned columns.
    assert mod.WEEKLY_RENAME == {
        "team": "recent_team",
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
    }


def test_identical_frames_verdict_identical() -> None:
    a = _f(id=[1, 2, 3], x=[1.0, 2.0, 3.0])
    b = _f(id=[1, 2, 3], x=[1.0, 2.0, 3.0])
    d = mod.compare_frames(a, b, ["id"], ["x"])
    assert d.verdict() == "IDENTICAL"
    assert d.n_keys_shared == 3
    assert d.keys_a_only == 0 and d.keys_b_only == 0
    assert d.value_results["x"]["n_mismatch"] == 0


def test_rename_only() -> None:
    a = _f(id=[1, 2], recent_team=["KC", "SF"])
    b = _f(id=[1, 2], team=["KC", "SF"])
    d = mod.compare_frames(a, b, ["id"], ["recent_team"], rename_b_to_a={"team": "recent_team"})
    assert d.applied_renames == {"team": "recent_team"}
    assert d.col_a_only == [] and d.col_b_only == []
    assert d.value_results["recent_team"]["n_mismatch"] == 0
    assert d.verdict() == "RENAME-ONLY"


def test_value_delta_numeric_counts_and_examples() -> None:
    a = _f(id=[1, 2, 3], x=[1.0, 2.0, 3.0])
    b = _f(id=[1, 2, 3], x=[1.0, 2.0, 99.0])
    d = mod.compare_frames(a, b, ["id"], ["x"])
    assert d.verdict() == "VALUE-DELTA"
    r = d.value_results["x"]
    assert r["kind"] == "numeric"
    assert r["n_mismatch"] == 1
    assert r["max_abs_delta"] == 96.0
    assert r["examples"][0]["key"] == "3"


def test_numeric_within_tolerance_is_match() -> None:
    a = _f(id=[1], x=[1.0])
    b = _f(id=[1], x=[1.0000001])
    d = mod.compare_frames(a, b, ["id"], ["x"], atol=1e-6)
    assert d.value_results["x"]["n_mismatch"] == 0
    assert d.verdict() == "IDENTICAL"


def test_row_delta_when_keys_differ() -> None:
    a = _f(id=[1, 2, 3], x=[1.0, 2.0, 3.0])
    b = _f(id=[1, 2], x=[1.0, 2.0])
    d = mod.compare_frames(a, b, ["id"], ["x"])
    assert d.verdict() == "ROW-DELTA"
    assert d.keys_a_only == 1
    assert d.keys_b_only == 0


def test_schema_break_when_consumed_col_missing() -> None:
    a = _f(id=[1, 2], logo=["u1", "u2"])
    b = _f(id=[1, 2], other=["x", "y"])
    d = mod.compare_frames(a, b, ["id"], ["logo"])
    assert d.verdict() == "SCHEMA-BREAK"
    assert any("logo" in m for m in d.value_cols_missing)


def test_dtype_only_when_values_match_but_types_differ() -> None:
    a = _f(id=[1, 2], x=pd.Series([1, 2], dtype="int64"))
    b = _f(id=[1, 2], x=pd.Series([1.0, 2.0], dtype="float64"))
    d = mod.compare_frames(a, b, ["id"], ["x"])
    assert "x" in d.dtype_diffs
    assert d.value_results["x"]["n_mismatch"] == 0
    assert d.verdict() == "DTYPE-ONLY"


def test_string_mismatch_and_nan_handling() -> None:
    # id2: B≠C mismatch; id3: NaN==NaN match; id4: D≠NaN mismatch.
    a = _f(id=[1, 2, 3, 4], s=["A", "B", None, "D"])
    b = _f(id=[1, 2, 3, 4], s=["A", "C", None, None])
    d = mod.compare_frames(a, b, ["id"], ["s"])
    r = d.value_results["s"]
    assert r["kind"] == "string"
    assert r["n_mismatch"] == 2


def test_composite_key_aligns_int_and_float_representations() -> None:
    """season as int (A) vs float (B, polars→pandas downcast) must still join."""
    a = _f(player_id=["p1", "p2"], season=[2012, 2013], x=[10.0, 20.0])
    b = _f(player_id=["p1", "p2"], season=[2012.0, 2013.0], x=[10.0, 20.0])
    d = mod.compare_frames(a, b, ["player_id", "season"], ["x"])
    assert d.n_keys_shared == 2
    assert d.keys_a_only == 0 and d.keys_b_only == 0
    assert d.value_results["x"]["n_mismatch"] == 0


def test_duplicate_keys_noted_and_first_kept() -> None:
    a = _f(id=[1, 1, 2], x=[1.0, 9.0, 2.0])
    b = _f(id=[1, 2], x=[1.0, 2.0])
    d = mod.compare_frames(a, b, ["id"], ["x"])
    assert d.a_dupe_keys == 1
    assert any("duplicate keys" in n for n in d.notes)
    # first row for id=1 (x=1.0) kept → matches b → no value mismatch
    assert d.value_results["x"]["n_mismatch"] == 0


def test_column_override_uses_full_schema_for_col_diff() -> None:
    """pbp passes column-subset frames + full-schema overrides: the column-set
    diff must reflect the overrides, while the value diff still runs on the
    actual (subset) frames.
    """
    a = _f(game_id=["g1"], play_id=[1], yardline_100=[50.0])
    b = _f(game_id=["g1"], play_id=[1], yardline_100=[50.0])
    d = mod.compare_frames(
        a,
        b,
        ["game_id", "play_id"],
        ["yardline_100"],
        a_cols_override=["game_id", "play_id", "yardline_100", "dakota"],
        b_cols_override=["game_id", "play_id", "yardline_100", "new_metric"],
    )
    # column diff comes from the overrides, not the 3-col subset frames
    assert d.col_a_only == ["dakota"]
    assert d.col_b_only == ["new_metric"]
    # value diff still ran on the subset frames
    assert d.value_results["yardline_100"]["n_mismatch"] == 0
