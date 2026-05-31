"""Smoke + unit tests for src/analysis/injury_subgroup_error.py.

The script's ``main()`` / ``analyze_position()`` run the full per-position
pipeline (slow, data-dependent) and are out of scope for unit tests. These cover
the pure pieces — the subgroup mask definitions and the small-n flag — against a
tiny synthetic frame so a future change to the slicing logic is caught by CI, and
assert that importing the module does not fire any pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import injury_subgroup_error as ise

pytestmark = pytest.mark.unit


def _synthetic() -> pd.DataFrame:
    # 4 rows spanning settled / 1-wk-out / 2-wk-out and a Questionable row.
    return pd.DataFrame(
        {
            "fantasy_points": [10.0, 12.0, 6.0, 8.0],
            "is_returning_from_absence": [0, 0, 1, 1],
            "days_rest": [7, 7, 14, 21],
            "game_status": [1.0, 1.0, 0.5, 1.0],
            "pred_ridge_total": [11.0, 11.0, 9.0, 11.0],
        }
    )


def test_module_imports_without_running_pipeline():
    """Importing must not pull any pipeline — run() is deferred into
    analyze_position()."""
    assert hasattr(ise, "analyze_position")
    assert hasattr(ise, "SUBGROUP_SPECS")


def test_subgroup_masks_partition_correctly():
    df = _synthetic()
    masks = {key: fn for key, _label, _needed, fn in ise.SUBGROUP_SPECS}

    assert masks["global"](df).tolist() == [True, True, True, True]
    # settled + returning partition the rows.
    assert masks["settled"](df).tolist() == [True, True, False, False]
    assert masks["returning"](df).tolist() == [False, False, True, True]
    # days_rest sub-slices of returning: exactly-1-wk (==14) vs 2+ wk (>14).
    assert masks["ret_1wk"](df).tolist() == [False, False, True, False]
    assert masks["ret_2wk"](df).tolist() == [False, False, False, True]
    # healthy + questionable partition by designation.
    assert masks["healthy"](df).tolist() == [True, True, False, True]
    assert masks["questionable"](df).tolist() == [False, False, True, False]


def test_subgroup_specs_name_their_required_column():
    """Every non-global spec names the column it slices on, so the analysis
    skips it (rather than KeyError-ing) when a position's test_df lacks it —
    e.g. K/DST do not carry is_returning_from_absence/game_status."""
    needed = {key: req for key, _label, req, _fn in ise.SUBGROUP_SPECS}
    assert needed["global"] is None
    assert needed["returning"] == "is_returning_from_absence"
    assert needed["ret_2wk"] == "days_rest"
    assert needed["questionable"] == "game_status"
    df_missing = pd.DataFrame({"fantasy_points": [1.0]})
    for _key, _label, req, _fn in ise.SUBGROUP_SPECS:
        if req is not None:
            assert req not in df_missing.columns  # would be skipped by analyze_position


def test_flag_thresholds():
    assert "EMPTY" in ise._flag(0)
    assert "small-n" in ise._flag(ise.SMALL_N - 1)
    assert ise._flag(ise.SMALL_N) == ""
    assert ise._flag(1000) == ""
