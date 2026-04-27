"""Live-schema integration test for the hvpkod/NFL-Data upstream.

This makes a real HTTP request to ``raw.githubusercontent.com`` to confirm
the upstream column schema still matches what ``src.data.nflcom_loader``
expects. It is **opt-in** via ``RUN_INTEGRATION=1`` (or pytest -m integration
combined with that env var) so it never runs in default CI — network in CI
is fragile and operator-driven analysis is not in the deployment path.

When upstream changes a column name (e.g. ``PassingTD`` -> ``PassingTDs``),
unit tests still pass against our hand-built fixture but the loader crashes
at runtime. This test catches the drift early.

Run:
    RUN_INTEGRATION=1 pytest tests/integration/test_nflcom_live_schema.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data.nflcom_loader import (
    NFLCOM_BASE,
    NFLCOM_COLUMN_MAP,
    NFLCOM_POSITIONS,
    _read_one_projection,
    load_nflcom_with_gsis_id,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Set RUN_INTEGRATION=1 to enable network-bound integration tests",
    ),
]


# A (year, week) combination that should be stable for the foreseeable future:
# 2024 week 1 is in the past, fully completed, all positions present.
_STABLE_YEAR = 2024
_STABLE_WEEK = 1


@pytest.mark.parametrize("position", list(NFLCOM_POSITIONS))
def test_upstream_csv_has_expected_columns(position):
    """Pull one CSV per position from raw.githubusercontent.com and assert
    every column we map in NFLCOM_COLUMN_MAP is still present in the upstream
    schema, plus the universal columns we always read."""
    df = _read_one_projection(_STABLE_YEAR, _STABLE_WEEK, position)
    assert df is not None, f"Upstream returned no rows for {position} {_STABLE_YEAR}W{_STABLE_WEEK}"
    assert len(df) > 0

    # Universal columns (read for every position).
    universal = {
        "PlayerName",
        "PlayerId",
        "Pos",
        "Team",
        "PlayerOpponent",
        "PlayerWeekProjectedPts",
    }
    missing_universal = universal - set(df.columns)
    assert not missing_universal, (
        f"Upstream {position} CSV is missing universal columns: {missing_universal}. "
        f"Schema may have drifted; got columns {sorted(df.columns)}"
    )

    # Per-position mapped columns.
    expected = set(NFLCOM_COLUMN_MAP.get(position, {}).keys())
    missing = expected - set(df.columns)
    assert not missing, (
        f"Upstream {position} CSV is missing mapped columns: {missing}. "
        f"Got columns {sorted(df.columns)}"
    )


def test_base_url_resolves():
    """Sanity check: the NFLCOM_BASE prefix + a known path resolves."""
    url = f"{NFLCOM_BASE}/{_STABLE_YEAR}/{_STABLE_WEEK}/projected/QB_projected.csv"
    df = pd.read_csv(url)
    assert len(df) > 0


def test_match_rate_above_threshold_2024(tmp_path):
    """Full end-to-end smoke: fetch + join 2024 NFL.com against nflverse rosters.
    Match rate must clear the 90% threshold; if it doesn't, the loader raises
    and the test fails (which is what we want)."""
    df = load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        min_match_rate=0.90,
    )
    matched = df["player_id"].notna().sum()
    total = len(df)
    rate = matched / total
    print(f"\n  2024 match rate: {matched}/{total} = {rate:.1%}")
    assert rate >= 0.90, f"Match rate {rate:.1%} below 90% threshold"
