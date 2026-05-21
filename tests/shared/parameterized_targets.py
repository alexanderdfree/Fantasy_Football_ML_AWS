"""Parameterized harness for the per-position ``test_targets.py`` suites.

QB/RB/WR/TE/K/DST each used to ship a ``TestCompute{POS}Targets`` class with
near-identical methods. The position-specific bits are:

* the import (``src.{pos}.targets.compute_targets``)
* the position-specific target list (``targets``)
* the row factory (single-row DataFrame builder with sensible defaults)
* the all-NaN row template (which columns to NaN out)
* whether the position aggregates ``fumbles_lost`` from three sources
  (QB/RB/WR/TE) or has no fumbles_lost (K/DST)

This module exposes :class:`PositionTargetSpec` and
:func:`install_parameterized_targets` that build a shared test class in
the caller's globals. Position-specific tests (QB's sanity-warning, RB's
PPR drift, DST's tier-bonus sweeps, K's signed fantasy_points) stay in
the per-position file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class PositionTargetSpec:
    """Per-position config for the shared target test harness.

    Attributes
    ----------
    position
        ``"QB"`` / ``"RB"`` / ``"WR"`` / ``"TE"`` / ``"K"`` / ``"DST"`` —
        used in assertion messages and the generated test-class name.
    compute_targets
        The position's ``compute_targets`` function under test.
    targets
        The list of target columns that must appear in the result.
    make_row
        Factory producing a single-row DataFrame with sensible defaults.
        Accepts arbitrary keyword overrides.
    nan_row_columns
        Columns to populate with ``np.nan`` for the all-NaN test. Defaults
        include the union of raw-stat inputs the position's
        ``compute_targets`` reads. ``fantasy_points`` is intentionally
        kept at 0.0 (not NaN) so the decomposition warning doesn't fire.
    nan_expected_zeros
        After NaN-fill, these columns must equal ``0.0`` in the result.
        Defaults to ``targets``.
    nan_extra_assertions
        Optional extra (col, value) pairs to assert on the all-NaN row.
        Used by DST for tier-default fallbacks (PA=21, YA=350, fp=-1).
    has_fumbles_lost
        If True, also installs ``test_fumbles_lost_sums_all_three_categories``
        and ``test_fumbles_lost_sack_only`` tests. QB/RB/WR/TE only.
    identity_targets
        Tuple of (target_col, override_value) pairs for the identity
        passthrough tests. Each generates ``test_{col}_identity``.
    """

    position: str
    compute_targets: Callable[[pd.DataFrame], pd.DataFrame]
    targets: Iterable[str]
    make_row: Callable[..., pd.DataFrame]
    nan_row_columns: Iterable[str] = ()
    nan_expected_zeros: Iterable[str] = ()
    nan_extra_assertions: Iterable[tuple[str, float]] = ()
    has_fumbles_lost: bool = False
    identity_targets: Iterable[tuple[str, float]] = ()
    big_game_overrides: Mapping[str, float] = field(default_factory=dict)


def _build_class(spec: PositionTargetSpec) -> type:
    """Build the TestCompute{POS}Targets class wired to ``spec``."""
    targets = list(spec.targets)
    nan_zeros = list(spec.nan_expected_zeros) or list(targets)
    nan_extras = list(spec.nan_extra_assertions)
    nan_columns = list(spec.nan_row_columns)

    @pytest.mark.unit
    class TestComputeTargets:
        """Shared compute_targets tests installed by the parameterized harness."""

        def test_all_targets_present(self):
            df = spec.make_row()
            result = spec.compute_targets(df)
            for col in targets:
                assert col in result.columns, f"missing {spec.position} target column: {col}"

        def test_all_nan_stats_treated_as_zero(self):
            row = {col: np.nan for col in nan_columns}
            row["fantasy_points"] = 0.0
            df = pd.DataFrame([row])
            result = spec.compute_targets(df)
            for col in nan_zeros:
                assert result[col].iloc[0] == 0.0, f"{spec.position}: {col} did not fill NaN to 0"
            for col, expected in nan_extras:
                assert pytest.approx(result[col].iloc[0]) == expected, (
                    f"{spec.position}: {col} expected {expected}"
                )

        def test_does_not_mutate_original(self):
            df = spec.make_row()
            original_cols = set(df.columns)
            _ = spec.compute_targets(df)
            assert set(df.columns) == original_cols

    TestComputeTargets.__name__ = f"TestCompute{spec.position}TargetsShared"
    TestComputeTargets.__qualname__ = TestComputeTargets.__name__

    # Identity / passthrough tests for each raw-stat target with an override
    # value. Default-arg trick captures loop values so each method binds its
    # own (col, val).
    for col, val in spec.identity_targets:

        def _test(self, *, _col=col, _val=val):
            df = spec.make_row(**{_col: _val})
            result = spec.compute_targets(df)
            assert pytest.approx(result[_col].iloc[0]) == _val, (
                f"{spec.position}: {_col} did not pass through"
            )

        _test.__name__ = f"test_{col}_identity"
        setattr(TestComputeTargets, _test.__name__, _test)

    # Fumbles_lost = sum of three categories (QB/RB/WR/TE only).
    if spec.has_fumbles_lost:

        def test_fumbles_lost_sums_all_three_categories(self):
            df = spec.make_row(
                sack_fumbles_lost=1,
                rushing_fumbles_lost=1,
                receiving_fumbles_lost=1,
            )
            result = spec.compute_targets(df)
            assert pytest.approx(result["fumbles_lost"].iloc[0]) == 3.0

        def test_fumbles_lost_sack_only(self):
            df = spec.make_row(
                sack_fumbles_lost=2,
                rushing_fumbles_lost=0,
                receiving_fumbles_lost=0,
            )
            result = spec.compute_targets(df)
            assert pytest.approx(result["fumbles_lost"].iloc[0]) == 2.0

        TestComputeTargets.test_fumbles_lost_sums_all_three_categories = (
            test_fumbles_lost_sums_all_three_categories
        )
        TestComputeTargets.test_fumbles_lost_sack_only = test_fumbles_lost_sack_only

    # Big game smoke — every override value pass-through.
    if spec.big_game_overrides:

        def test_big_game_stats_preserved(self):
            df = spec.make_row(**spec.big_game_overrides)
            result = spec.compute_targets(df)
            for col, val in spec.big_game_overrides.items():
                if col in targets:
                    assert pytest.approx(result[col].iloc[0]) == val, (
                        f"{spec.position}: {col} = {val} not preserved in big-game row"
                    )

        TestComputeTargets.test_big_game_stats_preserved = test_big_game_stats_preserved

    return TestComputeTargets


def install_parameterized_targets(globals_dict: dict, spec: PositionTargetSpec) -> None:
    """Install the shared compute_targets test class into ``globals_dict``.

    Called once per position from ``tests/{pos}/test_targets.py``. The
    caller's globals receive ``TestCompute{POS}TargetsShared``; pytest
    collects it under the caller's module. Position-specific tests stay
    in the per-position file.
    """
    cls = _build_class(spec)
    globals_dict[cls.__name__] = cls
