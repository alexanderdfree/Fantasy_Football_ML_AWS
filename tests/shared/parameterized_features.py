"""Parameterized harness for the per-position ``test_features.py`` suites.

QB/RB/WR/TE each used to ship ~190 LOC of near-identical
``TestCompute{POS}Features`` / ``TestFill{POS}Nans`` boilerplate. The only
genuine differences between the four positions are:

* the import (``src.{pos}.features._compute_features`` / ``fill_nans``)
* the position-specific feature-name list (``FEATURE_COLS``)
* the player-games factory (one position-specific helper builds a multi-week
  single-player DataFrame; each position carries its own column schema)
* a small set of "zero-input → no inf" tests whose inputs depend on the
  factory's keyword args

This module exposes :class:`PositionFeatureSpec` and a single helper —
:func:`install_parameterized_features` — that builds the two shared test
classes inside the caller's globals. Per-position test files become a thin
spec + position-only tests. K and DST stay in their own ``test_features.py``
files (custom kick-level / team-level feature logic).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class PositionFeatureSpec:
    """Per-position config for the shared feature test harness.

    Attributes
    ----------
    position
        ``"QB"`` / ``"RB"`` / ``"WR"`` / ``"TE"`` — used in assertion messages
        and (via :func:`install_parameterized_features`) the generated
        test-class names.
    compute_features, fill_nans
        Position-level functions under test. Imported from
        ``src.{pos}.features``.
    feature_cols
        The complete list of expected feature columns produced by
        ``compute_features`` for this position.
    make_player_games
        Factory that returns a multi-week single-player DataFrame. Accepts
        the same kwargs as the position's existing factory (``player_id``,
        ``season``, ``n_weeks``, plus raw-stat overrides).
    zero_input_overrides
        Kwargs the harness passes to ``make_player_games`` for the
        ``zero_*_no_division_error`` tests. Each entry's key is a label used
        in the generated test method name; the value is the kwargs dict.
    zero_input_features
        For each zero-input case, the list of feature columns that must not
        be inf. ``None`` means "every column in ``feature_cols``".
    first_week_check_features
        Subset of features the ``first_week_features_are_zero`` test must
        check. The original suites differed — QB checks 3, TE checks all,
        WR checks 2 — so each position keeps its own list. Empty tuple
        means "check every column in ``feature_cols``".
    seasons_check_feature
        Feature column whose week-1 value the ``multiple_seasons_independent``
        test asserts is 0 / NaN.
    seasons_factory_kwargs
        Tuple of kwargs (``kwargs_2022``, ``kwargs_2023``) passed when the
        multi-season test builds its two single-season DataFrames.
    """

    position: str
    compute_features: Callable[[pd.DataFrame], pd.DataFrame | None]
    fill_nans: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
    feature_cols: list[str]
    make_player_games: Callable[..., pd.DataFrame]
    zero_input_overrides: Mapping[str, dict] = field(default_factory=dict)
    zero_input_features: Mapping[str, list[str] | None] = field(default_factory=dict)
    first_week_check_features: tuple[str, ...] = ()
    seasons_check_feature: str = ""
    seasons_factory_kwargs: tuple[dict, dict] = field(default_factory=lambda: ({}, {}))


def _assert_no_inf(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf"


def _build_compute_class(spec: PositionFeatureSpec) -> type:
    """Build the TestCompute{POS}Features class wired to ``spec``."""

    @pytest.mark.unit
    class TestComputeFeatures:
        """Shared compute-features tests installed by the parameterized harness."""

        def test_all_features_created(self):
            df = spec.make_player_games()
            spec.compute_features(df)
            for col in spec.feature_cols:
                assert col in df.columns, f"Missing feature: {col}"

        def test_first_week_features_are_zero(self):
            """Week 1 has no prior data (shift=1), so features should be 0 (filled)."""
            df = spec.make_player_games(n_weeks=4)
            spec.compute_features(df)
            cols = spec.first_week_check_features or tuple(spec.feature_cols)
            week1 = df[df["week"] == 1]
            for col in cols:
                val = week1[col].iloc[0]
                assert val == 0.0 or np.isnan(val), f"{col} = {val} for week 1"

        def test_multiple_seasons_independent(self):
            """Features should not leak across seasons."""
            kw_2022, kw_2023 = spec.seasons_factory_kwargs
            s1 = spec.make_player_games(season=2022, n_weeks=3, **kw_2022)
            s2 = spec.make_player_games(season=2023, n_weeks=3, **kw_2023)
            df = pd.concat([s1, s2], ignore_index=True)
            spec.compute_features(df)
            w1_2023 = df[(df["season"] == 2023) & (df["week"] == 1)]
            val = w1_2023[spec.seasons_check_feature].iloc[0]
            assert val == 0.0 or np.isnan(val)

    TestComputeFeatures.__name__ = f"TestCompute{spec.position}Features"
    TestComputeFeatures.__qualname__ = TestComputeFeatures.__name__

    # Attach one zero-input test per entry in zero_input_overrides. The
    # default-argument bindings capture the loop-iteration values so each
    # method sees its own (overrides, check_cols) instead of the last
    # iteration's values from a shared closure cell.
    for label, overrides in spec.zero_input_overrides.items():

        def _test(self, *, _overrides=overrides, _check_cols=spec.zero_input_features.get(label)):
            df = spec.make_player_games(n_weeks=4, **_overrides)
            spec.compute_features(df)
            cols = _check_cols if _check_cols is not None else spec.feature_cols
            _assert_no_inf(df, cols)

        _test.__name__ = f"test_zero_{label}_no_division_error"
        setattr(TestComputeFeatures, _test.__name__, _test)

    return TestComputeFeatures


def _build_fillnans_class(spec: PositionFeatureSpec) -> type:
    """Build the TestFill{POS}Nans class.

    All five shared ``fill_nans`` tests look identical across positions —
    only the imported ``fill_nans`` differs. The class closes over the
    spec's function so the assertions hit the right code path.
    """
    fill_nans = spec.fill_nans

    @pytest.mark.unit
    class TestFillNans:
        """Shared NaN-fill tests installed by the parameterized harness."""

        def test_fills_nan_with_train_mean(self, make_splits):
            train, val, test = make_splits([1.0, 2.0, 3.0], [np.nan], [np.nan])
            train, val, test = fill_nans(train, val, test, ["feat1"])
            assert pytest.approx(val["feat1"].iloc[0]) == 2.0
            assert pytest.approx(test["feat1"].iloc[0]) == 2.0

        def test_replaces_inf_with_train_mean(self, make_splits):
            train, val, test = make_splits([1.0, 3.0], [np.inf], [-np.inf])
            train, val, test = fill_nans(train, val, test, ["feat1"])
            assert pytest.approx(val["feat1"].iloc[0]) == 2.0
            assert pytest.approx(test["feat1"].iloc[0]) == 2.0

        def test_train_inf_replaced_before_mean(self, make_splits):
            """Inf in training set should be replaced with NaN before computing mean."""
            train, val, test = make_splits([1.0, np.inf, 3.0], [np.nan], [np.nan])
            train, val, test = fill_nans(train, val, test, ["feat1"])
            assert pytest.approx(val["feat1"].iloc[0]) == 2.0

        def test_no_nans_unchanged(self, make_splits):
            train, val, test = make_splits([1.0, 2.0], [3.0], [4.0])
            train, val, test = fill_nans(train, val, test, ["feat1"])
            assert pytest.approx(val["feat1"].iloc[0]) == 3.0
            assert pytest.approx(test["feat1"].iloc[0]) == 4.0

        def test_multiple_columns(self):
            train = pd.DataFrame({"f1": [1.0, 3.0], "f2": [10.0, 20.0]})
            val = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
            test = pd.DataFrame({"f1": [5.0], "f2": [np.nan]})
            train, val, test = fill_nans(train, val, test, ["f1", "f2"])
            assert pytest.approx(val["f1"].iloc[0]) == 2.0
            assert pytest.approx(val["f2"].iloc[0]) == 15.0
            assert pytest.approx(test["f1"].iloc[0]) == 5.0
            assert pytest.approx(test["f2"].iloc[0]) == 15.0

    TestFillNans.__name__ = f"TestFill{spec.position}Nans"
    TestFillNans.__qualname__ = TestFillNans.__name__
    return TestFillNans


def install_parameterized_features(globals_dict: dict, spec: PositionFeatureSpec) -> None:
    """Install the shared compute-features + fill-nans test classes into ``globals_dict``.

    Called once per position from ``tests/{pos}/test_features.py``. The
    caller's globals receive ``TestCompute{POS}Features`` and
    ``TestFill{POS}Nans``; pytest collects them under the caller's module.
    Position-specific tests stay in the per-position file.
    """
    compute_cls = _build_compute_class(spec)
    globals_dict[compute_cls.__name__] = compute_cls

    fill_cls = _build_fillnans_class(spec)
    globals_dict[fill_cls.__name__] = fill_cls
