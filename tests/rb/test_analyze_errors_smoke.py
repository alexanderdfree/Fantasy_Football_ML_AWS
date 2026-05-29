"""Import-smoke test for the operator-only CLI ``src/rb/analyze_errors.py``.

The module is ``__main__``-guarded and has no other automated coverage, so a
signature change in any of its imports — ``src.rb.run_pipeline.run``, the
``src.shared.error_analysis`` helpers, or ``POSITION_CONFIG`` — would only
surface when an operator runs it by hand. The module-level
``from src.shared.error_analysis import (...)`` block resolves those names at
import time, so simply importing the module fails the unit shard the moment
one of those symbols is renamed or removed.

This guards the drift class called out in CLAUDE.md: operator CLIs
(``diagnose_outliers.py`` / ``analyze_errors.py`` / ``audit_features.py``)
should have at least an import-smoke test so signature-change drift fails the
unit-test shard rather than the PR-review pass.

Importing must NOT fire ``main()`` or run the RB pipeline — the import-time
guard is exactly the property under test, so we assert the import is cheap
(no pipeline) by checking the public surface is present without calling it.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.unit
def test_module_imports_without_running_pipeline():
    """A bare import resolves every ``from ... import`` at module top.

    If ``src.rb.run_pipeline.run`` or any ``src.shared.error_analysis`` helper
    is renamed/removed, this import raises ImportError and fails the shard.
    Importing the module does not call ``main()`` (it's ``__main__``-guarded),
    so no training round-trip happens here.
    """
    import src.rb.analyze_errors as mod

    assert mod is not None


@pytest.mark.unit
def test_key_functions_present():
    """Reference the module's public callables so a rename fails this test."""
    import src.rb.analyze_errors as mod

    assert callable(mod.build_model_pred_cols)
    assert callable(mod.main)


@pytest.mark.unit
def test_build_model_pred_cols_signature():
    """Lock ``build_model_pred_cols(df, targets)`` — a signature change (the
    drift this bundle guards against) flips this assertion."""
    import src.rb.analyze_errors as mod

    params = list(inspect.signature(mod.build_model_pred_cols).parameters)
    assert params == ["df", "targets"]


@pytest.mark.unit
def test_shared_error_analysis_symbols_imported():
    """The module re-binds the ``src.shared.error_analysis`` helpers at import
    time. Assert each name resolved (caught at import, re-checked here for an
    explicit failure message naming the drifted symbol)."""
    import src.rb.analyze_errors as mod

    for name in (
        "add_stratification_columns",
        "find_top_error_sources",
        "plot_bias_heatmap",
        "plot_error_by_stratum",
        "plot_td_zero_vs_scored",
        "print_stratified_table",
        "print_top_error_sources",
        "run_stratified_analysis",
    ):
        assert hasattr(mod, name), f"analyze_errors lost import of {name!r}"


@pytest.mark.unit
def test_module_constants_present():
    """The CLI's strata config and figure dir are module-level constants the
    pipeline-running ``main()`` reads; keep them addressable."""
    import src.rb.analyze_errors as mod

    assert isinstance(mod.STRATA_COLS, list) and mod.STRATA_COLS
    assert isinstance(mod.FIGURE_DIR, str) and mod.FIGURE_DIR
