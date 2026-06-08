"""Import + pure-helper smoke test for the operator CLI
``src/scripts/analyze_gpu_profile.py``.

The module is ``__main__``-guarded and never imported by the rest of the
pipeline, so signature drift in anything it pulls in (or a rename of one of its
own helpers) only surfaces at PR-review time — ``pytest -m unit`` never imports
it. Mirrors ``tests/qb/test_diagnose_outliers_smoke.py`` for the same
operator-CLI class (audit #350 F24). The checks are cheap: an import, a
public-surface assertion, and the two pure stat helpers exercised on synthetic
input. No S3, no GPU, no data.
"""

import importlib

import pytest


@pytest.mark.unit
def test_analyze_gpu_profile_imports_cleanly():
    """Importing runs the module's top-level imports, so a renamed/removed
    symbol fails here rather than at PR review."""
    mod = importlib.import_module("src.scripts.analyze_gpu_profile")
    assert mod is not None


@pytest.mark.unit
def test_analyze_gpu_profile_public_callables_exist():
    """Pin the CLI's helper surface so a rename/deletion fails the unit shard."""
    mod = importlib.import_module("src.scripts.analyze_gpu_profile")
    for name in (
        "main",
        "_read_profile_csv",
        "_percentile",
        "_stats",
        "_print_markdown_table",
        "_extract_csv_from_tarball",
        "_download_from_s3",
    ):
        fn = getattr(mod, name, None)
        assert callable(fn), f"analyze_gpu_profile.{name} missing or not callable"


@pytest.mark.unit
def test_analyze_gpu_profile_percentile_and_stats():
    """Exercise the pure stat helpers — a refactor that breaks the percentile
    interpolation or the ``_stats`` key set fails the unit shard, not review."""
    mod = importlib.import_module("src.scripts.analyze_gpu_profile")

    # _percentile: linear interpolation; empty -> 0.0; single value -> itself.
    assert mod._percentile([], 50) == 0.0
    assert mod._percentile([10.0], 95) == 10.0
    assert mod._percentile([0.0, 10.0], 50) == pytest.approx(5.0)

    # _stats over a tiny synthetic profile.
    parsed = {"util_gpu": [0.0, 50.0, 100.0], "mem_used_gb": [1.0, 2.0, 3.0]}
    s = mod._stats(parsed)
    assert s["n_samples"] == 3
    assert s["util_avg"] == pytest.approx(50.0)
    assert s["util_peak"] == 100.0
    assert s["mem_peak_gb"] == pytest.approx(3.0)
