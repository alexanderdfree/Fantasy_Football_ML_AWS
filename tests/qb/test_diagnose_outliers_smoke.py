"""Import smoke test for the operator CLI ``src/qb/diagnose_outliers.py``.

This module is ``__main__``-guarded and never imported by the rest of the
pipeline, so a signature drift in any of the shared helpers it pulls in
(``src.shared.pipeline._train_nn`` / ``_prepare_position_data`` /
``_tune_ridge_alphas_cv`` / ``_read_split``, ``src.shared.models.RidgeMultiTarget``,
``src.shared.feature_build.scale_and_clip``, ``src.qb.run_pipeline.CONFIG``)
would otherwise only surface at PR-review time — ``pytest -m unit`` never
imports the file. The 2026-05-21 ``_train_nn`` signature-change conflict
(W.SHARED-A dropped the ``position`` param while a CLI call site kept the old
signature) is the canonical instance this guards against.

These checks are deliberately cheap: importing the module exercises its
top-level imports (catching any renamed/removed shared symbol), and the
callable-exists assertions catch a helper being deleted or renamed within the
module itself. No data, no training, no GPU.
"""

import importlib
import inspect

import pytest


@pytest.mark.unit
def test_diagnose_outliers_imports_cleanly():
    """The module must import without error.

    Importing runs its top-level ``from src.shared.pipeline import (...)`` etc.,
    so a renamed/removed shared symbol fails here rather than at PR review.
    """
    mod = importlib.import_module("src.qb.diagnose_outliers")
    assert mod is not None


@pytest.mark.unit
def test_diagnose_outliers_public_callables_exist():
    """Key functions the CLI defines must exist and be callable.

    Pins the public surface so a rename/deletion (or a refactor that collapses
    one of these into an inline) fails the unit shard.
    """
    mod = importlib.import_module("src.qb.diagnose_outliers")
    for name in (
        "main",
        "_train_models",
        "ridge_attribution",
        "nn_integrated_gradients",
        "_pick_row",
        "_receiving_contamination",
        "_player_history",
        "_unfiltered_history",
        "_load_splits",
    ):
        fn = getattr(mod, name, None)
        assert callable(fn), f"diagnose_outliers.{name} missing or not callable"


@pytest.mark.unit
def test_diagnose_outliers_target_dataclass_shape():
    """The ``Target`` dataclass fields the CLI's ``TARGETS`` list relies on
    must stay (name/season/week)."""
    mod = importlib.import_module("src.qb.diagnose_outliers")
    fields = inspect.signature(mod.Target).parameters
    assert {"name", "season", "week"} <= set(fields)
    # TARGETS is the operator-curated list of outlier players the CLI explains.
    assert len(mod.TARGETS) > 0
    assert all(isinstance(t, mod.Target) for t in mod.TARGETS)


@pytest.mark.unit
def test_diagnose_outliers_attribution_signatures():
    """The two attribution entry points keep the arg names the CLI's ``main``
    passes positionally — a silent arg drop/reorder here is exactly the
    review-only failure mode F1 flags."""
    mod = importlib.import_module("src.qb.diagnose_outliers")

    ridge_params = list(inspect.signature(mod.ridge_attribution).parameters)
    assert ridge_params == ["ridge_model", "feature_cols", "x_raw", "target_name"]

    nn_params = list(inspect.signature(mod.nn_integrated_gradients).parameters)
    assert nn_params == [
        "nn_model",
        "nn_scaler",
        "feature_cols",
        "x_raw",
        "target_name",
        "device",
    ]
