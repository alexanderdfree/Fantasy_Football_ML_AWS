"""The dev entrypoint must compose the same routes and state as the imported app.

Previously routes registered on a different module's app and ``python -m``
served only 404s. The blueprint factory now supports distinct complete Flask
instances; assert actual route behavior and explicit state ownership.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.unit


def test_main_entrypoint_serves_complete_composed_app(monkeypatch):
    """The ``__main__`` branch must serve the complete composed application.

    ``runpy.run_module(..., run_name="__main__")`` mirrors ``python -m``: it
    executes app.py's code in a fresh ``__main__`` namespace (creating the
    throwaway duplicate Flask app) without touching sys.modules. ``Flask.run``
    is patched so no server binds; we only capture which instance the branch
    would serve.
    """
    import flask

    import src.serving.app as app_module

    # This exercises a cold application entrypoint, independently of a prior
    # serving test's deliberately degraded default-owner cache.
    monkeypatch.setattr(app_module._default_state, "cache", {})

    captured = {}

    def fake_run(self, *args, **kwargs):
        captured["app"] = self

    monkeypatch.setattr(flask.Flask, "run", fake_run)

    # ``run_module`` re-executes app.py as ``__main__`` while the canonical
    # ``src.serving.app`` is already imported (by sibling serving tests, or by
    # the __main__ branch's own ``from src.serving.app import app``). runpy's
    # ``_get_module_details`` then emits a generic "<mod> found in sys.modules
    # ... prior to execution ... may result in unpredictable behaviour"
    # RuntimeWarning. That *is* the double-module scenario this test asserts we
    # handle correctly, so the warning is expected here — suppress just that
    # message to keep the CI warnings summary clean (any other RuntimeWarning
    # still surfaces).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*found in sys\.modules.*prior to execution.*",
            category=RuntimeWarning,
        )
        runpy.run_module("src.serving.app", run_name="__main__")

    served = captured["app"]
    rules = {(rule.rule, tuple(sorted(rule.methods))) for rule in served.url_map.iter_rules()}
    expected = {
        (rule.rule, tuple(sorted(rule.methods))) for rule in app_module.app.url_map.iter_rules()
    }
    assert rules == expected
    assert len(rules) > 10
    assert served.extensions["ffp_state"] is app_module._default_state
    with served.test_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.is_json
