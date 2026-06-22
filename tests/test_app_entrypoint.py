"""Local dev entrypoint contract for ``python -m src.serving.app``.

Regression guard for the double-module-instance bug: when app.py executes as
``__main__`` (via ``-m`` or a direct path), routes.py's
``from src.serving.app import app`` re-executes app.py as a *second* module
instance — sys.modules holds only ``__main__`` at that point — so every
``@app.route`` handler registers on the second instance's Flask app. Before
the fix, the ``__main__`` branch ran its OWN route-less ``app`` and every
endpoint 404'd. The branch must serve the canonical ``src.serving.app``
instance (the one routes register on and ``app_pkg`` shared state lives on).

Production (gunicorn ``src.serving.app:app``) never runs the ``__main__``
branch and is unaffected; this is a dev-entrypoint-only contract.
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


def test_main_entrypoint_serves_canonical_app(monkeypatch):
    """The ``__main__`` branch must call ``run()`` on the canonical app.

    ``runpy.run_module(..., run_name="__main__")`` mirrors ``python -m``: it
    executes app.py's code in a fresh ``__main__`` namespace (creating the
    throwaway duplicate Flask app) without touching sys.modules. ``Flask.run``
    is patched so no server binds; we only capture which instance the branch
    would serve.
    """
    import flask

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

    import src.serving.app as app_module

    assert captured["app"] is app_module.app, (
        "python -m src.serving.app must serve the canonical src.serving.app "
        "instance, not the __main__ duplicate (which has no routes and 404s "
        "every endpoint)"
    )
    # Sanity: the served instance carries the real route table, not just the
    # built-in /static rule the duplicate had.
    rule_count = len(list(captured["app"].url_map.iter_rules()))
    assert rule_count > 10, f"canonical app should expose the full route table, got {rule_count}"
