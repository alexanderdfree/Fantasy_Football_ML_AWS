"""Shared CLI dispatcher for QB/RB/WR/TE ``run_pipeline.py`` modules.

The ``if __name__ == "__main__"`` block of QB/RB/WR/TE is otherwise identical
boilerplate: build an ``argparse`` parser with ``--tiny`` (and ``--cv`` for the
three that support it), swap in ``build_tiny_config(pos)`` when ``--tiny`` is
passed, then dispatch to ``run`` or ``run_cv``. Centralising it here keeps each
position's ``run_pipeline.py`` to just the imports + ``CONFIG`` + ``run``
(and optionally ``run_cv``).

Why this isn't a closure over ``run_pipeline``/``run_cv_pipeline``
-----------------------------------------------------------------
``tests/{pos}/test_run_pipeline_main.py`` monkeypatches the shared functions
via ``src.shared.pipeline.run_pipeline``; ``runpy.run_path`` then re-imports the
position module, which binds its local ``run_pipeline`` name to the patched
stub. That contract requires each per-position file to keep its own
``from src.shared.pipeline import run_pipeline`` line and to define ``run`` /
``run_cv`` locally (so they close over the position's module-level name, not a
factory-captured reference). The factory only consolidates the CLI scaffold —
it never calls ``src.shared.pipeline`` itself.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any


def cli_main(
    *,
    position_name: str,
    default_config: dict,
    run_fn: Callable[..., Any],
    run_cv_fn: Callable[..., Any] | None = None,
) -> None:
    """Parse ``--tiny``/``--cv`` and dispatch to ``run_fn`` or ``run_cv_fn``.

    Parameters
    ----------
    position_name:
        Position label used to build a tiny config (e.g. ``"QB"``).
    default_config:
        The position's ``CONFIG`` dict; used unless ``--tiny`` is passed.
    run_fn:
        The position's ``run(...)`` wrapper. Must accept ``config=`` kwarg.
    run_cv_fn:
        Optional CV wrapper. When supplied, the ``--cv`` flag is added to the
        parser. TE has no CV path so it leaves this ``None``.
    """
    parser = argparse.ArgumentParser()
    if run_cv_fn is not None:
        parser.add_argument("--cv", action="store_true", help="Use expanding-window CV")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use shrunk smoke-test config (from tests/_pipeline_e2e_utils)",
    )
    args = parser.parse_args()
    if args.tiny:
        from tests._pipeline_e2e_utils import build_tiny_config

        config = build_tiny_config(position_name)
    else:
        config = default_config
    if run_cv_fn is not None and args.cv:
        run_cv_fn(config=config)
    else:
        run_fn(config=config)
