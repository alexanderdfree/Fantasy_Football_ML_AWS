"""Shared CLI dispatcher for QB/RB/WR/TE ``run_pipeline.py`` modules.

The ``if __name__ == "__main__"`` block of QB/RB/WR/TE is otherwise identical
boilerplate: build an ``argparse`` parser with ``--tiny`` (and ``--cv`` for the
three that support it), swap in ``_build_tiny_config(pos)`` when ``--tiny`` is
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
import importlib
import os
from collections.abc import Callable
from typing import Any

# Tiny-config overrides shared with ``tests/_pipeline_e2e_utils``. Duplicated
# here so the factory's ``--tiny`` path doesn't take a ``src/`` → ``tests/``
# import (the only one in the repo until that import was dropped). The test
# helper remains the canonical source for E2E fixtures; this copy only powers
# the operator-facing ``--tiny`` CLI flag, which never reaches the test path.
_TINY_OVERRIDES: dict[str, Any] = {
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 64,
    "nn_patience": 1,
    "nn_log_every": 100,
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "ridge_pca_components": None,
    "train_attention_nn": False,
    "train_lightgbm": False,
}

_ALL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")


def _build_tiny_config(position: str) -> dict:
    """Return a shrunk run_pipeline config for the given position.

    Mirrors ``tests._pipeline_e2e_utils.build_tiny_config`` — kept inline so the
    ``--tiny`` CLI path does not need to import from ``tests/``. Both copies
    delegate to ``build_pipeline_config`` so they exercise the same production
    callables / POSITION_CONFIG-derived fields.
    """
    from src.shared.position_pipeline import build_pipeline_config

    position = position.upper()
    if position not in _ALL_POSITIONS:
        raise ValueError(f"Unknown position {position!r}")

    config_mod = importlib.import_module(f"src.{position.lower()}.config")
    cfg = build_pipeline_config(position, config_mod.POSITION_CONFIG)
    cfg.update(_TINY_OVERRIDES)

    if "ridge_alpha_grids" in cfg:
        cfg["ridge_alpha_grids"] = {t: [1.0, 10.0] for t in cfg.get("targets", [])}

    cfg_tiny = getattr(config_mod, "CONFIG_TINY", None)
    if cfg_tiny is not None:
        cfg.update(cfg_tiny)

    if position in ("K", "DST"):
        cfg.setdefault("compute_adjustment_fn", None)

    return cfg


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
        help="Use shrunk smoke-test config",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help=(
            "Compute device for the attention NN. Omitted: honour $FF_DEVICE if "
            "set, else 'auto'. 'auto' uses CUDA when available and CPU otherwise "
            "(the historical behaviour — Linux/macOS/CI unchanged). 'cpu' forces "
            "the CPU path even on a CUDA-visible box (e.g. macOS, or a Windows "
            "dev box with a flaky CUDA build); 'cuda' requires a GPU and errors "
            "if none is visible. An explicit value is exported as FF_DEVICE for "
            "the run, reaching the device selectors via src.shared.utils."
        ),
    )
    args = parser.parse_args()
    # Publish an explicit --device before dispatch so the device selectors deep
    # in the pipeline (_nn_device / _gpu_resident_device, via
    # src.shared.utils.cuda_enabled) observe it. Left unset, FF_DEVICE from the
    # surrounding shell (or its "auto" default) is respected — the flag overrides
    # the environment only when actually passed.
    if args.device is not None:
        os.environ["FF_DEVICE"] = args.device
    config = _build_tiny_config(position_name) if args.tiny else default_config
    if run_cv_fn is not None and args.cv:
        run_cv_fn(config=config)
    else:
        run_fn(config=config)
