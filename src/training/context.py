"""Per-run locations and effect ownership, independent of process cwd."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4


def execute_effect(effect: Callable[[], Any]) -> Any:
    return effect()


@dataclass(frozen=True)
class ExecutionPolicy:
    """Recorded effective overrides; resolving a context never changes them."""

    overrides: tuple[tuple[str, str], ...] = ()
    regime: str = "eager"


def _execution_options():
    return tuple(
        (key, os.environ[key])
        for key in (
            "FF_DEVICE",
            "FF_AMP_DTYPE",
            "FF_CUDA_GRAPH",
            "FF_CUDA_GRAPH_FULL",
            "FF_CUDA_GRAPH_OPT",
            "FF_DETERMINISTIC",
            "FF_NN_FIXED_EPOCHS",
            "FF_NN_NORM",
            "FF_COMPILE",
            "FF_FORCE_DROPOUT_ZERO",
        )
        if key in os.environ
    )


@dataclass(frozen=True)
class RunContext:
    output_root: Path
    data_root: Path
    seed: int = 42
    run_id: str = field(default_factory=lambda: uuid4().hex)
    raw_root: Path | None = None
    artifact_sink: Callable[[Callable[[], Any]], Any] | None = execute_effect
    report_sink: Callable[[Callable[[], Any]], Any] | None = execute_effect
    # Mutable optimization state is an execution service, never recipe data.
    memo: dict | None = field(default=None, compare=False, repr=False)
    execution_options: tuple[tuple[str, str], ...] = field(default_factory=_execution_options)

    @property
    def policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(self.execution_options)

    def metadata(self, *, device=None) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "regime": self.policy.regime,
            "device": device,
            "overrides": dict(self.execution_options),
        }

    def __post_init__(self):
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        object.__setattr__(
            self, "raw_root", Path(self.raw_root or self.data_root / "raw").resolve()
        )

    @classmethod
    def defaults(cls, *, seed=42, memo=None):
        from src import config

        return cls(
            output_root=Path.cwd(),
            data_root=Path(config.SPLITS_DIR).parent,
            raw_root=Path(config.CACHE_DIR),
            seed=seed,
            memo=memo,
            execution_options=_execution_options(),
        )

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "splits"

    def output_dir(self, position: str) -> Path:
        return self.output_root / position.lower() / "outputs"

    def with_seed(self, seed: int, *, memo: dict | None = None) -> RunContext:
        return replace(
            self,
            seed=seed,
            memo=self.memo if memo is None else memo,
            execution_options=_execution_options(),
        )

    def emit_artifacts(self, effect: Callable[[], Any]) -> Any:
        if self.artifact_sink is not None:
            return self.artifact_sink(effect)
        return None

    def emit_report(self, effect: Callable[[], Any]) -> Any:
        if self.report_sink is not None:
            return self.report_sink(effect)
        return None


_ACTIVE: ContextVar[RunContext | None] = ContextVar("training_run_context", default=None)


def current_context() -> RunContext | None:
    return _ACTIVE.get()


@contextmanager
def use_context(context: RunContext) -> Iterator[RunContext]:
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)


def raw_data_dir(default: Any) -> str:
    """Legacy loader bridge; explicit per-run roots take precedence."""
    context = current_context()
    return str(context.raw_root) if context is not None else str(default)


def resolve_context(context: RunContext | None, *, seed: int, memo=None) -> RunContext:
    active = context or current_context()
    return (
        active.with_seed(seed, memo=memo) if active else RunContext.defaults(seed=seed, memo=memo)
    )


def trial_memo(cfg):
    active = current_context()
    return active.memo if active is not None else cfg.get("trial_data_memo")


def training_entrypoint(function):
    """Resolve the canonical recipe/context once, before numerical work."""
    signature = inspect.signature(function)

    @wraps(function)
    def run(position, cfg, *args, context=None, **kwargs):
        from src.training.contracts import resolve_recipe

        bound = signature.bind(position, cfg, *args, **kwargs)
        bound.apply_defaults()
        seed = context.seed if context is not None else bound.arguments["seed"]
        execution = resolve_context(context, seed=seed, memo=cfg.get("trial_data_memo"))
        if context is None and current_context() is None:
            # Preserve existing module-level test/operator overrides while
            # resolving them to an absolute path exactly once per run.
            execution = replace(
                execution, data_root=Path(function.__globals__["SPLITS_DIR"]).parent
            )
        bound.arguments["cfg"] = resolve_recipe(position, cfg)
        bound.arguments["seed"] = seed
        bound.arguments["context"] = execution
        with use_context(execution):
            return function(*bound.args, **bound.kwargs)

    return run


def runner_context(function):
    """Let specialized dataset providers read explicit roots before training."""

    @wraps(function)
    def run(*args, context=None, **kwargs):
        if context is None:
            return function(*args, **kwargs)
        with use_context(context):
            return function(*args, context=context, **kwargs)

    return run
