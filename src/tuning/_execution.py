"""Shared isolated execution for eager ablations and A/B seed groups."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

Task = TypeVar("Task")
Result = TypeVar("Result")


@contextmanager
def _worker_environment(values: Mapping[str, str]):
    """Expose worker settings before spawn re-imports the entry module."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def isolated_outputs(data_dir: str, *, share_cache: bool = False):
    """Redirect pipeline outputs while retaining the caller's cache policy."""
    original = Path.cwd()
    temporary = tempfile.mkdtemp(prefix="ff-ab-")
    try:
        os.chdir(temporary)
        Path("data").symlink_to(data_dir, target_is_directory=True)
        if share_cache:
            Path(".cache").symlink_to(original / ".cache", target_is_directory=True)
        yield
    finally:
        os.chdir(original)
        shutil.rmtree(temporary, ignore_errors=True)


def run_tasks(
    tasks: list[Task],
    execute: Callable[[Task], Result],
    *,
    max_workers: int = 1,
    preserve_order: bool = True,
    on_error: Callable[[Task, Exception], Result],
    on_result: Callable[[int, Result], None] | None = None,
    initializer: Callable | None = None,
    initargs: tuple = (),
    environment: Mapping[str, str] | None = None,
) -> list[Result]:
    """Execute a grid, preserving order or returning completion order.

    Each parallel task receives a fresh spawned process: CUDA state, imported
    config, and RNG state cannot leak between cells. Serial timing runs execute
    directly. Callers own report shapes and per-cell failure records.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if not tasks:
        return []
    results: dict[int, Result] = {}

    def record(index: int, result: Result) -> None:
        results[index] = result
        if on_result is not None:
            on_result(index, result)

    if max_workers == 1:
        for index, task in enumerate(tasks):
            try:
                result = execute(task)
            except Exception as exc:  # one failed cell must not discard its peers
                result = on_error(task, exc)
            record(index, result)
    else:
        with (
            _worker_environment(environment or {}),
            ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                max_tasks_per_child=1,
                initializer=initializer,
                initargs=initargs,
            ) as pool,
        ):
            futures = {pool.submit(execute, task): i for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # includes serialization/bootstrap failures
                    result = on_error(tasks[index], exc)
                record(index, result)
    indices = range(len(tasks)) if preserve_order else results
    return [results[index] for index in indices]
