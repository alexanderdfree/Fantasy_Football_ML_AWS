"""Shared isolated execution for eager ablations and A/B seed groups."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from multiprocessing.connection import wait
from pathlib import Path
from typing import TypeVar

from src.training.context import RunContext, use_context

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
def isolated_outputs(
    data_dir: str, *, share_cache: bool = False, legacy_cwd: bool = False, seed: int = 42
):
    """Yield explicit paths for a cell; old external callbacks can opt into cwd isolation.

    Production runners use RunContext. ``legacy_cwd`` is a compatibility
    boundary for callbacks that still write hardcoded relative paths.
    """
    original = Path.cwd()
    temporary = tempfile.mkdtemp(prefix="ff-ab-")
    context = RunContext(output_root=Path(temporary), data_root=Path(data_dir), seed=seed)
    try:
        if legacy_cwd:
            os.chdir(temporary)
            Path("data").symlink_to(data_dir, target_is_directory=True)
            if share_cache:
                Path(".cache").symlink_to(original / ".cache", target_is_directory=True)
        with use_context(context):
            yield context
    finally:
        if legacy_cwd:
            os.chdir(original)
        shutil.rmtree(temporary, ignore_errors=True)


def _execute_in_child(sender, execute, task, initializer, initargs):
    try:
        if initializer is not None:
            initializer(*initargs)
        sender.send((True, execute(task)))
    except Exception as exc:
        try:
            sender.send((False, exc))
        except Exception:  # an exception may itself contain unpicklable state
            sender.send((False, RuntimeError(f"{type(exc).__name__}: {exc}")))
    finally:
        sender.close()


def _run_parallel(tasks, execute, max_workers, initializer, initargs, on_error, record):
    """Supervise independent children so a native crash affects only its cell."""
    context = mp.get_context("spawn")
    active = {}
    next_index = 0
    try:
        while next_index < len(tasks) or active:
            while next_index < len(tasks) and len(active) < max_workers:
                index = next_index
                next_index += 1
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_execute_in_child,
                    args=(sender, execute, tasks[index], initializer, initargs),
                )
                try:
                    process.start()
                except Exception as exc:
                    receiver.close()
                    sender.close()
                    record(index, on_error(tasks[index], exc))
                    continue
                sender.close()
                active[index] = (process, receiver)

            if not active:
                continue
            ready = wait(
                [
                    handle
                    for process, receiver in active.values()
                    for handle in (receiver, process.sentinel)
                ]
            )
            for index, (process, receiver) in list(active.items()):
                if receiver not in ready and process.sentinel not in ready:
                    continue
                payload = None
                decode_error = None
                try:
                    if receiver.poll():
                        payload = receiver.recv()
                except (EOFError, OSError):
                    pass  # native crashes close the pipe without a result
                except Exception as exc:  # result/exception reconstruction can fail
                    decode_error = exc
                finally:
                    process.join()
                    receiver.close()
                    del active[index]
                if decode_error is not None:
                    result = on_error(tasks[index], decode_error)
                elif payload is None:
                    exc = RuntimeError(
                        f"worker exited with code {process.exitcode} without a result"
                    )
                    result = on_error(tasks[index], exc)
                elif payload[0]:
                    result = payload[1]
                else:
                    result = on_error(tasks[index], payload[1])
                record(index, result)
    finally:
        for process, receiver in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            receiver.close()


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
        with _worker_environment(environment or {}):
            _run_parallel(tasks, execute, max_workers, initializer, initargs, on_error, record)
    indices = range(len(tasks)) if preserve_order else results
    return [results[index] for index in indices]
