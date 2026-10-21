"""Conservative identities for whole fits and pinned-artifact inference."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import platform
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.result_store import file_digest

ROOT = Path(__file__).resolve().parents[2]


class Uncacheable(ValueError):
    """A runtime value has no supported exact identity; execute it freshly."""


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def code_identity(code):
    # marshal's string-intern flags can change after a function executes.
    # Describe semantic code fields instead of serializing interpreter state.
    return digest(
        {
            "bytecode": code.co_code.hex(),
            "names": code.co_names,
            "variables": code.co_varnames,
            "free": code.co_freevars,
            "cells": code.co_cellvars,
            "arguments": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
            "flags": code.co_flags,
            "constants": [
                code_identity(item) if isinstance(item, types.CodeType) else stable(item)
                for item in code.co_consts
            ],
        }
    )


@functools.lru_cache(maxsize=4096)
def _file_hash(path, size, modified, changed, inode):
    return file_digest(Path(path))


def fingerprint_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    result = _file_hash(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino
    )
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise Uncacheable(f"Input changed while fingerprinting: {path}")
    return result


def stable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, np.generic):
        return stable(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, inspect.Signature):
        return {"signature": str(value)}
    if inspect.isclass(value):
        try:
            source = inspect.getsourcefile(value)
        except TypeError:
            source = None
        return {
            "class": f"{value.__module__}.{value.__qualname__}",
            "source": fingerprint_file(source) if source else None,
            "constants": {
                key: stable(item)
                for key, item in vars(value).items()
                if key.isupper() and not callable(item)
            },
        }
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        # Includes row/index order, dtypes, categorical metadata and frame attrs.
        return {
            "array_type": type(value).__name__,
            "sha256": hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest(),
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise Uncacheable("Non-string mapping key")
        return {key: stable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [stable(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {
            "set": sorted(
                [stable(item) for item in value], key=lambda item: json.dumps(item, sort_keys=True)
            )
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": stable(dataclasses.asdict(value)),
        }
    if isinstance(value, functools.partial):
        return {
            "partial": stable(value.func),
            "args": stable(value.args),
            "kwargs": stable(value.keywords),
        }
    if inspect.isfunction(value):
        closure = [stable(cell.cell_contents) for cell in (value.__closure__ or ())]
        source = inspect.getsourcefile(value)
        if source is None:
            raise Uncacheable("Callback has no source file")
        return {
            "function": f"{value.__module__}.{value.__qualname__}",
            "code": code_identity(value.__code__),
            "source": fingerprint_file(source),
            "closure": closure,
            "defaults": stable(value.__defaults__),
            "kwdefaults": stable(value.__kwdefaults__),
        }
    raise Uncacheable(f"Unsupported runtime value: {type(value).__name__}")


def source_manifest(position):
    files = {
        path
        for name in (
            "shared",
            "data",
            "features",
            "prediction",
            "training",
            "contracts",
            "evaluation",
            position.lower(),
        )
        for path in (ROOT / "src" / name).rglob("*.py")
    }
    files.update([ROOT / "src/config.py", ROOT / "pyproject.toml"])
    # Also identify loaded numerical callables. Research runners sometimes
    # patch trainer/model methods without changing a file on disk.
    runtime = {}
    for name in (
        "src.shared.pipeline",
        "src.shared.training",
        "src.shared.neural_net",
        "src.shared.models",
        "src.features.engineer",
        "src.data.preprocessing",
        "src.prediction.frames",
        "src.prediction.predictor",
        f"src.{position.lower()}.features",
    ):
        module = importlib.import_module(name)
        for label, value in vars(module).items():
            members = (
                vars(value).items()
                if inspect.isclass(value) and value.__module__ == name
                else (("", value),)
            )
            for member, function in members:
                if isinstance(function, (classmethod, staticmethod)):
                    function = function.__func__
                if inspect.isfunction(function):
                    if member in {"__repr__", "__str__"}:
                        continue
                    code = function.__code__
                    if hasattr(function, "__wrapped__"):
                        known = {
                            (
                                str(ROOT / "src/training/context.py"),
                                "training_entrypoint.<locals>.run",
                            ),
                            (str(ROOT / "src/training/context.py"), "runner_context.<locals>.run"),
                            (
                                str(ROOT / "src/prediction/bundle.py"),
                                "record_constructor.<locals>.initialize",
                            ),
                            (
                                str(Path(contextlib.__file__).resolve()),
                                "contextmanager.<locals>.helper",
                            ),
                        }
                        if (str(Path(code.co_filename).resolve()), code.co_qualname) not in known:
                            raise Uncacheable(
                                f"Active wrapper must execute: {name}.{label}.{member}"
                            )
                    runtime[f"{name}.{label}.{member}"] = stable(function)
    return {
        "files": {str(path.relative_to(ROOT)): fingerprint_file(path) for path in sorted(files)},
        "runtime": runtime,
    }


def source_identity(position):
    return digest(source_manifest(position))


def data_identity(context):
    """Cover raw side inputs as well as splits; never trust a marker alone."""
    files = {}
    for label, root in (("raw", context.raw_root), ("splits", context.splits_dir)):
        root = Path(root)
        if not root.is_dir():
            files[label] = None
            continue
        # Same historical root layout used by the sealed-release producer.
        paths = [p for p in root.iterdir() if p.is_file()]
        providers = root / "provider_sources"
        if providers.is_dir():
            paths.extend(p for p in providers.iterdir() if p.is_file())
        for path in sorted(paths):
            if path.name.startswith(".release-"):
                continue
            files[f"{label}/{path.relative_to(root)}"] = fingerprint_file(path)
    return digest(files)


def execution_identity(device=None):
    import torch

    from src.shared.pipeline import _nn_device
    from src.shared.utils import amp_dtype, cuda_graph_enabled

    device = torch.device(device) if device is not None else _nn_device()
    versions = {}
    for name in (
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "optuna",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    excluded = ("FF_RESULT_", "FF_CAMPAIGN_", "FF_BENCHMARK_", "FF_MODEL_S3_")
    bookkeeping = {
        "FF_FRESH",
        "FF_AB_RUN_ID",
        "FF_AB_S3_PREFIX",
        "FF_TUNE_AB_SPEC",
        "FF_TRAIN_GIT_SHA",
        "FF_CORE_POOL_ADDR",
        "FF_CORE_POOL_POS",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if (key.startswith("FF_") and not key.startswith(excluded) and key not in bookkeeping)
        or key in {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LGBM_N_JOBS"}
    }
    return {
        "device": str(device),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "environment": env,
        "threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "amp": str(amp_dtype()) if device.type == "cuda" else "float32",
        "graphs": cuda_graph_enabled() if device.type == "cuda" else False,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn": {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
            "allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def fit_identity(position, recipe, arguments, context, entrypoint):
    from src.shared.training import MultiHeadTrainer

    if MultiHeadTrainer.train.__module__ != "src.shared.training":
        raise Uncacheable("Active training instrumentation must execute")
    if recipe.get("epoch_callback") is not None:
        raise Uncacheable("Epoch callbacks must execute, including Optuna pruning")
    config = {
        key: value
        for key, value in recipe.items()
        if key not in {"trial_data_memo", "epoch_callback", "nn_log_every"}
    }
    return {
        "schema": 1,
        "kind": "fit",
        "position": position,
        "entrypoint": entrypoint,
        "recipe": stable(config),
        "code": source_identity(position),
        "data": data_identity(context),
        "seed": context.seed,
        "frames": {key: stable(value) for key, value in arguments.items() if key.endswith("_df")},
        "execution": execution_identity(),
    }
