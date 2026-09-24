"""Frozen campaign specifications shared by local and Batch runners."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
KINDS = ("ab", "nn_tune", "lgbm_tune", "benchmark")
ROOT = Path(__file__).resolve().parents[2]
PREFIX = "campaign_runs"
OPTIONS = {
    "ab": {
        "seeds",
        "only",
        "jobs",
        "stacked_seeds",
        "stacked_epochs",
        "feature_cache",
        "device",
        "max_cells",
    },
    "nn_tune": {
        "seed",
        "n_trials",
        "timeout",
        "n_jobs",
        "parallel_backend",
        "scope",
        "stacked_seeds",
        "stacked_epochs",
        "device",
    },
    "lgbm_tune": {"seeds", "n_trials", "timeout", "n_jobs"},
    "benchmark": {"seed", "jobs", "rolling_origin", "significance", "device"},
}
PROTECTED_ENV = (
    "FF_CAMPAIGN_",
    "FF_MODEL_",
    "FF_DATA",
    "FF_BUILD_",
    "FF_TRAIN_",
    "FF_LEGACY_",
    "FF_BENCHMARK_",
    "FF_CORE_POOL_",
    "FF_RESULT_",
    "FF_AB_",
)
EXECUTION_ENV = (
    "FF_DEVICE",
    "FF_AMP_DTYPE",
    "FF_CUDA_GRAPH",
    "FF_CUDA_GRAPH_FULL",
    "FF_CUDA_GRAPH_OPT",
    "FF_DETERMINISTIC",
    "FF_NN_NORM",
    "FF_NN_FIXED_EPOCHS",
    "FF_FORCE_DROPOUT_ZERO",
    "FF_COMPILE",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "LGBM_N_JOBS",
    "LOKY_MAX_CPU_COUNT",
    "NUMEXPR_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
)
DISPATCH_ENV = {
    "FF_TUNE_AB_SPEC",
    "FF_TUNE_ABLATE_MOD",
    "FF_TUNE_ENSEMBLE_AB",
    "FF_TUNE_ENSEMBLE_COMPARE",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError(
            "Campaign/step IDs must be safe, nonempty path components (max 96 characters)"
        )
    return value


def source_fingerprint(root=ROOT):
    """Include uncommitted source/config bytes, but not generated artifacts."""
    root = Path(root)
    files = [
        p
        for p in (root / "src").rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".yaml", ".yml"}
        and not {"outputs", "__pycache__", "node_modules"}.intersection(p.parts)
    ]
    files += [root / "pyproject.toml", *root.glob("requirements*.txt")]
    return identity(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))
            if p.is_file()
        }
    )


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate(document):
    result = copy.deepcopy(document)
    if not isinstance(result, dict) or result.get("version") != 1:
        raise ValueError("Campaign version must be 1")
    if set(result) - {"version", "id", "steps", "dataset_id", "data_prefix"}:
        raise ValueError("Unknown campaign fields")
    safe_id(result.get("id"))
    if "dataset_id" in result and (
        not isinstance(result["dataset_id"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", result["dataset_id"])
    ):
        raise ValueError("dataset_id must be an immutable SHA-256 identity")
    prefix = result.get("data_prefix", "data")
    if not isinstance(prefix, str):
        raise ValueError("Invalid data prefix")
    prefix = prefix.strip("/")
    if not prefix or any(p in {".", ".."} for p in prefix.split("/")):
        raise ValueError("Invalid data prefix")
    if prefix != "data":
        raise ValueError("Campaigns use the canonical immutable data registry")
    result["data_prefix"] = prefix
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Campaign requires at least one step")
    names = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) - {
            "id",
            "kind",
            "positions",
            "spec",
            "options",
            "env",
        }:
            raise ValueError("Unknown or invalid step fields")
        name = safe_id(step.get("id"))
        if name in names:
            raise ValueError(f"Duplicate step: {name}")
        names.add(name)
        kind = step.get("kind")
        if kind not in KINDS:
            raise ValueError(f"Unsupported workload: {kind}")
        if kind == "ab":
            if not isinstance(step.get("spec"), str) or not re.fullmatch(
                r"src\.tuning\.[A-Za-z_][A-Za-z_0-9.]*", step["spec"]
            ):
                raise ValueError("A/B steps require an importable src.tuning spec")
        elif "spec" in step:
            raise ValueError("spec is only valid for A/B steps")
        positions = step.get("positions", None if kind == "ab" else list(POSITIONS))
        if positions is not None and (
            not isinstance(positions, list)
            or not positions
            or not all(isinstance(pos, str) for pos in positions)
            or len(positions) != len(set(positions))
            or set(positions) - set(POSITIONS)
        ):
            raise ValueError("positions must be a nonempty unique subset of the six positions")
        step["positions"] = positions
        options = step.setdefault("options", {})
        if not isinstance(options, dict):
            raise ValueError("Step options must be a mapping")
        if set(options) - OPTIONS[kind]:
            raise ValueError(f"Unsupported {kind} options: {set(options) - OPTIONS[kind]}")
        for key in ("jobs", "n_trials", "timeout", "stacked_epochs", "max_cells"):
            if key in options:
                _positive(options[key], key)
        if "n_jobs" in options and options["n_jobs"] != "auto":
            _positive(options["n_jobs"], "n_jobs")
        if kind == "lgbm_tune" and options.get("n_jobs") == "auto":
            raise ValueError("LightGBM n_jobs must be an integer; omit for automatic")
        seeds = options.get("seeds", [options.get("seed", 42)])
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(seeds) != len(set(seeds))
        ):
            raise ValueError("Seeds must be unique integers in [0, 2**32)")
        if "only" in options and (
            not isinstance(options["only"], list)
            or not options["only"]
            or not all(isinstance(v, str) and v for v in options["only"])
        ):
            raise ValueError("only must be a nonempty list of variant names")
        for flag in ("feature_cache", "rolling_origin", "significance"):
            if flag in options and type(options[flag]) is not bool:
                raise ValueError(f"{flag} must be boolean")
        if "stacked_seeds" in options:
            value = options["stacked_seeds"]
            if kind == "ab" and type(value) is not bool:
                raise ValueError("A/B stacked_seeds must be boolean")
            if kind == "nn_tune" and (type(value) is not int or value < 0 or value == 1):
                raise ValueError("NN stacked_seeds must be 0 or >=2")
        for key, choices in (
            ("device", {"auto", "cpu", "cuda", "mps"}),
            ("scope", {"full", "history"}),
            ("parallel_backend", {"auto", "thread", "mps"}),
        ):
            if key in options and (
                not isinstance(options[key], str) or options[key] not in choices
            ):
                raise ValueError(f"Unsupported {key}: {options[key]}")
        if (
            kind == "nn_tune"
            and options.get("scope") == "history"
            and set(positions) & {"K", "DST"}
        ):
            raise ValueError("History-scope tuning supports QB/RB/WR/TE only")
        env = step.setdefault("env", {})
        if not isinstance(env, dict):
            raise ValueError("env must be a mapping")
        for key, value in env.items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"FF_[A-Z0-9_]+", key)
                or key.startswith(PROTECTED_ENV)
                or key in {"FF_FRESH", "FF_CACHE_DIR", *DISPATCH_ENV}
                or re.search(r"TOKEN|PASSWORD|SECRET|CREDENTIAL|API_KEY", key)
            ):
                raise ValueError(f"Environment key is managed or unsupported: {key}")
            if not isinstance(value, str):
                raise ValueError("Environment values must be strings")
    canonical(result)
    return result


def resource_class(step):
    device = step["options"].get("device", step["env"].get("FF_DEVICE", "auto"))
    return "cpu" if step["kind"] == "lgbm_tune" or device == "cpu" else "gpu"


def work_units(spec, backend):
    """One local campaign, or one allocated worker per position/resource lane."""
    if backend == "local":
        return [
            {
                "id": "local",
                "position": None,
                "resource": "local",
                "steps": [s["id"] for s in spec["steps"]],
            }
        ]
    groups = {}
    for step in spec["steps"]:
        if step["options"].get("device", step["env"].get("FF_DEVICE")) == "mps":
            raise ValueError("Apple MPS cannot run on AWS Batch")
        lane = resource_class(step)
        for pos in step["positions"]:
            key = f"{pos}-{lane}"
            unit = groups.setdefault(
                key, {"id": key, "position": pos, "resource": lane, "steps": []}
            )
            unit["steps"].append(step["id"])
    return list(groups.values())


def execution_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if key in EXECUTION_ENV
        or (
            key.startswith("FF_")
            and not key.startswith(PROTECTED_ENV)
            and key not in {"FF_FRESH", "FF_CACHE_DIR", *DISPATCH_ENV}
            and not re.search(r"TOKEN|PASSWORD|SECRET|CREDENTIAL|API_KEY", key)
        )
    }
