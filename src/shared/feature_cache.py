"""Disk-backed cache for ``_prepare_position_data`` output.

Feature engineering for each (position, train_df, val_df, test_df, cfg) tuple
is deterministic but expensive: ~20-30s per call between the schedule merge
and the position-specific rolling/lag work. CV folds (4 folds per position)
and Optuna re-runs (~50 trials × 4 positions) call it many times with the same
inputs — caching the result lets us pay that cost once.

Cache layout::

    .cache/features/<position>/<key>.pkl

``key`` binds the input frames, ordered feature schema, preparation recipe,
and the schedule/team-stat artifacts read during preparation. It is shared
with tuning's in-process memoization.

In-process LRU sits in front of the disk cache so the second hit within a
single process skips the parquet read too. Set
``FF_FEATURE_CACHE_DISABLE=1`` to bypass both layers (debugging).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import marshal
import os
import pickle
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

CACHE_ROOT = Path(".cache") / "features"
_LRU_SIZE = 8
_lru_lock = threading.Lock()
_lru: dict[str, tuple] = {}
_lru_order: list[str] = []
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Dependencies of build_position_features and the position callbacks. Hash
# contents, not git HEAD: an unrelated commit must not invalidate features,
# and an uncommitted feature fix must invalidate them. Callback source files
# and the position config are added below. Update this set when preparation
# gains another implementation dependency or external data source.
_PREPARATION_SOURCES = (
    "src/config.py",
    "src/shared/feature_build.py",
    "src/shared/comparison_truth.py",
    "src/shared/weather_features.py",
    "src/contracts/feature_names.py",
    "src/shared/team_box_score.py",
    "src/shared/position_data.py",
    "src/shared/position_config.py",
    "src/features/engineer.py",
    "src/features/roster_availability.py",
    "src/data/loader.py",
    "src/data/external_sources.py",
    "src/data/nflcom_loader.py",
    "src/data/identity.py",
    "src/data/dst_scoring.py",
    "src/data/release.py",
    "src/training/contracts.py",
    "src/training/context.py",
)
_CALLBACK_KEYS = (
    "filter_fn",
    "compute_targets_fn",
    "get_feature_columns_fn",
    "add_features_fn",
    "fill_nans_fn",
)


def _disabled() -> bool:
    return os.environ.get("FF_FEATURE_CACHE_DISABLE", "0") == "1"


def _df_fingerprint(df: pd.DataFrame | None) -> dict:
    """Content fingerprint for a DataFrame.

    Uses ``pd.util.hash_pandas_object`` which is vectorised and fast (~50ms on
    30K rows). The per-row hashes are then concatenated and SHA-256'd so the
    digest is *order-sensitive* — two frames with the same rows in different
    order produce distinct fingerprints. The earlier ``.sum()`` reduction was
    commutative and would silently collide on shuffles or non-deterministic
    groupby outputs. Shape + columns are tracked separately to flag trivially
    different frames before the hash is even computed.
    """
    if df is None:
        return {"none": True}
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "index_names": list(df.index.names),
        "hash": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }


def _config_fingerprint(cfg: dict) -> dict:
    """Pull the subset of cfg keys that affect engineered features.

    Everything that ``_prepare_position_data`` reads (filter_fn,
    compute_targets_fn, get_feature_columns_fn, add_features_fn, fill_nans_fn,
    specific_features) plus the attention-history stats lists that downstream
    callers read. Resolve the feature getter, including its order: its name
    alone does not identify the feature projection it currently returns.

    ``attn_static_features`` is deliberately *not* included: it parameterises
    the neural-net static branch's column projection downstream of the cached
    output and is not consumed by ``_prepare_position_data_uncached``. Including
    it would force cache misses on Optuna sweeps that vary only the attention
    static-feature list — the cached engineered frame is bit-identical across
    those trials.
    """

    return {
        "callbacks": {name: _callable_fingerprint(cfg[name]) for name in _CALLBACK_KEYS},
        "feature_cols": list(cfg["get_feature_columns_fn"]()),
        "specific_features": list(cfg.get("specific_features") or []),
        "targets": list(cfg.get("targets") or []),
        # The train-only min-games floor changes which rows survive into the cached
        # pos_train, so it must key the cache — else a threshold change silently
        # reuses the prior threshold's filtered frame.
        "min_games_per_season": (
            cfg.get("min_games_per_season")
            if cfg.get("min_games_per_season") is not None
            else config.MIN_GAMES_PER_SEASON
        ),
        "attn_history_stats": list(cfg.get("attn_history_stats") or []),
        "opp_attn_history_stats": list(cfg.get("opp_attn_history_stats") or []),
        "opp_attn_kind": cfg.get("opp_attn_kind", "defense"),
    }


def _callable_fingerprint(fn: Callable) -> dict:
    """Identify a callback's implementation, including closure/default values."""
    code = getattr(fn, "__code__", None) or getattr(type(fn).__call__, "__code__", None)
    return {
        "module": getattr(fn, "__module__", None),
        "name": getattr(fn, "__qualname__", repr(fn)),
        "code": hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None,
        "defaults": getattr(fn, "__defaults__", None),
        "kwdefaults": getattr(fn, "__kwdefaults__", None),
        "closure": [cell.cell_contents for cell in (getattr(fn, "__closure__", None) or ())],
        "state": vars(fn) if not inspect.isfunction(fn) else None,
    }


def _file_fingerprint(path: str | Path) -> str | None:
    """Hash bytes on every lookup; same-size rewrites with preserved mtime count."""
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def _recipe_fingerprint(position: str, cfg: dict) -> dict:
    paths = {_SOURCE_ROOT / path for path in _PREPARATION_SOURCES}
    paths.add(_SOURCE_ROOT / "src" / position.lower() / "config.py")
    for name in _CALLBACK_KEYS:
        callback = cfg[name]
        source = inspect.getsourcefile(callback if inspect.isfunction(callback) else type(callback))
        if source is not None:
            paths.add(Path(source))
    # Only the cached preparation function belongs to this recipe, not the
    # unrelated training/serialization implementation in the same large file.
    pipeline = _SOURCE_ROOT / "src/shared/pipeline.py"
    tree = ast.parse(pipeline.read_text())
    prepare = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_position_data_uncached"
    )
    return {
        "files": {str(path): _file_fingerprint(path) for path in sorted(paths)},
        "prepare": hashlib.sha256(ast.dump(prepare).encode()).hexdigest(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _side_input_fingerprint() -> dict:
    # Resolve the same module-level configuration as the actual consumers.
    # Do not fetch data merely to check the cache: missing artifacts have a
    # distinct identity, and creation during compute is handled below.
    from src.data.external_sources import _seasons_cache_signature
    from src.shared import team_box_score, weather_features
    from src.training.context import raw_data_dir

    schedules = Path(raw_data_dir(weather_features.CACHE_DIR)) / (
        f"schedules_{weather_features.SEASONS[0]}_{weather_features.SEASONS[-1]}.parquet"
    )
    team_stats = Path(raw_data_dir(team_box_score.CACHE_DIR)) / (
        f"team_stats_{_seasons_cache_signature(team_box_score.SEASONS)}.parquet"
    )
    return {str(path.resolve()): _file_fingerprint(path) for path in (schedules, team_stats)}


def preparation_identity(position: str, cfg: dict) -> str:
    """Shared recipe/side-input identity for feature caches and tuning memos."""
    payload = {
        "version": 2,
        "config": _config_fingerprint(cfg),
        "recipe": _recipe_fingerprint(position, cfg),
        "side_inputs": _side_input_fingerprint(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def cache_key(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    cfg: dict,
) -> str:
    """Content identity for the full prepared-data computation."""
    payload = {
        "position": position,
        "train": _df_fingerprint(train_df),
        "val": _df_fingerprint(val_df),
        "test": _df_fingerprint(test_df),
        "preparation": preparation_identity(position, cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(position: str, key: str) -> Path:
    return CACHE_ROOT / position.upper() / f"{key}.pkl"


def _lru_get(key: str):
    with _lru_lock:
        if key in _lru:
            _lru_order.remove(key)
            _lru_order.append(key)
            return _lru[key]
    return None


def _lru_put(key: str, value) -> None:
    with _lru_lock:
        if key in _lru:
            _lru_order.remove(key)
        elif len(_lru_order) >= _LRU_SIZE:
            evicted = _lru_order.pop(0)
            _lru.pop(evicted, None)
        _lru[key] = value
        _lru_order.append(key)


def load_or_compute(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    cfg: dict,
    compute_fn: Callable[[], tuple],
) -> tuple:
    """Return cached ``_prepare_position_data`` output, computing on miss.

    ``compute_fn`` is a zero-arg callable so the caller controls when (and
    whether) the expensive path runs.
    """
    if _disabled():
        return compute_fn()

    key = cache_key(position, train_df, val_df, test_df, cfg)
    hit = _lru_get(key)
    if hit is not None:
        print(f"  [feature_cache] hit (memory) {position}/{key}")
        return hit

    disk_path = _cache_path(position, key)
    if disk_path.exists():
        try:
            with open(disk_path, "rb") as f:
                value = pickle.load(f)
            print(f"  [feature_cache] hit (disk) {position}/{key}")
            _lru_put(key, value)
            return value
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            # Corrupt cache entry — drop and recompute. Don't crash the run.
            print(f"  [feature_cache] disk read failed ({exc!r}); recomputing")
            with contextlib.suppress(OSError):
                disk_path.unlink(missing_ok=True)

    print(f"  [feature_cache] miss {position}/{key} — computing features...")
    value = compute_fn()

    # A loader may create a missing side artifact, or another process may
    # replace one while computing. Never publish ambiguous output under the
    # old identity. The next lookup recomputes against the new snapshot.
    if cache_key(position, train_df, val_df, test_df, cfg) != key:
        print("  [feature_cache] inputs changed during preparation; skipping cache write")
        return value

    tmp_path = None
    try:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=disk_path.parent, prefix=f".{key}.", suffix=".tmp", delete=False
        ) as f:
            tmp_path = Path(f.name)
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, disk_path)
    except OSError as exc:
        # If disk write fails (read-only FS, full disk), still return the value.
        print(f"  [feature_cache] disk write failed ({exc!r}); in-memory only")
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    _lru_put(key, value)
    return value


def clear_in_memory_cache() -> None:
    """Drop the in-process LRU. Useful for tests."""
    with _lru_lock:
        _lru.clear()
        _lru_order.clear()
