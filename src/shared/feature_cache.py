"""Disk-backed cache for ``_prepare_position_data`` output.

Feature engineering for each (position, train_df, val_df, test_df, cfg) tuple
is deterministic but expensive: ~20-30s per call between the schedule merge
and the position-specific rolling/lag work. CV folds (4 folds per position)
and Optuna re-runs (~50 trials × 4 positions) call it many times with the same
inputs — caching the result lets us pay that cost once.

Cache layout::

    .cache/features/<position>/<key>.pkl

``key`` is a deterministic SHA-256 digest of ``(position, df content hashes,
cfg fingerprint)``. Any data or config change invalidates automatically.

In-process LRU sits in front of the disk cache so the second hit within a
single process skips the parquet read too. Set
``FF_FEATURE_CACHE_DISABLE=1`` to bypass both layers (debugging).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import threading
from collections.abc import Callable
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(".cache") / "features"
_LRU_SIZE = 8
_lru_lock = threading.Lock()
_lru: dict[str, tuple] = {}
_lru_order: list[str] = []


def _disabled() -> bool:
    return os.environ.get("FF_FEATURE_CACHE_DISABLE", "0") == "1"


def _df_fingerprint(df: pd.DataFrame | None) -> dict:
    """Content fingerprint for a DataFrame.

    Uses ``pd.util.hash_pandas_object`` which is vectorised and fast (~50ms on
    30K rows). The sum across rows is content-sensitive — any value or row
    change flips it. Shape + columns guard against collisions for trivially
    different frames.
    """
    if df is None:
        return {"none": True}
    if len(df) == 0:
        return {"empty": True, "cols": list(df.columns)}
    return {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "hash": int(pd.util.hash_pandas_object(df, index=False).values.sum()),
    }


def _config_fingerprint(cfg: dict) -> dict:
    """Pull the subset of cfg keys that affect engineered features.

    Everything that ``_prepare_position_data`` reads (filter_fn,
    compute_targets_fn, get_feature_columns_fn, add_features_fn, fill_nans_fn,
    specific_features) plus the attention-history stats lists that downstream
    callers read. Function objects are fingerprinted by qualname.
    """

    def _fn_name(v):
        return getattr(v, "__qualname__", None) or getattr(v, "__name__", None) or str(v)

    return {
        "filter_fn": _fn_name(cfg["filter_fn"]),
        "compute_targets_fn": _fn_name(cfg["compute_targets_fn"]),
        "get_feature_columns_fn": _fn_name(cfg["get_feature_columns_fn"]),
        "add_features_fn": _fn_name(cfg["add_features_fn"]),
        "fill_nans_fn": _fn_name(cfg["fill_nans_fn"]),
        "specific_features": list(cfg.get("specific_features") or []),
        "targets": list(cfg.get("targets") or []),
        "attn_history_stats": list(cfg.get("attn_history_stats") or []),
        "opp_attn_history_stats": list(cfg.get("opp_attn_history_stats") or []),
        "attn_static_features": list(cfg.get("attn_static_features") or []),
    }


def cache_key(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    cfg: dict,
) -> str:
    """Stable 16-char hex digest for the (position, data, cfg) tuple."""
    payload = {
        "position": position,
        "train": _df_fingerprint(train_df),
        "val": _df_fingerprint(val_df),
        "test": _df_fingerprint(test_df),
        "config": _config_fingerprint(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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

    disk_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = disk_path.with_suffix(".pkl.tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, disk_path)
    except OSError as exc:
        # If disk write fails (read-only FS, full disk), still return the value.
        print(f"  [feature_cache] disk write failed ({exc!r}); in-memory only")
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)

    _lru_put(key, value)
    return value


def clear_in_memory_cache() -> None:
    """Drop the in-process LRU. Useful for tests."""
    with _lru_lock:
        _lru.clear()
        _lru_order.clear()
