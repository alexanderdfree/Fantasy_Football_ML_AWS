"""Unit tests for ``src.shared.feature_cache``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.shared import feature_cache


def _toy_df(seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0 + seed]})


def _toy_cfg(name: str = "fnA") -> dict:
    """Minimal cfg matching the keys ``cache_key`` reads."""

    def filter_fn(df):
        return df

    filter_fn.__qualname__ = f"toy.{name}.filter_fn"

    def compute_targets_fn(df):
        return df

    compute_targets_fn.__qualname__ = f"toy.{name}.compute_targets_fn"

    def get_feature_columns_fn():
        return ["a", "b"]

    get_feature_columns_fn.__qualname__ = f"toy.{name}.get_feature_columns_fn"

    def add_features_fn(*args, **kwargs):
        return args

    add_features_fn.__qualname__ = f"toy.{name}.add_features_fn"

    def fill_nans_fn(*args):
        return args

    fill_nans_fn.__qualname__ = f"toy.{name}.fill_nans_fn"

    return {
        "filter_fn": filter_fn,
        "compute_targets_fn": compute_targets_fn,
        "get_feature_columns_fn": get_feature_columns_fn,
        "add_features_fn": add_features_fn,
        "fill_nans_fn": fill_nans_fn,
        "specific_features": ["a"],
        "targets": ["y"],
        "attn_history_stats": ["a"],
        "opp_attn_history_stats": [],
        "attn_static_features": ["a"],
    }


@pytest.fixture(autouse=True)
def _clear_cache(tmp_path, monkeypatch):
    """Isolate every test with a per-test cache dir + clean LRU."""
    monkeypatch.setattr(feature_cache, "CACHE_ROOT", tmp_path / "features")
    feature_cache.clear_in_memory_cache()
    monkeypatch.delenv("FF_FEATURE_CACHE_DISABLE", raising=False)
    yield
    feature_cache.clear_in_memory_cache()


@pytest.mark.unit
class TestCacheKey:
    def test_same_inputs_yield_same_key(self):
        df = _toy_df()
        cfg = _toy_cfg()
        k1 = feature_cache.cache_key("RB", df, df, None, cfg)
        k2 = feature_cache.cache_key("RB", df.copy(), df.copy(), None, cfg)
        assert k1 == k2

    def test_different_position_yields_different_key(self):
        df = _toy_df()
        cfg = _toy_cfg()
        k_rb = feature_cache.cache_key("RB", df, df, None, cfg)
        k_qb = feature_cache.cache_key("QB", df, df, None, cfg)
        assert k_rb != k_qb

    def test_different_data_yields_different_key(self):
        cfg = _toy_cfg()
        k1 = feature_cache.cache_key("RB", _toy_df(0), _toy_df(0), None, cfg)
        k2 = feature_cache.cache_key("RB", _toy_df(1), _toy_df(0), None, cfg)
        assert k1 != k2

    def test_different_config_yields_different_key(self):
        df = _toy_df()
        k1 = feature_cache.cache_key("RB", df, df, None, _toy_cfg("A"))
        k2 = feature_cache.cache_key("RB", df, df, None, _toy_cfg("B"))
        assert k1 != k2

    def test_test_df_none_vs_empty_differ(self):
        df = _toy_df()
        cfg = _toy_cfg()
        k_none = feature_cache.cache_key("RB", df, df, None, cfg)
        k_empty = feature_cache.cache_key("RB", df, df, df.iloc[:0], cfg)
        assert k_none != k_empty


@pytest.mark.unit
class TestLoadOrCompute:
    def test_miss_then_hit_in_memory(self):
        df = _toy_df()
        cfg = _toy_cfg()
        calls = []

        def compute():
            calls.append(1)
            return ("value", 42)

        v1 = feature_cache.load_or_compute("RB", df, df, None, cfg, compute)
        v2 = feature_cache.load_or_compute("RB", df, df, None, cfg, compute)
        assert v1 == ("value", 42)
        assert v2 == ("value", 42)
        # compute_fn called exactly once — second call hit the LRU
        assert sum(calls) == 1

    def test_disk_persists_across_lru_eviction(self):
        df = _toy_df()
        cfg = _toy_cfg()
        calls = []

        def compute():
            calls.append(1)
            return ("disk_value",)

        feature_cache.load_or_compute("RB", df, df, None, cfg, compute)
        feature_cache.clear_in_memory_cache()  # Force LRU miss
        v = feature_cache.load_or_compute(
            "RB", df, df, None, cfg, lambda: pytest.fail("should not recompute")
        )
        assert v == ("disk_value",)
        assert sum(calls) == 1

    def test_disabled_bypasses_cache(self, monkeypatch):
        monkeypatch.setenv("FF_FEATURE_CACHE_DISABLE", "1")
        df = _toy_df()
        cfg = _toy_cfg()
        calls = []

        def compute():
            calls.append(1)
            return ("x",)

        feature_cache.load_or_compute("RB", df, df, None, cfg, compute)
        feature_cache.load_or_compute("RB", df, df, None, cfg, compute)
        # Disabled: every call recomputes, neither layer used
        assert sum(calls) == 2

    def test_corrupt_disk_entry_recovers(self, tmp_path):
        df = _toy_df()
        cfg = _toy_cfg()

        # Pre-populate
        feature_cache.load_or_compute("RB", df, df, None, cfg, lambda: ("good",))

        # Corrupt the disk entry
        key = feature_cache.cache_key("RB", df, df, None, cfg)
        path = feature_cache._cache_path("RB", key)
        path.write_bytes(b"not a pickle")

        # Drop LRU so the disk path is exercised
        feature_cache.clear_in_memory_cache()

        v = feature_cache.load_or_compute("RB", df, df, None, cfg, lambda: ("recovered",))
        assert v == ("recovered",)
