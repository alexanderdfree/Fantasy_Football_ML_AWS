"""Unit tests for ``src.shared.feature_cache``."""

from __future__ import annotations

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

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
    @pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
    def test_production_recipes_have_stable_complete_identity(self, position):
        from src.shared.registry import get_config

        cfg = get_config(position)
        df = _toy_df()
        assert feature_cache._config_fingerprint(cfg)["feature_cols"] == list(
            cfg["get_feature_columns_fn"]()
        )
        assert feature_cache.cache_key(position, df, df, None, cfg) == feature_cache.cache_key(
            position, df.copy(), df.copy(), None, cfg
        )

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

    def test_resolved_feature_columns_and_order_invalidate(self):
        columns = ["a"]
        cfg = _toy_cfg()
        cfg["get_feature_columns_fn"] = lambda: list(columns)
        df = _toy_df()

        def compute():
            return tuple(columns)

        assert feature_cache.load_or_compute("RB", df, df, None, cfg, compute) == ("a",)
        columns.append("b")
        assert feature_cache.load_or_compute("RB", df, df, None, cfg, compute) == ("a", "b")
        columns.reverse()
        assert feature_cache.load_or_compute("RB", df, df, None, cfg, compute) == ("b", "a")

    def test_same_named_changed_callback_invalidates(self):
        df = _toy_df()
        cfg = _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        namespace = {}
        exec("def filter_fn(df):\n    return df.iloc[1:]", namespace)
        replacement = namespace["filter_fn"]
        replacement.__qualname__ = cfg["filter_fn"].__qualname__
        replacement.__module__ = cfg["filter_fn"].__module__
        cfg["filter_fn"] = replacement
        assert feature_cache.cache_key("RB", df, df, None, cfg) != before

    def test_helper_source_change_invalidates(self, tmp_path, monkeypatch):
        source = tmp_path / "helper.py"
        source.write_text("def helper(): return 1\n")
        monkeypatch.setattr(feature_cache, "_PREPARATION_SOURCES", (str(source),))
        df, cfg = _toy_df(), _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        source.write_text("def helper(): return 2\n")
        assert feature_cache.cache_key("RB", df, df, None, cfg) != before

    @pytest.mark.parametrize("kind", ["schedules", "team_stats"])
    def test_side_input_content_change_invalidates_with_same_stat(
        self, kind, tmp_path, monkeypatch
    ):
        from src.shared import team_box_score, weather_features

        monkeypatch.setattr(weather_features, "CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(weather_features, "SEASONS", [2025])
        monkeypatch.setattr(team_box_score, "CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(team_box_score, "SEASONS", [2025])
        path = tmp_path / f"{kind}_2025_2025.parquet"
        path.write_bytes(b"old snapshot")
        stat = path.stat()
        df, cfg = _toy_df(), _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        path.write_bytes(b"new snapshot")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert feature_cache.cache_key("RB", df, df, None, cfg) != before

    def test_frame_schema_and_index_invalidate(self):
        df, cfg = _toy_df(), _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        assert feature_cache.cache_key("RB", df.astype({"a": "float64"}), df, None, cfg) != before
        reindexed = df.set_axis([3, 4, 5])
        assert feature_cache.cache_key("RB", reindexed, df, None, cfg) != before

    def test_training_hyperparameters_do_not_invalidate(self):
        df, cfg = _toy_df(), _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        cfg.update(nn_lr=0.123, nn_epochs=999, attn_static_features=["b"])
        assert feature_cache.cache_key("RB", df, df, None, cfg) == before

    def test_numpy_runtime_change_invalidates(self, monkeypatch):
        df, cfg = _toy_df(), _toy_cfg()
        before = feature_cache.cache_key("RB", df, df, None, cfg)
        monkeypatch.setattr(feature_cache.np, "__version__", "different-runtime")
        assert feature_cache.cache_key("RB", df, df, None, cfg) != before


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

    def test_disabled_does_not_resolve_identity(self, monkeypatch):
        monkeypatch.setenv("FF_FEATURE_CACHE_DISABLE", "1")
        monkeypatch.setattr(feature_cache, "cache_key", lambda *args: pytest.fail("cache used"))
        assert feature_cache.load_or_compute(
            "RB", _toy_df(), _toy_df(), None, {}, lambda: (1,)
        ) == (1,)

    def test_side_input_created_during_compute_is_not_cached(self, monkeypatch):
        identity = {"snapshot": None}
        monkeypatch.setattr(feature_cache, "_side_input_fingerprint", lambda: dict(identity))
        df, cfg = _toy_df(), _toy_cfg()
        old_key = feature_cache.cache_key("RB", df, df, None, cfg)

        def compute():
            identity["snapshot"] = "created"
            return ("value",)

        assert feature_cache.load_or_compute("RB", df, df, None, cfg, compute) == ("value",)
        assert feature_cache._lru_get(old_key) is None
        assert not feature_cache._cache_path("RB", old_key).exists()

    def test_concurrent_writers_use_unique_temporary_files(self, monkeypatch):
        df, cfg = _toy_df(), _toy_cfg()
        barrier = threading.Barrier(2)
        original_dump = pickle.dump
        temporary_names = []

        def synchronized_dump(value, stream, **kwargs):
            temporary_names.append(stream.name)
            barrier.wait(timeout=10)
            original_dump(value, stream, **kwargs)

        monkeypatch.setattr(feature_cache.pickle, "dump", synchronized_dump)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    feature_cache.load_or_compute,
                    "RB",
                    df,
                    df,
                    None,
                    cfg,
                    lambda value=value: (value,),
                )
                for value in ("first", "second")
            ]
            assert {future.result() for future in futures} == {("first",), ("second",)}
        assert len(set(temporary_names)) == 2
        path = feature_cache._cache_path("RB", feature_cache.cache_key("RB", df, df, None, cfg))
        with path.open("rb") as stream:
            assert pickle.load(stream) in {("first",), ("second",)}
        assert not list(path.parent.glob("*.tmp"))
