"""Unit tests for ``src.shared.feature_cache``."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.shared import feature_cache, team_box_score, weather_features
from src.tuning.feature_groups import drop_columns_mutator

_LOAD_SCHEDULES = weather_features._load_schedules


def _schedule_input():
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "home_team": "KC",
                "away_team": "BUF",
                "spread_line": 4.0,
                "total_line": 44.0,
                "roof": "outdoors",
                "surface": "grass",
                "temp": 60.0,
                "wind": 1.0,
                "home_rest": 7,
                "away_rest": 7,
                "div_game": 0,
                "home_score": 24,
                "away_score": 20,
            }
        ]
    )


def _box_input():
    return pd.DataFrame(
        [
            {
                "team": team,
                "season": 2025,
                "week": 1,
                "attempts": 30.0,
                "completions": 20.0,
                "passing_yards": 200.0,
                "carries": 20.0,
                "rushing_yards": 100.0,
                "passing_interceptions": 0.0,
                "rushing_fumbles_lost": 0.0,
                "receiving_fumbles_lost": 0.0,
                "sack_fumbles_lost": 0.0,
            }
            for team in ("KC", "BUF")
        ]
    )


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
    source_root = tmp_path / "src"
    source_root.mkdir()
    monkeypatch.setattr(feature_cache, "_SOURCE_ROOT", source_root, raising=False)
    monkeypatch.setattr(weather_features, "_load_schedules", _schedule_input)
    monkeypatch.setattr(team_box_score, "load_team_week_stats", lambda *a, **k: _box_input())
    feature_cache.clear_in_memory_cache()
    monkeypatch.delenv("FF_FEATURE_CACHE_DISABLE", raising=False)
    yield
    feature_cache.clear_in_memory_cache()


def _runtime_frame_and_config():
    frame = pd.DataFrame(
        {
            "player_id": ["p"],
            "season": [2025],
            "week": [1],
            "recent_team": ["KC"],
            "opponent_team": ["BUF"],
            "position": ["QB"],
            "y": [1.0],
        }
    )
    cfg = _toy_cfg()
    cfg.update(
        filter_fn=lambda df: df.copy(),
        compute_targets_fn=lambda df: df.copy(),
        get_feature_columns_fn=lambda: ["wind_adjusted", "team_pass_attempts"],
        add_features_fn=lambda a, b, c, **kwargs: (a, b, c),
        fill_nans_fn=lambda a, b, c, *args: (a, b, c),
        min_games_per_season=1,
        specific_features=[],
    )
    return frame, cfg


@pytest.mark.unit
@pytest.mark.parametrize("disk_hit", [False, True])
@pytest.mark.parametrize("source", ["schedule", "box_score"])
def test_runtime_lookup_changes_invalidate_prepared_features(monkeypatch, disk_hit, source):
    from src.shared.pipeline import _prepare_position_data, _prepare_position_data_uncached

    schedule, box = _schedule_input(), _box_input()
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: schedule)
    monkeypatch.setattr(team_box_score, "load_team_week_stats", lambda *a, **k: box)
    frame, cfg = _runtime_frame_and_config()
    first = _prepare_position_data("QB", cfg, frame, frame, frame)
    assert first[0].tolist() == [[1.0, 30.0]]
    if source == "schedule":
        schedule["wind"] = 20.0
    else:
        box.loc[box.team.eq("KC"), "attempts"] = 45.0
    if disk_hit:
        feature_cache.clear_in_memory_cache()
    result = _prepare_position_data("QB", cfg, frame, frame, frame)
    expected = _prepare_position_data_uncached("QB", cfg, frame, frame, frame)
    assert result[0].tolist() == expected[0].tolist()
    assert result[0].tolist() != first[0].tolist()


@pytest.mark.unit
@pytest.mark.parametrize("source", ["schedule", "box_score"])
def test_runtime_lookup_disk_reload_changes_cache_key(tmp_path, monkeypatch, source):
    from src.data.loader import load_team_week_stats
    from src.shared.pipeline import _prepare_position_data

    raw = tmp_path / "raw"
    raw.mkdir()
    schedule_path = raw / "schedules_2025_2025.parquet"
    box_path = raw / "team_stats_2025_2025.parquet"
    schedule, box = _schedule_input(), _box_input()
    schedule.to_parquet(schedule_path)
    box.to_parquet(box_path)
    monkeypatch.setattr(weather_features, "CACHE_DIR", str(raw))
    monkeypatch.setattr(weather_features, "SEASONS", [2025])
    monkeypatch.setattr(weather_features, "_schedule_cache", None)
    monkeypatch.setattr(weather_features, "_load_schedules", _LOAD_SCHEDULES)
    monkeypatch.setattr(team_box_score, "CACHE_DIR", str(raw))
    monkeypatch.setattr(team_box_score, "SEASONS", [2025])
    monkeypatch.setattr(team_box_score, "load_team_week_stats", load_team_week_stats)
    frame, cfg = _runtime_frame_and_config()
    first = _prepare_position_data("QB", cfg, frame, frame, frame)
    if source == "schedule":
        schedule["wind"] = 20.0
        schedule.to_parquet(schedule_path)
    else:
        box.loc[box.team.eq("KC"), "attempts"] = 45.0
        box.to_parquet(box_path)
    # Simulate a fresh CLI/context reloading its canonical schedule cache.
    weather_features._schedule_cache = None
    feature_cache.clear_in_memory_cache()
    result = _prepare_position_data("QB", cfg, frame, frame, frame)
    assert first[0].tolist() == [[1.0, 30.0]]
    assert result[0].tolist() == ([[20.0, 30.0]] if source == "schedule" else [[1.0, 45.0]])


@pytest.mark.unit
@pytest.mark.parametrize("source", ["schedule", "box_score"])
def test_runtime_change_during_compute_does_not_publish_cache(monkeypatch, source):
    schedule, box = _schedule_input(), _box_input()
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: schedule)
    monkeypatch.setattr(team_box_score, "load_team_week_stats", lambda *a, **k: box)
    frame, cfg = _runtime_frame_and_config()
    key = feature_cache.cache_key("QB", frame, frame, frame, cfg)

    def changing_compute():
        if source == "schedule":
            schedule["wind"] = 20.0
        else:
            box["attempts"] = 45.0
        return ("computed before source update",)

    result = feature_cache.load_or_compute("QB", frame, frame, frame, cfg, changing_compute)
    assert result == ("computed before source update",)
    assert feature_cache._lru_get(key) is None
    assert not list(feature_cache.CACHE_ROOT.rglob("*.pkl"))


@pytest.mark.unit
def test_unused_lookup_columns_and_model_settings_keep_cache_hit(monkeypatch):
    schedule, box = _schedule_input(), _box_input()
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: schedule)
    monkeypatch.setattr(team_box_score, "load_team_week_stats", lambda *a, **k: box)
    frame, cfg = _runtime_frame_and_config()
    feature_cache.load_or_compute("QB", frame, frame, frame, cfg, lambda: ("features",))
    schedule["unused_note"] = "changed"
    box["unused_stat"] = 999.0
    changed = {**cfg, "nn_lr": 0.5, "nn_dropout": 0.9, "attn_static_features": []}
    feature_cache.clear_in_memory_cache()
    result = feature_cache.load_or_compute(
        "QB", frame, frame, frame, changed, lambda: pytest.fail("irrelevant inputs should hit")
    )
    assert result == ("features",)


@pytest.mark.unit
def test_premerged_frames_do_not_read_unused_lookups(monkeypatch):
    frame, cfg = _runtime_frame_and_config()
    frame["_schedule_merged"] = True
    frame["_team_box_score_merged"] = True
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: pytest.fail("already merged"))
    monkeypatch.setattr(
        team_box_score, "_build_team_box_score_lookup", lambda: pytest.fail("already merged")
    )
    assert feature_cache.load_or_compute("QB", frame, frame, frame, cfg, lambda: ("ready",)) == (
        "ready",
    )


@pytest.mark.unit
def test_runtime_input_disappearing_after_compute_keeps_result_uncached(monkeypatch):
    frame, cfg = _runtime_frame_and_config()
    key = feature_cache.cache_key("QB", frame, frame, frame, cfg)

    def disappeared():
        raise FileNotFoundError("schedule removed during refresh")

    def compute():
        monkeypatch.setattr(weather_features, "_load_schedules", disappeared)
        return ("completed features",)

    assert feature_cache.load_or_compute("QB", frame, frame, frame, cfg, compute) == (
        "completed features",
    )
    assert feature_cache._lru_get(key) is None
    assert not list(feature_cache.CACHE_ROOT.rglob("*.pkl"))


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
    def test_unusable_disk_cache_keeps_computed_value_in_memory(self):
        feature_cache.CACHE_ROOT.write_text("not a directory")
        cfg, df = _toy_cfg(), _toy_df()
        result = feature_cache.load_or_compute("WR", df, df, None, cfg, lambda: ("features",))
        assert result == ("features",)
        assert (
            feature_cache.load_or_compute(
                "WR", df, df, None, cfg, lambda: pytest.fail("memory cache should still work")
            )
            == result
        )

    @pytest.mark.parametrize(
        "relative_path",
        [
            "wr/features.py",
            "shared/feature_build.py",
            "data/loader.py",
            "features/engineer.py",
            "config.py",
        ],
    )
    def test_implementation_change_invalidates_disk_cache(self, relative_path):
        source = feature_cache._SOURCE_ROOT / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("def transform(value):\n    return value + 1\n")
        cfg, df = _toy_cfg(), _toy_df()
        feature_cache.load_or_compute("WR", df, df, None, cfg, lambda: ("old features",))
        feature_cache.clear_in_memory_cache()

        source.write_text("def transform(value):\n    return value + 2\n")
        result = feature_cache.load_or_compute(
            "WR", df, df, None, cfg, lambda: ("corrected features",)
        )

        assert result == ("corrected features",)
        source.touch()
        feature_cache.clear_in_memory_cache()
        assert (
            feature_cache.load_or_compute(
                "WR", df, df, None, cfg, lambda: pytest.fail("mtime alone must not invalidate")
            )
            == result
        )

    def test_new_untracked_source_module_invalidates_cache(self):
        cfg, df = _toy_cfg(), _toy_df()
        feature_cache.load_or_compute("WR", df, df, None, cfg, lambda: ("old features",))
        source = feature_cache._SOURCE_ROOT / "shared" / "new_helper.py"
        source.parent.mkdir()
        source.write_text("VALUE = 2\n")
        result = feature_cache.load_or_compute("WR", df, df, None, cfg, lambda: ("new features",))
        assert result == ("new features",)

    @pytest.mark.parametrize("disk_hit", [False, True])
    def test_feature_selection_closures_do_not_share_cached_columns(self, disk_hit):
        cfg = _toy_cfg()
        drop_a = drop_columns_mutator(frozenset({"a"}))(deepcopy(cfg))
        drop_b = drop_columns_mutator(frozenset({"b"}))(deepcopy(cfg))
        df = _toy_df()

        first = feature_cache.load_or_compute(
            "WR", df, df, None, drop_a, lambda: (drop_a["get_feature_columns_fn"](),)
        )
        if disk_hit:
            feature_cache.clear_in_memory_cache()
        second = feature_cache.load_or_compute(
            "WR", df, df, None, drop_b, lambda: (drop_b["get_feature_columns_fn"](),)
        )

        assert first == (["b"],)
        assert second == (["a"],)
        assert (
            feature_cache.load_or_compute(
                "WR", df, df, None, drop_b, lambda: pytest.fail("same selection should hit")
            )
            == second
        )

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
