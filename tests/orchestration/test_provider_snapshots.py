import json
from dataclasses import asdict

import pandas as pd
import pytest

from src.data.providers import snapshot

pytestmark = pytest.mark.unit


def test_snapshot_record_rename_preserves_serialized_transport_contract():
    values = {
        "provider": "nflreadpy",
        "provider_version": "0.1.5",
        "loader": "weekly_data",
        "request": {"seasons": [2025]},
        "retrieved_at": "2026-09-10T12:00:00Z",
        "status": "observed",
        "content_digest": "a" * 64,
        "rows": 3,
        "error": None,
    }
    assert asdict(snapshot.ProviderSnapshotRecord(**values)) == values


@pytest.fixture(autouse=True)
def isolated_snapshot_state(monkeypatch):
    monkeypatch.delenv("FF_DATASET_ID", raising=False)
    monkeypatch.delenv("FF_CAPTURE_PROVIDER_SOURCES", raising=False)
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    monkeypatch.delenv("FF_DATA_FORMAT", raising=False)
    monkeypatch.setattr("src.config.CACHE_DIR", "data/raw")
    monkeypatch.setattr(snapshot, "_missing", set())


@pytest.mark.parametrize("empty", [False, True])
def test_capture_and_replay_preserve_observed_empty_distinction(tmp_path, monkeypatch, empty):
    calls = []

    @snapshot.snapshot_source
    def provider(seasons):
        calls.append(seasons)
        return pd.DataFrame({"season": [] if empty else seasons})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FF_CAPTURE_PROVIDER_SOURCES", "data/raw/provider_sources")
    expected = provider([2025])
    metadata = json.loads(next((tmp_path / "data/raw/provider_sources").glob("*.json")).read_text())
    assert metadata["status"] == ("empty" if empty else "observed")
    assert metadata["request"] == {"seasons": [2025]}
    assert metadata["retrieved_at"] and metadata["content_digest"]
    monkeypatch.delenv("FF_CAPTURE_PROVIDER_SOURCES")
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    pd.testing.assert_frame_equal(provider(seasons=[2025]), expected)
    assert len(calls) == 1


def test_missing_snapshot_never_fetches_and_swallowed_error_blocks_publication(
    tmp_path, monkeypatch
):
    calls = []

    @snapshot.snapshot_source
    def provider(seasons):
        calls.append(seasons)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    try:
        provider([2025])
    except Exception:
        pass  # emulate a legacy loader's fallback
    assert calls == []
    with pytest.raises(snapshot.SourceUnavailable, match="lacks required"):
        snapshot.assert_snapshot_sources_complete()


def test_unavailable_capture_has_explicit_status(tmp_path, monkeypatch):
    @snapshot.snapshot_source
    def provider():
        raise OSError("provider unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FF_CAPTURE_PROVIDER_SOURCES", str(tmp_path))
    with pytest.raises(OSError):
        provider()
    metadata = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert metadata["status"] == "unavailable"
    assert metadata["rows"] is None
    assert metadata["content_digest"] is None


def test_every_nfl_source_provider_entrypoint_is_snapshot_aware():
    import inspect

    from src.data import nfl_source

    public_sources = {
        name: function
        for name, function in inspect.getmembers(nfl_source, inspect.isfunction)
        if not name.startswith("_") and function.__module__ == nfl_source.__name__
    }
    assert {"pbp_data", "team_week_stats_release", "teams"}.issubset(public_sources)
    assert all(hasattr(function, "__wrapped__") for function in public_sources.values())


@pytest.mark.parametrize("override_raw_root", [False, True])
def test_replay_uses_selected_run_context_instead_of_checkout_data(
    tmp_path, monkeypatch, override_raw_root
):
    from src.training.context import RunContext, use_context

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    selected = tmp_path / "selected-data"
    raw_root = tmp_path / "explicit-raw" if override_raw_root else selected / "raw"
    context = RunContext(
        output_root=tmp_path / "outputs",
        data_root=selected,
        raw_root=raw_root if override_raw_root else None,
    )
    calls = []
    value = 1

    @snapshot.snapshot_source
    def provider(seasons):
        calls.append(value)
        return pd.DataFrame({"season": seasons, "value": value})

    for directory, captured_value in (
        (checkout / "data/raw/provider_sources", 1),
        (raw_root / "provider_sources", 2),
    ):
        value = captured_value
        monkeypatch.setenv("FF_CAPTURE_PROVIDER_SOURCES", str(directory))
        provider([2025])
    monkeypatch.delenv("FF_CAPTURE_PROVIDER_SOURCES")
    monkeypatch.setenv("FF_DATASET_ID", "b" * 64)
    with use_context(context):
        assert provider([2025])["value"].tolist() == [2]
    assert provider([2025])["value"].tolist() == [1]
    assert calls == [1, 2], "Replay must not invoke the live provider"
