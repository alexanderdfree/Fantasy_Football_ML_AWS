"""Provider transport and complete derived caches share one sealed release."""

import json

import pandas as pd
import pytest

from src.data import release
from src.data.providers import snapshot
from src.training.context import RunContext, use_context
from tests.test_data_release import FakeS3, producer  # noqa: F401
from tests.test_release_loader_replay import cached_inputs  # noqa: F401

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_source_mode(monkeypatch):
    for name in (
        "FF_DATA_RELEASE",
        "FF_DATASET_ID",
        "FF_DATA_FORMAT",
        "FF_CAPTURE_PROVIDER_SOURCES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(snapshot, "_missing", set())


def test_captured_provider_files_seal_hydrate_and_replay(producer, monkeypatch, tmp_path):
    calls = []

    @snapshot.snapshot_source
    def provider(seasons):
        calls.append(seasons)
        return pd.DataFrame({"season": seasons, "value": [7]})

    with snapshot.capture_provider_sources(producer["raw_dir"] / "provider_sources"):
        expected = provider([2025])
    manifest = release.seal_inputs(**producer)
    assert (
        len([name for name in manifest["files"] if name.startswith("raw/provider_sources/")]) == 2
    )
    s3 = FakeS3()
    release_id = release.publish_release(s3, "bucket", **producer)
    selected = tmp_path / "selected"
    release.download_release(
        s3,
        "bucket",
        raw_dir=selected / "raw",
        splits_dir=selected / "splits",
        release_id=release_id,
    )
    assert (
        json.loads((selected / "raw/.release.json").read_text())["provider_sources"] == "captured"
    )
    monkeypatch.setenv("FF_DATA_RELEASE", release_id)
    monkeypatch.setenv("FF_DATASET_ID", release_id)
    with use_context(RunContext(tmp_path / "outputs", selected)):
        pd.testing.assert_frame_equal(provider([2025]), expected)
        snapshot.assert_snapshot_sources_complete()
    assert calls == [[2025]]


def test_old_release_uses_derived_caches_without_provider_fetch(producer, monkeypatch, tmp_path):
    s3 = FakeS3()
    release_id = release.publish_release(s3, "bucket", **producer)
    selected = tmp_path / "selected"
    stale = selected / "raw/provider_sources"
    stale.mkdir(parents=True)
    (stale / ("f" * 64 + ".json")).write_text("{}")
    release.download_release(
        s3,
        "bucket",
        raw_dir=selected / "raw",
        splits_dir=selected / "splits",
        release_id=release_id,
    )
    monkeypatch.setenv("FF_DATA_RELEASE", release_id)
    monkeypatch.setenv("FF_DATASET_ID", release_id)

    @snapshot.snapshot_source
    def provider():
        pytest.fail("A derived-only historical release performed a provider fetch")

    with use_context(RunContext(tmp_path / "outputs", selected)):
        assert snapshot.provider_replay_mode() == "derived_only"
        assert len(pd.read_parquet(selected / "raw/weekly.parquet")) == 2
        with pytest.raises(release.DataReleaseError, match="derived caches only"):
            provider()
        with pytest.raises(release.DataReleaseError, match="missing or incompatible"):
            release.assert_source_fetch_allowed(selected / "raw/missing.parquet")
        snapshot.assert_snapshot_sources_complete()
    assert not list(stale.glob("*.json"))


def test_seal_rejects_corrupt_captured_provider_bytes(producer):
    @snapshot.snapshot_source
    def provider():
        return pd.DataFrame({"value": [1]})

    directory = producer["raw_dir"] / "provider_sources"
    with snapshot.capture_provider_sources(directory):
        provider()
    next(directory.glob("*.parquet")).write_bytes(b"corrupt")
    with pytest.raises(snapshot.SourceUnavailable, match="Invalid captured provider content"):
        release.seal_inputs(**producer)


def test_actual_loader_workers_inherit_live_provider_scope(cached_inputs, monkeypatch, tmp_path):
    from src.data import loader, nfl_source

    raw, _, _ = cached_inputs
    path = raw / "weekly_2012_2013.parquet"
    frame = pd.read_parquet(path)
    path.unlink()
    calls = []

    @snapshot.snapshot_source
    def weekly_data(seasons):
        calls.append(seasons)
        return frame.copy()

    monkeypatch.setattr(nfl_source, "weekly_data", weekly_data)
    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
    with use_context(RunContext(tmp_path / "outputs", tmp_path, raw_root=raw)):
        with release.live_source_cache(raw):
            result = loader.load_raw_data([2012, 2013], cache_dir=str(raw))
    assert len(result) == 2
    assert calls == [[2012, 2013]]
