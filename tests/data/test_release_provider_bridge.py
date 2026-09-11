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


@pytest.mark.parametrize("entrypoint", ["download", "materialize"])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    "relative",
    [
        "provider_sources",
        ".quarantine",
        ".quarantine/{release}",
        ".quarantine/{release}/provider_sources",
    ],
)
def test_hydration_rejects_nested_symlinks_before_changing_any_files(
    producer, tmp_path, monkeypatch, entrypoint, captured, relative
):
    from src.orchestration.datasets import DatasetError, materialize_dataset

    if captured:

        @snapshot.snapshot_source
        def provider():
            return pd.DataFrame({"value": [1]})

        with snapshot.capture_provider_sources(producer["raw_dir"] / "provider_sources"):
            provider()
        release.seal_inputs(**producer)
    s3 = FakeS3()
    selected = release.publish_release(s3, "bucket", **producer)
    raw, splits = tmp_path / "consumer/raw", tmp_path / "consumer/splits"
    raw.mkdir(parents=True)
    splits.mkdir()
    (raw / "weekly.parquet").write_bytes(b"prior raw")
    (splits / "train.parquet").write_bytes(b"prior split")
    external = tmp_path / "shared-provider-cache"
    external.mkdir()
    (external / ("f" * 64 + ".json")).write_bytes(b"unrelated shared capture")
    nested = raw / relative.format(release=selected)
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.symlink_to(external, target_is_directory=True)
    before = {p.name: p.read_bytes() for p in external.iterdir()}
    monkeypatch.setattr(
        s3, "download_file", lambda *args: pytest.fail("Preflight must reject first")
    )
    with pytest.raises((ValueError, DatasetError), match="symlinked release directory"):
        if entrypoint == "download":
            release.download_release(
                s3, "bucket", raw_dir=raw, splits_dir=splits, release_id=selected
            )
        else:
            materialize_dataset(s3, "bucket", selected, raw_dir=raw, splits_dir=splits)
    assert {p.name: p.read_bytes() for p in external.iterdir()} == before
    assert (raw / "weekly.parquet").read_bytes() == b"prior raw"
    assert (splits / "train.parquet").read_bytes() == b"prior split"
    assert not (raw / ".release.json").exists()


def test_direct_download_preserves_existing_root_directories_and_explicit_root_alias(
    producer, tmp_path
):
    s3 = FakeS3()
    selected = release.publish_release(s3, "bucket", **producer)
    raw, splits = tmp_path / "mounted/raw", tmp_path / "mounted/splits"
    raw.mkdir(parents=True)
    splits.mkdir()
    inodes = (raw.stat().st_ino, splits.stat().st_ino)
    alias = tmp_path / "selected-raw"
    alias.symlink_to(raw, target_is_directory=True)
    release.download_release(s3, "bucket", raw_dir=alias, splits_dir=splits, release_id=selected)
    assert (raw.stat().st_ino, splits.stat().st_ino) == inodes
    assert alias.is_symlink()
    assert (raw / "weekly.parquet").read_bytes() == (
        producer["raw_dir"] / "weekly.parquet"
    ).read_bytes()
    assert json.loads((raw / ".release.json").read_text())["release_id"] == selected


def test_hydration_rechecks_nested_destinations_after_remote_downloads(
    producer, tmp_path, monkeypatch
):
    import threading

    s3 = FakeS3()
    selected = release.publish_release(s3, "bucket", **producer)
    raw, splits = tmp_path / "consumer/raw", tmp_path / "consumer/splits"
    raw.mkdir(parents=True)
    (raw / "weekly.parquet").write_bytes(b"prior raw")
    external = tmp_path / "external"
    external.mkdir()
    captured = external / ("f" * 64 + ".json")
    captured.write_bytes(b"shared capture")
    download = s3.download_file
    lock = threading.Lock()

    def changed_directory(*args):
        with lock:
            if not (raw / "provider_sources").is_symlink():
                (raw / "provider_sources").symlink_to(external, target_is_directory=True)
        download(*args)

    monkeypatch.setattr(s3, "download_file", changed_directory)
    with pytest.raises(ValueError, match="symlinked release directory"):
        release.download_release(s3, "bucket", raw_dir=raw, splits_dir=splits, release_id=selected)
    assert captured.read_bytes() == b"shared capture"
    assert (raw / "weekly.parquet").read_bytes() == b"prior raw"
    assert not splits.exists()


def test_dst_scoring_workers_capture_and_replay_contextual_provider_responses(
    monkeypatch, tmp_path
):
    from src.data import dst_scoring, nfl_source

    calls = []

    @snapshot.snapshot_source
    def pbp_data(seasons, cols):
        calls.append(seasons)
        return pd.DataFrame(
            [
                {
                    **dict.fromkeys(dst_scoring.PBP_COLUMNS, 0),
                    "season": seasons[0],
                    "season_type": "REG",
                    "week": 1,
                    "game_id": str(seasons[0]),
                    "play_id": 1,
                    "home_team": "BUF",
                    "away_team": "NYJ",
                    "posteam": "BUF",
                    "defteam": "NYJ",
                    "td_team": None,
                    "play_type": "run",
                }
            ]
        )

    monkeypatch.setattr(nfl_source, "pbp_data", pbp_data)
    raw = tmp_path / "data/raw"
    captured = raw / "provider_sources"
    with use_context(RunContext(tmp_path / "outputs", tmp_path / "data")):
        with snapshot.capture_provider_sources(captured):
            expected = dst_scoring.load_dst_scoring_events([2024, 2025], raw)
        assert len(list(captured.glob("*.json"))) == 2
        assert len(list(captured.glob("*.parquet"))) == 2
        for metadata in captured.glob("*.json"):
            assert json.loads(metadata.read_text())["loader"] == "pbp_data"
        next(raw.glob("dst_scoring_*.parquet")).unlink()
        monkeypatch.setenv("FF_DATASET_ID", "a" * 64)
        replayed = dst_scoring.load_dst_scoring_events([2024, 2025], raw)
        snapshot.assert_snapshot_sources_complete()
    pd.testing.assert_frame_equal(replayed, expected)
    assert sorted(calls) == [[2024], [2025]]


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
