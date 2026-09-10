"""Raw and split releases stay coherent across stale caches and failed S3 writes."""

import io
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.data import release

pytestmark = pytest.mark.unit


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_upload_suffix = None
        self.corrupt_download_suffix = None
        self.read_keys = []

    def upload_file(self, path, bucket, key):
        if self.fail_upload_suffix and key.endswith(self.fail_upload_suffix):
            raise OSError("interrupted upload")
        self.objects[key] = Path(path).read_bytes()

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[Key] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        self.read_keys.append(Key)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        content = self.objects[key]
        if self.corrupt_download_suffix and key.endswith(self.corrupt_download_suffix):
            content = b"corrupt"
        Path(path).write_bytes(content)


@pytest.fixture
def producer(tmp_path, monkeypatch):
    # Storage/atomicity fixtures intentionally contain no model-ready corpus;
    # real loader replay is exercised separately in test_release_loader_replay.
    monkeypatch.setattr(release, "verify_historical_loader_inputs", lambda *a, **k: None)
    raw, splits = tmp_path / "raw", tmp_path / "splits"
    raw.mkdir()
    splits.mkdir()
    for name in release.SPLIT_NAMES:
        pd.DataFrame({"season": [2024], "value": [1]}).to_parquet(splits / name)
    pd.DataFrame({"season": [2013, 2024], "value": [1, 2]}).to_parquet(raw / "weekly.parquet")
    pd.DataFrame({"season": [2013], "value": [1]}).to_parquet(raw / "snap_counts.parquet")
    (raw / "weekly.json").write_text('{"schema":2}')
    kwargs = dict(raw_dir=raw, splits_dir=splits, repo_root=tmp_path)
    release.seal_inputs(**kwargs)
    return kwargs


def test_upload_verifies_raw_splits_sidecars_and_records_coverage(producer):
    s3 = FakeS3()
    rid = release.publish_release(s3, "bucket", **producer)
    selected, manifest = release.resolve_release(s3, "bucket")
    assert selected == rid
    assert len(manifest["files"]) == 6
    assert manifest["coverage"]["raw/snap_counts.parquet"]["seasons_present"] == [2013]
    assert 2012 in manifest["coverage"]["raw/snap_counts.parquet"]["global_seasons_absent"]
    assert f"data/releases/{rid}/raw/weekly.json" in s3.objects
    assert "data/train.parquet" not in s3.objects


def test_stat_correction_replaces_warm_raw_and_split_caches_together(producer, tmp_path):
    """A corrected historical value must reach both live history and split readers."""
    s3 = FakeS3()
    raw, splits = producer["raw_dir"], producer["splits_dir"]
    consumer = dict(raw_dir=tmp_path / "consumer/raw", splits_dir=tmp_path / "consumer/splits")
    releases = []
    for yards in (-2, 3):
        corrected = pd.DataFrame(
            {"player_id": ["00-0039918"], "season": [2025], "week": [6], "rushing_yards": [yards]}
        )
        corrected.to_parquet(raw / "weekly.parquet")
        for name in release.SPLIT_NAMES:
            corrected.to_parquet(splits / name)
        release.seal_inputs(**producer)
        releases.append(release.publish_release(s3, "bucket", **producer))
        result = release.download_release(s3, "bucket", **consumer)
        assert result["release_id"] == releases[-1]
        assert pd.read_parquet(consumer["raw_dir"] / "weekly.parquet").rushing_yards.tolist() == [
            yards
        ]
        for name in release.SPLIT_NAMES:
            assert pd.read_parquet(consumer["splits_dir"] / name).rushing_yards.tolist() == [yards]
    assert releases[0] != releases[1]
    # A pinned historical run remains reproducible after correction publication.
    assert release.resolve_release(s3, "bucket", release_id=releases[0])[0] == releases[0]


def test_failed_upload_leaves_old_pointer_untouched(producer):
    s3 = FakeS3()
    old = release.publish_release(s3, "bucket", **producer)
    pd.DataFrame({"season": [2025]}).to_parquet(producer["raw_dir"] / "weekly.parquet")
    release.seal_inputs(**producer)
    s3.fail_upload_suffix = "val.parquet"
    with pytest.raises(OSError, match="interrupted"):
        release.publish_release(s3, "bucket", **producer)
    assert release.resolve_release(s3, "bucket")[0] == old


def test_cannot_publish_unsealed_or_mixed_generation(producer):
    (producer["raw_dir"] / "weekly.json").write_text('{"schema":3}')
    with pytest.raises(RuntimeError, match="changed after build"):
        release.publish_release(FakeS3(), "bucket", **producer)
    (producer["splits_dir"] / release.SEAL_NAME).unlink()
    with pytest.raises(RuntimeError, match="Unsealed"):
        release.publish_release(FakeS3(), "bucket", **producer)


def test_producer_change_rejects_old_seal(producer):
    source = producer["repo_root"] / "src/config.py"
    source.parent.mkdir()
    source.write_text("SEASONS = [2025]")
    with pytest.raises(RuntimeError, match="producer changed"):
        release.publish_release(FakeS3(), "bucket", **producer)


def test_warm_cache_is_checked_and_failed_download_installs_nothing(producer, tmp_path):
    s3 = FakeS3()
    rid = release.publish_release(s3, "bucket", **producer)
    raw, splits = tmp_path / "consumer-raw", tmp_path / "consumer-splits"
    raw.mkdir()
    splits.mkdir()
    (raw / "weekly.parquet").write_bytes(b"old raw generation")
    (splits / "train.parquet").write_bytes(b"old split generation")
    s3.corrupt_download_suffix = "val.parquet"
    with pytest.raises(ValueError, match="checksum"):
        release.download_release(s3, "bucket", raw_dir=raw, splits_dir=splits)
    assert (raw / "weekly.parquet").read_bytes() == b"old raw generation"
    assert (splits / "train.parquet").read_bytes() == b"old split generation"
    s3.corrupt_download_suffix = None
    result = release.download_release(s3, "bucket", raw_dir=raw, splits_dir=splits)
    assert result["release_id"] == rid
    assert (raw / "weekly.parquet").read_bytes() == (
        producer["raw_dir"] / "weekly.parquet"
    ).read_bytes()
    assert (splits / "train.parquet").read_bytes() == (
        producer["splits_dir"] / "train.parquet"
    ).read_bytes()


def test_pinned_release_survives_concurrent_promotion(producer, tmp_path):
    s3 = FakeS3()
    old = release.publish_release(s3, "bucket", **producer)
    expected = (producer["raw_dir"] / "weekly.parquet").read_bytes()
    pd.DataFrame({"season": [2025], "value": [999]}).to_parquet(
        producer["raw_dir"] / "weekly.parquet"
    )
    release.seal_inputs(**producer)
    new = release.publish_release(s3, "bucket", **producer)
    assert old != new
    out = tmp_path / "consumer"
    result = release.download_release(
        s3, "bucket", raw_dir=out / "raw", splits_dir=out / "splits", release_id=old
    )
    assert result["release_id"] == old
    assert (out / "raw/weekly.parquet").read_bytes() == expected


def test_missing_manifest_fails_instead_of_using_unversioned_data():
    s3 = FakeS3()
    s3.objects["data/train.parquet"] = b"legacy"
    with pytest.raises(ClientError):
        release.resolve_release(s3, "bucket")


def test_manifest_corruption_rejected(producer):
    s3 = FakeS3()
    rid = release.publish_release(s3, "bucket", **producer)
    s3.objects[f"data/releases/{rid}/manifest.json"] += b" "
    with pytest.raises(ValueError, match="manifest checksum"):
        release.resolve_release(s3, "bucket")


def test_split_jobs_receive_same_release(monkeypatch):
    from src.batch import launch

    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    monkeypatch.setattr(launch, "JOB_DEFINITION_CPU", "cpu")
    monkeypatch.setattr(launch, "JOB_QUEUE_CPU", "cpu-q")
    batch = mock.Mock()
    batch.submit_job.side_effect = [{"jobId": f"job-{i}"} for i in range(3)]
    launch._submit_split_for_position("WR", 42, "run", batch)
    for call in batch.submit_job.call_args_list:
        env = {e["name"]: e["value"] for e in call.kwargs["containerOverrides"]["environment"]}
        assert env["FF_DATA_RELEASE"] == "a" * 64


def test_batch_raw_and_splits_pin_once(producer, monkeypatch, tmp_path):
    from src import config
    from src.batch import train

    # The bootstrap mutates the environment; setenv records cleanup for absent keys too.
    monkeypatch.setenv("FF_DATA_RELEASE", "")
    s3 = FakeS3()
    rid = release.publish_release(s3, "bucket", **producer)
    monkeypatch.setattr(train.boto3, "client", lambda *a, **k: s3)
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "local/raw"))
    monkeypatch.setenv("TRAINING_DATA_DIR", str(tmp_path / "local/splits"))
    train.sync_raw_data("bucket")
    train.download_data("bucket", "data", str(tmp_path / "local/splits"))
    assert s3.read_keys.count("data/manifest.json") == 1
    assert json.loads((tmp_path / "local/raw/.release.json").read_text())["release_id"] == rid


@pytest.mark.parametrize("module", ["src.tuning.tune_nn", "src.tuning.tune_lgbm"])
def test_tuning_checks_existing_cache_instead_of_skipping(module, monkeypatch):
    import importlib

    from src.batch import train

    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    download = mock.Mock()
    monkeypatch.setattr(train, "download_data", download)
    mod = importlib.import_module(module)
    mod._ensure_data_from_s3()
    download.assert_called_once_with("bucket", "data", mod.SPLITS_DIR)


def test_serving_hydrates_snapshot_before_exposing_data(producer, monkeypatch, tmp_path):
    import boto3

    from src.shared import model_sync

    s3 = FakeS3()
    rid = release.publish_release(s3, "bucket", **producer)
    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "bucket")
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    monkeypatch.setattr(model_sync, "_repo_root", lambda: tmp_path / "consumer")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: s3)
    assert model_sync.sync_data_from_s3()["release_id"] == rid


def test_remote_content_is_verified_before_promotion(producer):
    class CorruptS3(FakeS3):
        def upload_file(self, path, bucket, key):
            super().upload_file(path, bucket, key)
            if key.endswith("weekly.parquet"):
                self.objects[key] = b"bad transfer"

    s3 = CorruptS3()
    with pytest.raises(RuntimeError, match="Uploaded training input failed checksum"):
        release.publish_release(s3, "bucket", **producer)
    assert "data/manifest.json" not in s3.objects


def test_split_merge_rejects_data_from_another_release(monkeypatch, tmp_path):
    from src.batch import train

    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    monkeypatch.setattr(train, "_load_split_manifest", lambda *a: {"data_release": "b" * 64})
    with pytest.raises(RuntimeError, match="split data release mismatch"):
        train._download_split_branch_artifacts(
            mock.Mock(), "bucket", "run", "WR", "nn", "", str(tmp_path)
        )


def test_metrics_preserve_release_provenance(monkeypatch):
    from src.batch import train

    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    result = {"cohorts": {"available": True}}
    assert train._extract_metrics("WR", result)["data_release"] == "a" * 64


def test_pinned_cache_miss_cannot_mix_new_source_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    with pytest.raises(RuntimeError, match="missing or incompatible"):
        release.assert_source_fetch_allowed(tmp_path / "weekly.parquet")
    monkeypatch.delenv("FF_DATA_RELEASE")
    (tmp_path / ".release.json").write_text('{"release_id":"recorded"}')
    with pytest.raises(RuntimeError, match="sealed snapshot"):
        release.assert_source_fetch_allowed(tmp_path / "weekly.parquet")
    monkeypatch.setenv("FF_DATA_RELEASE", "legacy")
    release.assert_source_fetch_allowed(tmp_path / "weekly.parquet")


def test_clean_producer_can_fetch_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("FF_DATA_RELEASE", raising=False)
    release.assert_source_fetch_allowed(tmp_path / "missing.parquet")


def test_live_builder_uses_separate_cache_without_unpinning_history(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    raw, live = tmp_path / "historical", tmp_path / "live"
    raw.mkdir()
    (raw / ".release.json").write_text('{"release_id":"recorded"}')
    with release.live_source_cache(live):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(release.assert_source_fetch_allowed, live / "weekly.parquet").result()
        with pytest.raises(release.DataReleaseError):
            release.assert_source_fetch_allowed(raw / "weekly.parquet")
    with pytest.raises(release.DataReleaseError):
        release.assert_source_fetch_allowed(live / "weekly.parquet")
    with pytest.raises(release.DataReleaseError, match="separate"):
        with release.live_source_cache(raw):
            pass


@pytest.mark.parametrize("kind", ["tune", "ab", "ablate", "scheduler"])
def test_operator_launchers_forward_the_selected_release(kind, monkeypatch):
    from src.tuning import launch_ab, launch_ablate, launch_ablate_scheduler, launch_tune

    monkeypatch.setenv("FF_DATA_RELEASE", "c" * 64)
    batch = mock.Mock()
    batch.submit_job.return_value = {"jobId": "job"}
    common = dict(
        run_id="run",
        s3_prefix="experiment",
        job_definition="job:1",
        image_sha="image",
        seeds=[42],
        only=None,
        cuda_graph="auto",
        batch_client=batch,
    )
    if kind == "tune":
        launch_tune.submit_tune_job("WR", n_trials=1, batch_client=batch)
    elif kind == "ab":
        launch_ab.submit_ab_job("WR", spec_dotted="example.spec", **common)
    elif kind == "ablate":
        launch_ablate.submit_ablate_job("WR", mod_dotted="example.spec", **common)
    else:
        launch_ablate_scheduler.submit_ablate_job("WR", seeds="42", batch_client=batch)
    env = {
        item["name"]: item["value"]
        for item in batch.submit_job.call_args.kwargs["containerOverrides"]["environment"]
    }
    assert env["FF_DATA_RELEASE"] == "c" * 64


@pytest.mark.parametrize("omit_position", [None, "K"])
def test_release_prewarms_and_requires_the_shared_reference(producer, monkeypatch, omit_position):
    from types import SimpleNamespace

    from src import config
    from src.data import dst_scoring, identity, loader
    from src.dst import data as defense_data
    from src.k import config as kicker_config
    from src.k import data as kicker_data
    from src.scripts import build_evaluation_reference
    from src.shared.evaluation_cohorts import REFERENCE_FILENAME

    monkeypatch.setattr(defense_data, "build_data", lambda **kwargs: None)
    raw = producer["raw_dir"]
    pd.DataFrame({"player_id": ["one"], "name": ["Player"]}).to_parquet(
        raw / "player_metadata_v1.parquet"
    )
    monkeypatch.setattr(config, "CACHE_DIR", str(raw))
    monkeypatch.setattr(config, "SEASONS", [2024, 2025])
    monkeypatch.setattr(config, "TEST_SEASONS", [2025])
    monkeypatch.setattr(kicker_config, "POSITION_CONFIG", SimpleNamespace(seasons=[2024, 2025]))

    def materialize(name):
        pd.DataFrame({"season": [2025]}).to_parquet(raw / name)

    monkeypatch.setattr(
        identity, "load_player_id_bridge", lambda d: materialize("player_id_bridge_v2.parquet")
    )
    monkeypatch.setattr(
        loader, "load_team_week_stats", lambda s: materialize("team_stats_2024_2025.parquet")
    )
    monkeypatch.setattr(
        dst_scoring,
        "load_dst_scoring_events",
        lambda s: materialize("dst_scoring_pbp_v1_2024_2025.parquet"),
    )
    monkeypatch.setattr(
        kicker_data,
        "reconstruct_kicker_weekly_from_pbp",
        lambda s: materialize("kicker_pbp_2024_2024.parquet"),
    )
    monkeypatch.setattr(
        kicker_data,
        "reconstruct_kicker_kicks_from_pbp",
        lambda s: materialize("kicker_kicks_pbp_2024_2025.parquet"),
    )
    monkeypatch.setattr(
        kicker_data, "load_data", lambda: materialize("kicker_backfill_pbp_v1_2025.parquet")
    )

    def reference(seasons, *, upload):
        assert seasons == [2025] and upload is False
        positions = [p for p in ("QB", "RB", "WR", "TE", "K", "DST") if p != omit_position]
        frame = pd.DataFrame({"position": positions, "season": [2025] * len(positions)})
        frame.to_parquet(raw / REFERENCE_FILENAME)
        return frame

    monkeypatch.setattr(build_evaluation_reference, "write_reference", reference)
    if omit_position:
        with pytest.raises(release.DataReleaseError, match="Evaluation reference unavailable.*K"):
            release.prewarm_training_dependencies()
    else:
        release.prewarm_training_dependencies()
        manifest = release.seal_inputs(**producer)
        assert "raw/player_id_bridge_v2.parquet" in manifest["files"]
        assert "raw/player_metadata_v1.parquet" in manifest["files"]
        assert f"raw/{REFERENCE_FILENAME}" in manifest["files"]
        assert manifest["coverage"][f"raw/{REFERENCE_FILENAME}"]["rows"] == 6


def test_compatible_recipe_stays_reachable_after_other_source_promotion(producer):
    source = producer["repo_root"] / "src/config.py"
    source.parent.mkdir()
    source.write_text("SEASONS = [2024]\n")
    release.seal_inputs(**producer)
    first_recipe = release.data_producer_hashes(producer["repo_root"])
    s3 = FakeS3()
    first = release.publish_release(s3, "bucket", **producer)
    source.write_text("SEASONS = [2025]\n")
    release.seal_inputs(**producer)
    second = release.publish_release(s3, "bucket", **producer)
    assert first != second
    assert release.resolve_compatible_release(s3, "bucket", first_recipe)[0] == first


def test_hydration_quarantines_unlisted_cache_hits_preserving_unrelated_files(
    producer, monkeypatch, tmp_path
):
    from src.data import identity

    source = producer["raw_dir"] / "player_metadata_v1.parquet"
    pd.DataFrame({"gsis_id": ["stale-A"]}).to_parquet(source)
    release.seal_inputs(**producer)
    s3 = FakeS3()
    first = release.publish_release(s3, "bucket", **producer)
    raw, splits = tmp_path / "consumer/raw", tmp_path / "consumer/splits"
    release.download_release(s3, "bucket", raw_dir=raw, splits_dir=splits, release_id=first)
    for name in ("kicker_notes.md", "player_metadata_notes.txt", "research.parquet"):
        (raw / name).write_text("user-owned")
    source.unlink()
    release.seal_inputs(**producer)
    second = release.publish_release(s3, "bucket", **producer)
    release.download_release(s3, "bucket", raw_dir=raw, splits_dir=splits, release_id=second)
    assert not (raw / source.name).exists()
    assert (raw / ".quarantine" / second / source.name).is_file()
    for name in ("kicker_notes.md", "player_metadata_notes.txt", "research.parquet"):
        assert (raw / name).read_text() == "user-owned"
    monkeypatch.setenv("FF_DATA_RELEASE", second)
    monkeypatch.setattr(
        identity.nfl_source, "player_metadata", lambda: pytest.fail("unrecorded fetch")
    )
    with pytest.raises(release.DataReleaseError):
        identity.load_player_metadata(str(raw))


def test_prewarm_rejects_incomplete_completed_dst_games_before_seal(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "verify_historical_loader_inputs", lambda *a, **k: None)
    from src import config
    from src.data import dst_scoring, identity, loader
    from src.dst import data as defense
    from tests.dst.test_data_build import _make_schedules, _make_team_stats, _make_weekly
    from tests.dst.test_scoring_events import _plays

    incomplete = dst_scoring.aggregate_dst_scoring_events(_plays({}))
    actual_build = defense.build_data
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(identity, "load_player_id_bridge", lambda *_: None)
    monkeypatch.setattr(loader, "load_team_week_stats", lambda *_: _make_team_stats())
    monkeypatch.setattr(dst_scoring, "load_dst_scoring_events", lambda *_: incomplete)

    def validate(*, scoring_events, allow_scoring_fetch):
        assert allow_scoring_fetch is False
        return actual_build(
            weekly=_make_weekly(),
            schedules=_make_schedules(),
            team_stats=_make_team_stats(),
            scoring_events=scoring_events,
        )

    monkeypatch.setattr(defense, "build_data", validate)
    with pytest.raises(ValueError, match="missing completed team-weeks"):
        release.prewarm_training_dependencies()
    assert not (tmp_path / release.SEAL_NAME).exists()


def test_unpinned_local_launcher_uses_its_recipe_after_global_current_changes(
    producer, monkeypatch
):
    from src.batch import launch

    monkeypatch.setenv("FF_DATA_RELEASE", "")
    source = producer["repo_root"] / "src/config.py"
    source.parent.mkdir()
    source.write_text("SEASONS = [2024]\n")
    release.seal_inputs(**producer)
    local_recipe = release.data_producer_hashes(producer["repo_root"])
    s3 = FakeS3()
    first = release.publish_release(s3, "bucket", **producer)
    source.write_text("SEASONS = [2025]\n")
    release.seal_inputs(**producer)
    second = release.publish_release(s3, "bucket", **producer)
    assert first != second
    monkeypatch.setattr(release, "data_producer_hashes", lambda root: local_recipe)
    assert launch.pin_data_release(s3) == first
    # An explicit pin means a deliberately selected remote image; don't replace it
    # with the local checkout's recipe when those two source revisions differ.
    monkeypatch.setenv("FF_DATA_RELEASE", second)
    assert launch.pin_data_release(s3) == second
