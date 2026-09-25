"""No-fit checks for historical-input identity and intervention isolation."""

import hashlib
import io
import json

import pandas as pd
import pytest

from src.analysis.wr_pr1564_inputs import (
    RECIPE,
    apply_depth,
    build,
    historical_numerical_files,
    selected_inventory,
    validate_numerical_source,
    verify_recovered_inputs,
)
from src.analysis.wr_pr1564_recovery import recover_object
from src.analysis.wr_pr1564_report import load, shapley
from src.tuning.ab_wr_pr1564 import _activate, _immutable_put, _metrics

pytestmark = pytest.mark.unit


class Source:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def get_object(self, **kwargs):
        self.requests.append(kwargs)
        return {"Body": io.BytesIO(self.content), "VersionId": "archived-version"}


def test_recovery_pins_version_and_rejects_changed_archive(tmp_path):
    original = b"archived input"
    task = {
        "key": "data/train.parquet",
        "version": "old-version",
        "sha256": hashlib.sha256(original).hexdigest(),
        "bytes": len(original),
        "destination": "baseline/train.parquet",
    }
    source = Source(b"newer bytes")
    record = recover_object(source, "bucket", task, tmp_path)
    assert not record["ok"]
    assert source.requests[0]["VersionId"] == "old-version"
    assert not (tmp_path / task["destination"]).exists()
    source.content = original
    record = recover_object(source, "bucket", task, tmp_path)
    assert record["ok"]
    assert (tmp_path / task["destination"]).read_bytes() == original


def test_depth_intervention_does_not_change_training_or_other_positions():
    frame = pd.DataFrame(
        {
            "player_id": ["a", "a", "b", "c"],
            "season": [2024, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "position": ["WR", "WR", "QB", "WR"],
            "depth_chart_rank": [2.0, 2.0, 2.0, 2.0],
            "is_top_available": [1.0, 0.0, 0.0, 1.0],
        }
    )
    lookup = pd.Series([1.0], index=pd.MultiIndex.from_tuples([("a", 2025, 1)]))
    result = apply_depth(frame, lookup)
    assert result.depth_chart_rank.tolist() == [2.0, 1.0, 2.0, -1.0]
    pd.testing.assert_frame_equal(
        result.drop(columns="depth_chart_rank"), frame.drop(columns="depth_chart_rank")
    )
    assert frame.depth_chart_rank.eq(2).all()


def test_inventory_includes_key_features_and_added_rows():
    left = pd.DataFrame(
        {"player_id": ["a"], "season": [2025], "week": [1], "depth_chart_rank": [3.0]}
    )
    right = pd.concat(
        [left.assign(depth_chart_rank=1), left.assign(player_id="b")], ignore_index=True
    )
    result = selected_inventory(left, right, ["week", "depth_chart_rank"])
    assert result["changed"] == {"depth_chart_rank": 1}
    assert result["added"] == [["b", 2025, 1]]


def test_metrics_keep_fixed_truth_and_subset():
    frame = pd.DataFrame({"actual": [0.0, 10.0, 1000.0]})
    for family in ("ridge", "nn", "attn_nn", "lgbm"):
        frame[f"pred_{family}_total"] = [2.0, 8.0, 0.0]
    block = _metrics(frame, pd.Series([True, True, False]))
    assert block["n"] == 2
    assert all(
        values == {"mae": 2.0, "rmse": 2.0, "bias": 0.0} for values in block["models"].values()
    )


def test_existing_evidence_cannot_be_overwritten():
    class Conflict(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class Existing(Source):
        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"
            raise Conflict()

    source = Existing(b"saved")
    _immutable_put(source, "bucket", "key", b"saved")
    with pytest.raises(ValueError, match="Refusing to replace"):
        _immutable_put(source, "bucket", "key", b"different")


def test_attribution_allocates_interactions_and_reconciles():
    import itertools

    values = {
        (r, a, d): 3 * r + 2 * a + 4 * d + 6 * r * d
        for r, a, d in itertools.product((0, 1), repeat=3)
    }
    assert shapley(values) == {"source": 6.0, "availability": 2.0, "depth": 7.0}


def test_incomplete_experiment_cannot_produce_a_conclusion(tmp_path):
    with pytest.raises(ValueError, match="Incomplete factorial"):
        load(tmp_path)


def test_preparation_and_experiment_reject_local_fitting_before_io(monkeypatch, tmp_path):
    monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="require AWS Batch"):
        build(tmp_path / "missing", tmp_path / "output")
    with pytest.raises(ValueError, match="require AWS Batch"):
        _activate("r0a0")({})
    assert not (tmp_path / "output").exists()


def test_historical_numerical_source_is_verified_without_fitting(tmp_path):
    path = tmp_path / "recipe.py"
    path.write_bytes(b"historical recipe")
    manifest = {
        "recipe": RECIPE,
        "numerical_files": {"recipe.py": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    validate_numerical_source(manifest, tmp_path)
    path.write_bytes(b"new defaults")
    with pytest.raises(ValueError, match="Historical numerical source mismatch"):
        validate_numerical_source(manifest, tmp_path)
    with pytest.raises(ValueError, match="Missing historical numerical-source"):
        validate_numerical_source({"recipe": RECIPE}, tmp_path)


def test_changed_or_extra_recovered_inputs_cannot_be_relabelled(monkeypatch, tmp_path):
    import src.analysis.wr_pr1564_inputs as inputs

    release = b'{"files": {}}'
    monkeypatch.setattr(inputs, "FIXED_RELEASE", hashlib.sha256(release).hexdigest())
    (tmp_path / "fixed-release-manifest.json").write_bytes(release)
    name = "baseline/data/splits/train.parquet"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"verified input")
    record = {
        "destination": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "ok": True,
    }
    receipt = {
        "fixed_release": inputs.FIXED_RELEASE,
        "records": [record],
        "recovered": 1,
        "missing": [],
    }
    receipt_bytes = json.dumps(receipt).encode()
    (tmp_path / "recovery.json").write_bytes(receipt_bytes)
    assert verify_recovered_inputs(tmp_path) == hashlib.sha256(receipt_bytes).hexdigest()
    path.write_bytes(b"changed target")
    with pytest.raises(ValueError, match="changed after verification"):
        verify_recovered_inputs(tmp_path)
    path.write_bytes(b"verified input")
    (path.parent / "extra.parquet").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="Unrecorded"):
        verify_recovered_inputs(tmp_path)


def test_historical_manifest_records_verified_git_recipe(monkeypatch, tmp_path):
    import src.analysis.wr_pr1564_inputs as inputs

    name = "src/wr/config.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"historical config")

    def git(command, **kwargs):
        if command[1] == "ls-tree":
            return name + "\nsrc/analysis/diagnostic.py\n"
        assert command == ["git", "show", f"{RECIPE}:{name}"]
        return b"historical config"

    monkeypatch.setattr(inputs.subprocess, "check_output", git)
    assert historical_numerical_files(tmp_path) == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
    }


def test_wrong_recipe_cannot_replace_a_callers_data_symlink(monkeypatch, tmp_path):
    import src.tuning.ab_wr_pr1564 as spec

    persistent = tmp_path / "persistent"
    persistent.mkdir()
    link = tmp_path / "data"
    link.symlink_to(persistent, target_is_directory=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "input-manifest.json").write_text(
        json.dumps(
            {
                "recipe": RECIPE,
                "numerical_files": {"missing-historical-file.py": "0" * 64},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "test-only")
    monkeypatch.setattr(spec, "_inputs", lambda: archive)
    with pytest.raises(ValueError, match="Historical numerical source mismatch"):
        spec._activate("r0a0")({})
    assert link.is_symlink() and link.resolve() == persistent
