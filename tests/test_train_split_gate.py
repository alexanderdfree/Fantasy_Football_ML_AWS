"""Both training backends require compatible canonical data before compute."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

pytestmark = pytest.mark.unit


def _load(name: str) -> dict:
    with (WORKFLOWS / name).open() as f:
        return yaml.safe_load(f)


def _job_steps(doc: dict, job: str) -> list[dict]:
    return list(doc.get("jobs", {}).get(job, {}).get("steps", []))


@pytest.mark.parametrize(
    "wf,job,train_step",
    [
        ("train-batch.yml", "train", "Submit Batch jobs and wait"),
        ("train-ec2.yml", "train", "Run training for all positions (sequential)"),
    ],
)
def test_train_waits_for_fresh_splits(wf, job, train_step):
    """Both workflows retain their documented input selection gate before training."""
    doc = _load(wf)
    steps = _job_steps(doc, job)
    if wf == "train-batch.yml":
        gate = next((step for step in steps if step.get("id") == "build_plan"), None)
        assert gate is not None, "Batch must select an immutable dataset/build plan"
        training = next(step for step in steps if step.get("name") == train_step)
        assert steps.index(gate) < steps.index(training)
        assert "if" not in gate, "Manual dispatch must also resolve an identified dataset"
        assert not gate.get("continue-on-error", False)
        run_body = gate["run"]
        assert "set -euo pipefail" in run_body
        assert "python -m src.orchestration.build_plan" in run_body
        assert "--timeout" in run_body
        assert "|| true" not in run_body
        assert "splits-rebuild-markers" not in run_body
        environment = training["env"]
        assert environment["FF_REQUIRE_BUILD_PLAN"] == "1"
        for key, output in (
            ("FF_BUILD_PLAN_ID", "build_plan_id"),
            ("FF_DATASET_ID", "dataset_id"),
            ("FF_DATA_RELEASE", "dataset_id"),
            ("FF_TRAIN_GIT_SHA", "git_sha"),
        ):
            assert environment[key] == "${{ steps.build_plan.outputs." + output + " }}"
        return

    gate = next(step for step in steps if step.get("id") == "data-release")
    training = next(step for step in steps if step.get("name") == train_step)
    start = next(
        step for step in steps if step.get("name") == "Start instance (no-op if already running)"
    )
    assert steps.index(gate) < steps.index(start) < steps.index(training)
    assert "if" not in gate and not gate.get("continue-on-error", False)
    assert gate["env"]["HEAD_SHA"] == "${{ steps.image.outputs.image_sha }}"
    assert "src.scripts.wait_data_release" in gate["run"]
    assert "--pin-training" in gate["run"]
    assert "|| true" not in gate["run"]


@pytest.mark.parametrize("legacy_marker", [None, "pending", "ready"])
def test_batch_selection_timeout_never_accepts_legacy_markers_or_mutable_splits(legacy_marker):
    from src.orchestration.datasets import DatasetError, select_dataset
    from tests.orchestration.test_datasets import MemoryS3

    class TrackedS3(MemoryS3):
        def __init__(self):
            super().__init__()
            self.reads = []

        def get_object(self, Bucket, Key):  # noqa: N803
            self.reads.append(Key)
            return super().get_object(Bucket, Key)

    s3 = TrackedS3()
    source_id = "a" * 64
    s3.objects["data/train.parquet"] = b"old mutable splits"
    if legacy_marker:
        s3.objects[f"splits-rebuild-markers/{legacy_marker}/image-sha.txt"] = b"old marker"
    now = [0]

    def sleep(duration):
        now[0] += duration

    with pytest.raises(DatasetError, match="fallback is disabled"):
        select_dataset(
            s3, "bucket", source_id, timeout=2, poll=1, clock=lambda: now[0], sleep=sleep
        )
    assert now[0] == 2
    assert s3.reads == [f"data/by-producer/{source_id}/manifest.json"] * 3
