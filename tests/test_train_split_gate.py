"""Both training paths verify and pin published data before starting jobs.

A missing pending marker or a timeout cannot permit training against incompatible
inputs. Manifest/hash behavior is covered by scripts/test_wait_data_release.py;
these checks preserve its wiring in both the Batch and EC2 workflows.
"""

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


def _find_split_gate(steps: list[dict]) -> dict | None:
    for step in steps:
        if "src.scripts.wait_data_release" in (step.get("run", "") or ""):
            return step
    return None


@pytest.mark.parametrize(
    "wf,job,train_step",
    [
        ("train-batch.yml", "train", "Submit Batch jobs and wait"),
        ("train-ec2.yml", "train", "Run training for all positions (sequential)"),
    ],
)
def test_train_waits_for_fresh_splits(wf, job, train_step):
    """Every event verifies the image's producer contract, pins it, then trains."""
    steps = _job_steps(_load(wf), job)
    gate = _find_split_gate(steps)
    assert gate is not None, f"{wf}:{job} must verify published data before training"
    assert not gate.get("if"), "Manual dispatch must obey the same compatibility gate"
    assert not gate.get("continue-on-error"), "A failed data gate must block training"
    env = gate.get("env", {}) or {}
    assert ".outputs.image_sha" in env.get("HEAD_SHA", "")
    run_body = gate.get("run", "")
    assert '--revision "$HEAD_SHA"' in run_body
    assert "--timeout-seconds 3600" in run_body
    assert "--pin-training" in run_body
    assert "|| true" not in run_body
    assert gate.get("timeout-minutes", 0) > 60
    names = [step.get("name", "") for step in steps]
    assert train_step in names
    assert steps.index(gate) < names.index(train_step)


def test_deploy_prepares_verified_cache_from_the_same_pinned_data_before_rollout():
    steps = _job_steps(_load("deploy.yml"), "deploy")
    data_index = next(i for i, step in enumerate(steps) if step.get("id") == "data-release")
    cache_index = next(
        i
        for i, step in enumerate(steps)
        if "src.scripts.build_serving_cache" in step.get("run", "")
    )
    rollout_index = next(
        i
        for i, step in enumerate(steps)
        if "amazon-ecs-deploy-task-definition" in step.get("uses", "")
    )
    assert data_index < cache_index < rollout_index
    cache = steps[cache_index]
    assert cache["run"] == "python -m src.scripts.build_serving_cache --reuse-valid"
    assert cache["env"]["FF_DATA_RELEASE"] == "${{ steps.data-release.outputs.release_id }}"
    assert not steps[data_index].get("continue-on-error")
    assert not cache.get("continue-on-error")
    task = next(step for step in steps if step.get("name") == "Pull current task definition")
    assert task["env"]["DATA_RELEASE"] == cache["env"]["FF_DATA_RELEASE"]


def test_ec2_registers_the_resolved_source_before_remote_training():
    steps = _job_steps(_load("train-ec2.yml"), "train")
    training = next(step for step in steps if step.get("id") == "train")
    body = training["run"]
    assert training["env"]["FF_TRAIN_GIT_SHA"] == "${{ steps.image.outputs.image_sha }}"
    assert body.index('register_training_source --sha "$FF_TRAIN_GIT_SHA"') < body.index(
        "aws ssm send-command"
    )
    assert "FF_TRAIN_GIT_SHA='$FF_TRAIN_GIT_SHA'" in body
    assert "FF_TRAIN_IMAGE='$FF_TRAIN_IMAGE'" in body
    assert "FF_DATA_RELEASE='$DATA_RELEASE'" in body
