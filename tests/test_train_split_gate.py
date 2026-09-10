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
