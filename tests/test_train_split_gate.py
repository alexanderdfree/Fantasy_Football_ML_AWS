"""Contract tests for the refresh-splits → train fresh-splits gate.

refresh-splits.yml (the S3 split producer) and the batch-image.yml →
train-batch.yml / train-ec2.yml consumer chain fire in parallel off the same
push, with no ordering between them — so a data-affecting merge could train on
the previous commit's splits (surfaced by PR #383; see TODO.md). PR #516 added a
fail-safe pending/ready S3-marker handshake: refresh-splits writes
``splits-rebuild-markers/pending/<sha>`` before the rebuild and ``.../ready/<sha>``
after the upload, and the train workflows poll the ready marker before training.

#516 gated only train-batch.yml and shipped no tests. This suite pins the wait
gate on BOTH train workflows — train-ec2.yml (the BATCH_ACTIVE=false rollback
path) shares the identical batch-image.yml→workflow_run trigger and S3 split
read, so it has the same race — so a future edit can't silently drop either
gate. The producer's pending/ready marker writes are pinned in
tests/test_refresh_splits_workflow.py.
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
    """The fresh-splits gate head-objects the splits-rebuild markers. Anchored
    on BOTH 'splits-rebuild-markers' and 'head-object' so it's not confused with
    the model-artifact freshness step (which head-objects models/<pos>/manifest.json)."""
    for step in steps:
        run_body = step.get("run", "") or ""
        if "splits-rebuild-markers" in run_body and "head-object" in run_body:
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
    """Both train workflows must gate on refresh-splits' fresh-split marker
    before training: (a) head-object the splits-rebuild ready marker,
    (b) gated on the workflow_run event, (c) keyed on workflow_run.head_sha,
    (d) fail-safe (proceeds on timeout, never exit 1), (e) before the train step."""
    doc = _load(wf)
    steps = _job_steps(doc, job)
    gate = _find_split_gate(steps)
    assert gate is not None, (
        f"{wf}:{job} is missing the fresh-splits wait gate — it would race "
        f"refresh-splits and train one commit behind on a data-affecting merge "
        f"(the PR #383 incident; PR #516 fixed train-batch, this is the "
        f"train-ec2 follow-up)."
    )
    cond = str(gate.get("if", ""))
    assert "workflow_run" in cond, (
        f"{wf} gate must be scoped to the workflow_run event (workflow_dispatch "
        f"is deliberate/operator-driven); got if: {cond!r}"
    )
    env = gate.get("env", {}) or {}
    assert any("workflow_run.head_sha" in str(v) for v in env.values()), (
        f"{wf} gate must key the marker lookup on "
        f"github.event.workflow_run.head_sha (== the merge commit; matches the "
        f"job-def-revisions/<sha>.txt convention)."
    )
    run_body = gate.get("run", "")
    assert "ready/" in run_body, f"{wf} gate must wait on the splits-rebuild 'ready/' marker."
    # Fail-safe: every path proceeds; a gate bug must never block the GPU
    # pipeline. So NO `exit 1`, and the timeout path warns + proceeds.
    assert "exit 1" not in run_body, (
        f"{wf} gate must be fail-safe (proceed on timeout), never `exit 1` — a "
        f"gate bug should degrade to the old behavior, not block training."
    )
    assert "::warning::" in run_body, (
        f"{wf} gate should emit a ::warning:: on the poll-timeout path so a "
        f"stale-split train is at least visible in the run annotations."
    )
    names = [s.get("name", "") for s in steps]
    gate_idx = steps.index(gate)
    assert train_step in names, f"{wf}: could not find the training step {train_step!r}"
    assert gate_idx < names.index(train_step), (
        f"{wf}: the fresh-splits gate (idx {gate_idx}) must come before the "
        f"training step '{train_step}' (idx {names.index(train_step)}) — "
        f"waiting after training defeats the gate."
    )
