"""Contract tests for [.github/workflows/refresh-splits.yml](
.github/workflows/refresh-splits.yml) — the workflow that regenerates
``data/splits/*.parquet`` in S3 and triggers a fresh ECS deploy so
serving containers re-sync their local splits.

The motivating bug (2026-05-21): refresh-splits.yml uploaded fresh
splits to S3 but did NOT kick the ECS service. The serving container's
in-flight refresh poller only watches model manifests, never the splits
parquet — so running tasks kept their boot-time stale copy of the
splits, ``_apply_position_models`` filtered ``attn_history_stats`` down
to the cols actually in their stale ``pos_test``, and the freshly-trained
model (with the new col set) hit a state_dict shape mismatch on the
in-flight model swap. A manual ``aws ecs update-service
--force-new-deployment`` cleared it.

This test pins the workflow's two coupled responsibilities so a future
edit can't silently break the contract:

1. Uploading splits to S3 (``upload_data``).
2. Kicking the ECS service so new tasks pick up those splits.

If either disappears in isolation, or the order ever flips, this test
catches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "refresh-splits.yml"

pytestmark = pytest.mark.unit


def _load_workflow() -> dict:
    """Parse the workflow YAML. ``yaml.safe_load`` returns the document
    with ``on:`` as the boolean ``True`` key (Python YAML quirk); callers
    that need the trigger block should fetch ``doc[True]`` or look it up
    via the helper below."""
    with WORKFLOW_PATH.open() as f:
        return yaml.safe_load(f)


def _refresh_job_steps() -> list[dict]:
    doc = _load_workflow()
    jobs = doc.get("jobs", {})
    refresh_job = jobs.get("refresh", {})
    return list(refresh_job.get("steps", []))


def _step_names() -> list[str]:
    """Step ``name:`` strings in order. ``uses:``-only steps return ''."""
    return [s.get("name", "") for s in _refresh_job_steps()]


def test_workflow_yaml_parses():
    """Smoke check: the workflow is valid YAML and has the expected
    top-level shape."""
    doc = _load_workflow()
    assert "jobs" in doc
    assert "refresh" in doc["jobs"]
    # Triggers — on: is the boolean True key under PyYAML.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict)
    assert "workflow_dispatch" in triggers
    assert "push" in triggers


def test_upload_to_s3_step_exists():
    """The workflow MUST have a step that runs ``upload_data`` against S3.
    Otherwise the regen does nothing operationally — splits never land."""
    names = _step_names()
    matches = [n for n in names if n == "Upload to S3 with force"]
    assert len(matches) == 1, (
        f"expected exactly 1 'Upload to S3 with force' step, found {len(matches)} in: {names}"
    )


def test_ecs_kick_step_exists_after_s3_upload():
    """The workflow MUST kick the ECS service AFTER uploading splits to S3
    so running serving tasks roll fresh and re-sync the new splits at boot.
    Without this, the in-flight model refresh poller swaps in models
    trained on the new splits while running tasks still have stale local
    splits — yielding state_dict shape mismatches (2026-05-21 RB outage)."""
    names = _step_names()
    upload_idx = names.index("Upload to S3 with force")
    # The ECS kick step's name need not be exact — match on any step that
    # both follows the upload AND runs ``aws ecs update-service
    # --force-new-deployment`` in its ``run:`` body.
    found_idx = None
    for idx, step in enumerate(_refresh_job_steps()):
        if idx <= upload_idx:
            continue
        run_body = step.get("run", "") or ""
        if "aws ecs update-service" in run_body and "--force-new-deployment" in run_body:
            found_idx = idx
            break
    assert found_idx is not None, (
        "no step after 'Upload to S3 with force' calls "
        "`aws ecs update-service --force-new-deployment`. "
        "Serving containers won't re-sync the freshly uploaded splits."
    )
    assert found_idx > upload_idx, (
        f"ECS kick step at index {found_idx} must come AFTER 'Upload to S3 with "
        f"force' (index {upload_idx}). Otherwise tasks could roll on STALE S3."
    )


def test_ecs_kick_targets_correct_cluster_and_service():
    """The ECS kick must target the same cluster/service the rest of the
    serving infra deploys to. Hardcoded names match ``train-batch.yml`` /
    ``deploy.yml`` / the existing ECS service ARN."""
    doc = _load_workflow()
    env = doc.get("env", {})
    assert env.get("ECS_CLUSTER") == "fantasy-cluster", (
        f"ECS_CLUSTER must be 'fantasy-cluster' (matches train-batch.yml / "
        f"deploy.yml); got: {env.get('ECS_CLUSTER')!r}"
    )
    assert env.get("ECS_SERVICE") == "fantasy-service", (
        f"ECS_SERVICE must be 'fantasy-service' (matches train-batch.yml / "
        f"deploy.yml); got: {env.get('ECS_SERVICE')!r}"
    )
