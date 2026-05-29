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


def test_splits_rebuild_markers_written_pending_then_ready():
    """PR #516's fail-safe gate: refresh-splits MUST write a per-SHA "pending"
    marker BEFORE the rebuild and a "ready" marker AFTER the S3 upload. The
    train workflows' "Wait for fresh data splits" gate keys off these (pending
    present ⇒ rebuild in flight ⇒ wait; ready present ⇒ splits fresh ⇒ proceed).
    If either write moves or disappears, the consumer gate silently degrades to
    the racy proceed-immediately path — the bug PR #516 closed."""
    steps = _refresh_job_steps()
    names = _step_names()
    regen_idx = names.index("Regenerate splits from nflverse + PBP")
    upload_idx = names.index("Upload to S3 with force")
    pending_idx = ready_idx = None
    for i, step in enumerate(steps):
        run_body = step.get("run", "") or ""
        if "splits-rebuild-markers/pending/" in run_body:
            pending_idx = i
        if "splits-rebuild-markers/ready/" in run_body:
            ready_idx = i
    assert pending_idx is not None, "missing the splits-rebuild 'pending' marker write"
    assert ready_idx is not None, "missing the splits-rebuild 'ready' marker write"
    assert pending_idx < regen_idx, (
        "the 'pending' marker must be written BEFORE the regenerate step so the "
        "train gate knows a rebuild is in flight for this SHA."
    )
    assert ready_idx > upload_idx, (
        "the 'ready' marker must be written AFTER 'Upload to S3 with force' so "
        "its presence means the S3 splits actually reflect this commit."
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


def test_regen_step_calls_build_features():
    """The ``Regenerate splits from nflverse + PBP`` step MUST call
    ``build_features`` between ``preprocess`` and ``temporal_split``.
    Origin: 8c46b59 (2026-05-21) trained on parquets missing 150 engineered
    cols because this step was ``preprocess(load_raw_data())`` →
    ``temporal_split`` with no ``build_features`` step in between. The
    ``build_position_features`` backfill silently zero-filled the missing
    cols. See TODO.md "[FIXED] refresh-splits.yml never called build_features"."""
    steps = _refresh_job_steps()
    regen_steps = [s for s in steps if s.get("name") == "Regenerate splits from nflverse + PBP"]
    assert len(regen_steps) == 1, (
        f"expected exactly 1 'Regenerate splits from nflverse + PBP' step, found {len(regen_steps)}"
    )
    run_body = regen_steps[0].get("run", "") or ""
    assert "from src.features.engineer import build_features" in run_body, (
        "Regenerate step must import build_features."
    )
    assert "build_features(" in run_body, (
        "Regenerate step must call build_features() between preprocess() "
        "and temporal_split() — otherwise the ~150 engineered cols are "
        "absent from the parquet and the pipeline trains on constant zeros."
    )


def test_verify_step_gates_s3_upload():
    """A ``Verify engineered columns present`` step MUST run between the
    regen and the S3 upload. Catches the 8c46b59 regression if a future
    edit drops the build_features call again — the upload won't go through
    if the column set is wrong."""
    names = _step_names()
    assert "Verify engineered columns present" in names, (
        "missing 'Verify engineered columns present' step — without it the "
        "regen-then-upload contract has no column-presence gate."
    )
    verify_idx = names.index("Verify engineered columns present")
    upload_idx = names.index("Upload to S3 with force")
    regen_idx = names.index("Regenerate splits from nflverse + PBP")
    assert regen_idx < verify_idx < upload_idx, (
        f"verify step must sit between regen (idx {regen_idx}) and upload "
        f"(idx {upload_idx}); got verify at idx {verify_idx}."
    )


def test_verify_step_carves_out_rb_runtime_computed_cols():
    """The verify step's ``runtime_added_per_pos`` map MUST exclude RB's
    ``prior_season_mean_catch_rate`` and ``prior_season_mean_yards_per_carry``.
    Both cols live under ``_INCLUDE_FEATURES["prior_season"]`` in
    [src/rb/config.py](../src/rb/config.py) for semantic reasons but are
    actually computed at training time by
    [src/rb/features.py](../src/rb/features.py)'s ``_compute_features``,
    not by ``build_features`` — so they will never be in the parquet,
    and without the carve-out the verify step false-positive-fails every
    refresh-splits run (regression observed on the 2026-05-21 run of
    PR #337's squash-merge commit ``6e82c2a``)."""
    steps = _refresh_job_steps()
    verify_steps = [s for s in steps if s.get("name") == "Verify engineered columns present"]
    assert len(verify_steps) == 1, (
        f"expected exactly 1 'Verify engineered columns present' step, found {len(verify_steps)}"
    )
    run_body = verify_steps[0].get("run", "") or ""
    assert "runtime_added_per_pos" in run_body, (
        "verify step is missing the runtime_added_per_pos exception map; "
        "without it, RB will false-positive-fail on prior_season_mean_catch_rate "
        "and prior_season_mean_yards_per_carry (computed at training time, "
        "not in the parquet)."
    )
    for col in ("prior_season_mean_catch_rate", "prior_season_mean_yards_per_carry"):
        assert col in run_body, (
            f"verify step is missing RB carve-out for {col!r} — see "
            f"src/rb/features.py::_compute_features which builds it at "
            f"training time, not in src/features/engineer.py::build_features."
        )


def test_build_position_features_raises_on_missing_whitelist_cols():
    """``build_position_features`` must raise ``KeyError`` when feature_cols
    references a column that the upstream pipeline didn't produce. Pre-fix
    behaviour silently back-filled the missing cols with constant zero and
    logged a print-statement warning, which is exactly how the 8c46b59
    regression went undetected in the Batch logs.

    This test bypasses the position-specific ``add_features_fn`` /
    ``fill_nans_fn`` to assert the contract directly on the broader
    whitelist back-fill block in ``src/shared/feature_build.py``.
    """
    import pandas as pd

    from src.shared.feature_build import build_position_features

    df = pd.DataFrame(
        {
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
            "recent_team": ["KC"] * 4,
            "player_id": ["pX"] * 4,
            "opponent_team": ["BUF"] * 4,
        }
    )
    feature_cols = ["should_have_been_engineered_by_build_features"]
    cfg = {
        "add_features_fn": lambda a, b, c: (a, b, c),
        "fill_nans_fn": lambda a, b, c, _cols: (a, b, c),
        "specific_features": [],
    }

    with pytest.raises(KeyError, match="whitelisted feature columns are missing"):
        build_position_features(df.copy(), df.copy(), df.copy(), cfg, feature_cols)
