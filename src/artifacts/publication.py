"""Source/intent-ordered publication in a namespace legacy writers cannot mutate."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import uuid
from pathlib import Path

from src.artifacts.intent import reserve_intent, validate_intent
from src.artifacts.model_sync import (
    _extract_tarball,
    build_manifest,
    history_prefix,
    load_legacy_manifest,
    load_manifest_snapshot,
    load_publication_manifest,
    new_history_key,
    write_manifest,
)
from src.artifacts.source import image_source_sha, load_source, source_key

_CONFLICT = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}
_LABELS = ("stable", "previous_stable", "current", "previous")


def _standalone_data_identity():
    """Bind identified standalone runs without relabeling legacy data as a release."""
    release = os.environ.get("FF_DATA_RELEASE", "")
    dataset = os.environ.get("FF_DATASET_ID", "")
    if release and release != "legacy":
        if re.fullmatch(r"[0-9a-f]{64}", release) is None or dataset != release:
            raise RuntimeError("Standalone release training requires equal canonical data aliases")
        return release
    if dataset:
        raise RuntimeError("Standalone dataset identity requires the same FF_DATA_RELEASE")
    if release != "legacy" and (
        os.environ.get("AWS_BATCH_JOB_ID") or os.environ.get("FF_REQUIRE_DATA_RELEASE") == "1"
    ):
        raise RuntimeError(
            "Remote standalone training requires a data release or explicit legacy mode"
        )
    return None


def validate_seed(position, data, source_sha):
    """Validate captured operator-import bytes without changing any remote state."""
    from src.shared.smoke_test import run_smoke_test

    source_key("models", source_sha)
    with tempfile.TemporaryDirectory(prefix="model-seed-") as temp:
        directory = Path(temp)
        _extract_tarball(data, directory)
        metrics = json.loads((directory / "benchmark_metrics.json").read_text())
        if (
            not isinstance(metrics, dict)
            or metrics.get("git_sha") != source_sha
            or metrics.get("position", position) != position
        ):
            raise RuntimeError("Seed artifact metrics must identify its actual source and position")
        run_smoke_test(position, directory)


def initialize_seed(s3, bucket, prefix, position, data, source_sha):
    """Import a validated seed only into an absent publication namespace.

    This explicit operator action makes no claim to be a new training run and
    therefore creates no training intent/receipt. Existing manifests from any
    supported protocol are never replaced. A concurrent upgraded writer wins
    the create-only CAS and its approved artifact is verified by the caller.
    """
    from botocore.exceptions import ClientError

    if load_publication_manifest(s3, bucket, prefix, position) is not None:
        return None
    source = load_source(s3, bucket, prefix, source_sha)
    validate_seed(position, data, source_sha)
    # Validation can be slow; recheck all predecessor heads before any writes.
    if load_publication_manifest(s3, bucket, prefix, position) is not None:
        return None
    digest = hashlib.sha256(data).hexdigest()
    key = new_history_key(prefix, position, "seed", digest)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in _CONFLICT:
            raise
        if s3.get_object(Bucket=bucket, Key=key)["Body"].read() != data:
            raise RuntimeError("Immutable seed artifact has conflicting content") from error
    # Catch predecessor publications that completed while uploading the seed.
    # Once the new pointer commits, old binaries cannot mutate its namespace.
    if load_publication_manifest(s3, bucket, prefix, position) is not None:
        return None
    manifest = build_manifest(key, digest[:7], len(data), "operator-seed", smoke_passed=True)
    manifest["current"].update(sha256=digest, git_sha=source_sha, origin="operator-seed")
    manifest["source_frontier"] = {name: source[name] for name in ("source_sha", "source_order")}
    try:
        write_manifest(s3, bucket, prefix, position, manifest, expected_etag=None)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in _CONFLICT:
            raise
        return None
    return manifest


def prepare_training(s3, bucket, prefix, position, plan=None):
    """Authenticate the executable and reserve/reuse intent before computation."""
    actual = image_source_sha()
    declared = os.environ.get("FF_TRAIN_GIT_SHA")
    if declared and declared != actual:
        raise RuntimeError("Training source environment disagrees with the actual image")
    source = load_source(s3, bucket, prefix, actual)
    if plan is not None:
        if plan["git_sha"] != actual:
            raise RuntimeError("Build plan source disagrees with the actual image")
        if plan.get("model_prefix", prefix) != prefix:
            raise RuntimeError("Build plan publication namespace differs from the training job")
        intent = plan.get("intents", {}).get(position)
        initial_revision = plan.get("publication_revisions", {}).get(position)
        expected = {
            "position": position,
            "source_sha": actual,
            "dataset_id": plan["dataset_id"],
            "run_id": plan["run_id"],
        }
        if not isinstance(intent, dict) or any(
            intent.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("Build plan publication intent does not match its training inputs")
    else:
        dataset_id = _standalone_data_identity()
        old = load_publication_manifest(s3, bucket, prefix, position)
        initial_revision = (old or {}).get("rollback_epoch")
        run_id = (
            os.environ.get("FF_LEGACY_RUN_ID")
            or os.environ.get("FF_SPLIT_RUN_ID")
            or os.environ.get("AWS_BATCH_JOB_ID")
            or uuid.uuid4().hex
        )
        intent = reserve_intent(
            s3,
            bucket,
            prefix,
            position,
            actual,
            dataset_id,
            run_id,
            publication_revision=initial_revision,
        )
        initial_revision = intent.get("publication_revision")
    validate_intent(s3, bucket, prefix, intent)  # Authenticate even an already-superseded run.
    os.environ["FF_TRAIN_GIT_SHA"] = actual
    os.environ["FF_PUBLICATION_INTENT"] = json.dumps(intent, sort_keys=True)
    os.environ["FF_PUBLICATION_REVISION"] = json.dumps(initial_revision)
    return {"source": source, "intent": intent, "initial_revision": initial_revision}


def artifact_context(s3, bucket, prefix, position, metrics):
    """Validate persisted pre-training bindings; never allocate intent at upload."""
    if metrics.get("position") != position:
        raise RuntimeError("Artifact metrics position differs from its publication target")
    source = load_source(s3, bucket, prefix, metrics.get("git_sha"))
    intent = metrics.get("publication_intent")
    expected = {
        "position": position,
        "source_sha": source["source_sha"],
        "dataset_id": metrics.get("dataset_id"),
    }
    if not isinstance(intent, dict) or any(
        intent.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Artifact has no matching pre-training publication intent")
    if metrics.get("build_plan_id"):
        from src.orchestration.build_plan import load_plan

        plan = load_plan(s3, bucket, metrics["build_plan_id"])
        if (
            plan["git_sha"] != source["source_sha"]
            or plan["dataset_id"] != intent["dataset_id"]
            or plan["run_id"] != intent["run_id"]
            or plan.get("intents", {}).get(position) != intent
            or plan.get("model_prefix", prefix) != prefix
            or plan.get("publication_revisions", {}).get(position)
            != metrics.get("publication_revision")
        ):
            raise RuntimeError("Artifact publication intent differs from its immutable build plan")
    if metrics.get("publication_revision") != intent.get("publication_revision"):
        raise RuntimeError(
            "Artifact publication revision differs from its pre-training reservation"
        )
    validate_intent(s3, bucket, prefix, intent)
    return {
        "source": source,
        "intent": intent,
        "initial_revision": metrics.get("publication_revision"),
    }


def references(manifest):
    manifest = manifest or {}
    return set(manifest.get("history") or []) | {
        entry["key"] for label in _LABELS if (entry := manifest.get(label)) and entry.get("key")
    }


def _manifest_digest(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def protect_legacy(s3, bucket, prefix, position, source, *, require_lineage=True, dry_run=False):
    """Copy retained bytes before cutover; no protected pointer references legacy GC space."""
    from botocore.exceptions import ClientError

    old = load_legacy_manifest(s3, bucket, prefix, position)
    if old is None:
        return {}
    frontier = old.get("source_frontier")
    if frontier is not None and (
        not isinstance(frontier, dict)
        or frontier.get("source_sha") not in source["lineage"]
        or type(frontier.get("source_order")) is not int
        or frontier["source_order"]
        != source["source_order"] - source["lineage"].index(frontier["source_sha"])
    ):
        raise RuntimeError("Predecessor source frontier is not verified by the migrating lineage")
    copies, entries, source_shas = {}, {}, []
    for key in references(old):
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            try:
                stream = archive.extractfile("benchmark_metrics.json")
            except KeyError:
                stream = None
            metrics = json.load(stream) if stream is not None else {}
        sha = metrics.get("git_sha")
        if sha is not None and sha not in source["lineage"]:
            raise RuntimeError(
                "Legacy artifact source is missing or newer than the migrating executable"
            )
        if require_lineage and sha is None:
            raise RuntimeError(
                "Legacy artifact has no verifiable source; use explicit validated promotion"
            )
        if sha is not None:
            source_shas.append(sha)
        digest = hashlib.sha256(data).hexdigest()
        protected = new_history_key(prefix, position, "legacy", digest)
        if not dry_run:
            try:
                s3.put_object(Bucket=bucket, Key=protected, Body=data, IfNoneMatch="*")
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in _CONFLICT:
                    raise
                existing = s3.get_object(Bucket=bucket, Key=protected)["Body"].read()
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise RuntimeError("Protected legacy copy has conflicting content") from error
        copies[key] = protected
        entries[key] = {
            "key": protected,
            "sha256": digest,
            "bytes": len(data),
            "git_sha": sha,
            "source_sha": sha,
            "legacy_key": key,
        }
    migrated = {
        **old,
        "schema_version": 3,
        "history": [copies[key] for key in old.get("history", [])],
        "legacy_keys": copies,
        "predecessor_manifest_digest": _manifest_digest(old),
    }
    for label in _LABELS:
        entry = old.get(label)
        migrated[label] = {**entry, **entries[entry["key"]]} if entry else None
    if frontier:
        source_shas.append(frontier["source_sha"])
    if source_shas:
        newest = min(source_shas, key=source["lineage"].index)
        migrated["source_frontier"] = {
            "source_sha": newest,
            "source_order": source["source_order"] - source["lineage"].index(newest),
        }
    elif not require_lineage:
        # Explicit operator validation establishes a new publication frontier,
        # without pretending an unlabelled legacy model was trained by this code.
        migrated["source_frontier"] = {key: source[key] for key in ("source_sha", "source_order")}
    return migrated


def publish_candidate(s3, bucket, prefix, position, *, entry, context, initial_revision=None):
    """Return the promoted manifest, or None for an authenticated superseded run."""
    from botocore.exceptions import ClientError

    source, intent = context["source"], context["intent"]
    if (
        intent.get("position") != position
        or intent.get("source_sha") != source["source_sha"]
        or entry.get("git_sha") != source["source_sha"]
        or entry.get("publication_intent") != intent
        or entry.get("dataset_id") != intent.get("dataset_id")
    ):
        raise RuntimeError("Publication entry differs from its authenticated source/intent binding")
    if not entry["key"].startswith(history_prefix(prefix, position)):
        raise RuntimeError("Publication requires protected v3 artifact storage")
    migrated = None
    for _ in range(12):
        old, etag = load_manifest_snapshot(s3, bucket, prefix, position)
        if old is None:
            if migrated is None:
                migrated = protect_legacy(s3, bucket, prefix, position, source)
            # Copying a retained history can take time. Re-evaluate any old-
            # protocol publication/rollback that completed during that work.
            predecessor = load_legacy_manifest(s3, bucket, prefix, position)
            if (migrated or predecessor) and migrated.get(
                "predecessor_manifest_digest"
            ) != _manifest_digest(predecessor):
                migrated = None
                continue
            old = migrated
        frontier = old.get("source_frontier") or {}
        if frontier and frontier["source_sha"] not in source["lineage"]:
            return None
        barrier = old.get("rollback_source_barrier") or {}
        if barrier.get("source_sha") == source["source_sha"]:
            return None
        prior_intent = old.get("intent_frontier") or {}
        if (
            prior_intent.get("source_sha") == source["source_sha"]
            and prior_intent.get("sequence", 0) > intent["sequence"]
        ):
            return None
        if not validate_intent(s3, bucket, prefix, intent):
            return None
        if old.get("rollback_epoch") != initial_revision:
            return None
        if (
            (old.get("current") or {}).get("key") == entry["key"]
            and old.get("intent_frontier") == intent
            and (old.get("current") or {}).get("smoke_passed") == entry["smoke_passed"]
            and (not entry["smoke_passed"] or (old.get("stable") or {}).get("key") == entry["key"])
        ):
            return old
        new = build_manifest(
            entry["key"],
            entry["sha7"],
            entry["bytes"],
            entry["uploaded_at"],
            old_manifest=old,
            smoke_passed=entry["smoke_passed"],
        )
        new["current"].update(entry)
        new["source_frontier"] = {key: source[key] for key in ("source_sha", "source_order")}
        new["intent_frontier"] = intent
        new["rollback_epoch"] = old.get("rollback_epoch")
        new["promotion_mode"] = "automatic"
        if any(not key.startswith(history_prefix(prefix, position)) for key in references(new)):
            raise RuntimeError("Protected manifest cannot reference legacy artifact storage")
        try:
            write_manifest(s3, bucket, prefix, position, new, expected_etag=etag)
            return new
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in _CONFLICT:
                raise
    raise RuntimeError(
        "Artifact publication repeatedly conflicted; retained candidate without promotion"
    )
