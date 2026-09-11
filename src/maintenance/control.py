"""Import-light Step Functions control and independent public-freshness verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from src.maintenance.coordination import lease
from src.maintenance.storage import (
    assert_models_current,
    digest,
    get_json,
    model_pins,
    now_iso,
    put_json,
    run_key,
    timestamp,
)


class NotVisibleYet(RuntimeError):
    pass


class BotoEcs:
    def __init__(self, client, elbv2=None, assert_lease=lambda: None):
        self.client = client
        self.elbv2 = elbv2
        self.assert_lease = assert_lease

    def call(self, api, operation, **arguments):
        self.assert_lease()
        if api == "ecs":
            client = self.client
        elif api == "elbv2":
            if self.elbv2 is None:
                import boto3

                self.elbv2 = boto3.client("elbv2", config=self.client.meta.config)
            client = self.elbv2
        else:
            raise ValueError(f"Unsupported deployment API: {api}")
        return getattr(client, operation.replace("-", "_"))(**arguments)


def service_state(ecs, cluster, service):
    response = ecs.describe_services(cluster=cluster, services=[service])
    if response.get("failures") or len(response.get("services", [])) != 1:
        raise RuntimeError("Serving service is unavailable")
    live = response["services"][0]
    definition = ecs.describe_task_definition(taskDefinition=live["taskDefinition"])[
        "taskDefinition"
    ]
    container = next(
        c for c in definition["containerDefinitions"] if c["name"] == "fantasy-predictor"
    )
    return live, container


def begin(s3, ecs, settings, event, metadata):
    from src.data.release import resolve_release

    kind = event["kind"]
    if kind not in {"inference", "weekly"}:
        raise ValueError("Unknown maintenance workflow")
    arguments = event.get("request") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Workflow input must be an object")
    if arguments.get("resume_run_id"):
        run_id = arguments["resume_run_id"]
        key = run_key(run_id, "request.json")
        existing, _ = get_json(s3, settings["bucket"], key)
        _validate_resumption(existing, kind, settings, metadata)
        return {"run_id": run_id, "request_key": key}
    # Scheduler retries of one occurrence resolve the same request and outputs.
    identity = arguments.get("scheduled_at") or event["execution_id"]
    run_id = digest([kind, identity, metadata["source_sha"], settings["mode"]])
    key = run_key(run_id, "request.json")
    existing, _ = get_json(s3, settings["bucket"], key, optional=True)
    if existing:
        _validate_resumption(existing, kind, settings, metadata)
        return {"run_id": run_id, "request_key": key}
    live, container = service_state(ecs, settings["cluster"], settings["service"])
    env = {x["name"]: x["value"] for x in container.get("environment", [])}
    selected = env.get("FF_DATA_RELEASE")
    if settings["mode"] == "shadow":
        selected = arguments.get("data_release", selected)
    elif container["image"].rsplit(":", 1)[-1] != metadata["source_sha"]:
        raise ValueError(
            "Maintenance image differs from the deployed source; update the maintenance stack"
        )
    if arguments.get("expected_data_release") not in (None, selected):
        raise ValueError("Expected data release is no longer deployed")
    _, manifest = resolve_release(s3, settings["bucket"], release_id=selected)
    if (
        settings["mode"] == "active"
        and manifest["data_producer_sha256"] != metadata["data_producer_sha256"]
    ):
        raise ValueError("Deployed data and maintenance producer differ")
    request = {
        "run_id": run_id,
        "kind": kind,
        "mode": settings["mode"],
        "check_sources": arguments.get("check_sources", True),
        "source_sha": metadata["source_sha"],
        "producer_sha": metadata["data_producer_sha256"],
        "data_release": selected,
        "models": model_pins(s3, settings["bucket"]),
        "service_revision": live["taskDefinition"],
        "created_at": now_iso(),
    }
    if kind == "weekly":
        pointer, etag = get_json(
            s3, settings["bucket"], "models/predictions_cache/current.json", optional=True
        )
        request["cache_pointer"] = {"value": pointer, "etag": etag}
        keys = [
            "data/manifest.json",
            f"data/by-producer/{metadata['data_producer_sha256']}/manifest.json",
        ]
        request["data_pointers"] = {
            key: {"value": value, "etag": etag}
            for key in keys
            for value, etag in [get_json(s3, settings["bucket"], key, optional=True)]
        }
    if not isinstance(request["check_sources"], bool):
        raise ValueError("check_sources must be a boolean")
    try:
        put_json(s3, settings["bucket"], key, request, IfNoneMatch="*")
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") != "PreconditionFailed":
            raise
    return {"run_id": run_id, "request_key": key}


def _validate_resumption(request, kind, settings, metadata):
    if (
        request["kind"] != kind
        or request["mode"] != settings["mode"]
        or request["source_sha"] != metadata["source_sha"]
        or request["producer_sha"] != metadata["data_producer_sha256"]
    ):
        raise ValueError("Saved run does not match this workflow/runtime; start a fresh execution")
    age = (datetime.now(UTC) - timestamp(request["created_at"])).total_seconds()
    if not -300 <= age <= (4 * 3600 if kind == "inference" else 24 * 3600):
        raise ValueError("Saved preparation expired; start a fresh execution")


def _assert_data_pointers(s3, bucket, request):
    for key, expected in request.get("data_pointers", {}).items():
        _, etag = get_json(s3, bucket, key, optional=True)
        if etag != expected["etag"]:
            raise RuntimeError("A newer data publication superseded this preparation")


def verified_blob(s3, bucket, descriptor):
    response = s3.get_object(Bucket=bucket, Key=descriptor["key"])
    stream = response["Body"]
    try:
        body = stream.read()
    finally:
        stream.close()
    if len(body) != descriptor["bytes"] or hashlib.sha256(body).hexdigest() != descriptor["sha256"]:
        raise ValueError("Staged artifact checksum mismatch")
    return body


def _check_live_request(s3, ecs, settings, request):
    live, container = service_state(ecs, settings["cluster"], settings["service"])
    env = {x["name"]: x["value"] for x in container.get("environment", [])}
    if live["taskDefinition"] != request["service_revision"]:
        raise RuntimeError("Serving definition changed during preparation; start a new run")
    if (
        container["image"].rsplit(":", 1)[-1] != request["source_sha"]
        or env.get("FF_DATA_RELEASE") != request["data_release"]
    ):
        raise RuntimeError("Serving inputs changed during preparation")
    assert_models_current(s3, settings["bucket"], request["models"])
    if request["kind"] == "weekly":
        _assert_data_pointers(s3, settings["bucket"], request)
        _, etag = get_json(
            s3, settings["bucket"], "models/predictions_cache/current.json", optional=True
        )
        if etag != request["cache_pointer"]["etag"]:
            raise RuntimeError("A newer serving snapshot superseded this preparation")


def staged_cache(s3, bucket, result):
    """Validate the candidate against the request's canonical model/data pins."""
    from src.artifacts.serving_snapshot import CACHE_SCHEMA_VERSION, FILES, SnapshotBuild

    cache, request = result["cache"], result["request"]
    if set(cache["files"]) != set(FILES):
        raise ValueError("Staged serving file inventory mismatch")
    values = dict(cache["build"])
    values["model_generations"] = tuple(tuple(x) for x in values["model_generations"])
    build = SnapshotBuild(**values)
    expected = {
        (pos, pin["etag"], pin["artifact"]["key"]) for pos, pin in request["models"].items()
    }
    if (
        len(build.model_generations) != len(expected)
        or set(build.model_generations) != expected
        or build.dataset_id != result["data_release"]
        or build.pointer_etag != request["cache_pointer"]["etag"]
    ):
        raise ValueError("Staged serving build differs from the pinned maintenance request")
    content = {}
    for name in FILES:
        descriptor = cache["files"][name]
        if descriptor["key"] != run_key(result["run_id"], name):
            raise ValueError("Staged serving file belongs to another maintenance run")
        content[name] = verified_blob(s3, bucket, descriptor)
    if json.loads(content["fingerprint.json"])["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("Staged serving cache schema is incompatible")
    return build, content


def _transport(ecs, settings):
    return BotoEcs(ecs, settings.get("elbv2"), settings.get("assert_lease", lambda: None))


def _weekly_receipt(result, task_definition, generation):
    return {
        "run_id": result["run_id"],
        "mode": "active",
        "kind": "weekly",
        "published_at": now_iso(),
        "data_release": result["data_release"],
        "task_definition": task_definition,
        "snapshot_generation": generation,
    }


def finish(s3, dynamo, ecs, settings, run_id):
    bucket = settings["bucket"]
    done, _ = get_json(s3, bucket, run_key(run_id, "published.json"), optional=True)
    if done and not done.get("rolled_back") and not done.get("rollback_pending"):
        return done
    result, _ = get_json(s3, bucket, run_key(run_id, "result.json"))
    request = result["request"]
    if request["mode"] != settings["mode"]:
        raise ValueError("Deployment mode changed during a run")
    if settings["mode"] == "shadow":
        if request["kind"] == "weekly":
            staged_cache(s3, bucket, result)
        else:
            verified_blob(s3, bucket, result["forecast"])
        receipt = {
            "run_id": run_id,
            "mode": "shadow",
            "kind": request["kind"],
            "published_at": now_iso(),
            "data_release": result.get("data_release", request["data_release"]),
        }
    else:
        # The Lambda is bounded at 900 seconds. Reserve time after the bounded
        # readiness wait for restoration; a successor cannot acquire this lease
        # while the current invocation still has authority.
        with lease(dynamo, settings["table"], "serving-publication", seconds=900) as owned:
            settings = {**settings, "assert_lease": owned}
            if request["kind"] == "weekly":
                from src.artifacts import deployment, serving_snapshot

                backup, _ = get_json(
                    s3, bucket, run_key(run_id, "activation-backup.json"), optional=True
                )
                rollout = (backup or {}).get("rollout")
                if rollout and rollout["phase"] == "complete":
                    pointer, _ = get_json(s3, bucket, "models/predictions_cache/current.json")
                    if pointer == backup[
                        "published_snapshot"
                    ] and deployment._expected_release_ready(_transport(ecs, settings), rollout):
                        assert_models_current(s3, bucket, request["models"])
                        serving_snapshot.verify_remote(
                            s3, bucket, expected_dataset_id=result["data_release"]
                        )
                        receipt = _weekly_receipt(
                            result, rollout["expected_task_definition"], pointer["generation"]
                        )
                        put_json(s3, bucket, run_key(run_id, "published.json"), receipt)
                        return receipt
                if backup:
                    restore_activation(s3, ecs, settings, result)
            _check_live_request(s3, ecs, settings, request)
            owned()
            if request["kind"] == "weekly":
                try:
                    receipt = activate_data(s3, ecs, settings, result)
                except Exception:
                    owned()
                    restore_activation(s3, ecs, settings, result)
                    raise
            else:
                receipt = publish_forecast(s3, bucket, result)
    put_json(s3, bucket, run_key(run_id, "published.json"), receipt)
    return receipt


def publish_forecast(s3, bucket, result):
    body = verified_blob(s3, bucket, result["forecast"])
    payload = json.loads(body)
    key = "models/predictions_cache/upcoming_week.json"
    old, etag = get_json(s3, bucket, key, optional=True)
    new_time = timestamp(payload.get("inputs_fetched_at", payload["generated_at"]))
    if not -300 <= (datetime.now(UTC) - new_time).total_seconds() <= 4 * 3600:
        raise ValueError("Staged forecast expired; start a fresh inference execution")
    if old and timestamp(old.get("inputs_fetched_at", old["generated_at"])) > new_time:
        raise RuntimeError("A newer forecast is already published")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
    )
    return {
        "run_id": result["run_id"],
        "mode": "active",
        "kind": "inference",
        "published_at": now_iso(),
        "data_release": result["request"]["data_release"],
    }


def activate_data(s3, ecs, settings, result):
    from src.artifacts import serving_snapshot
    from src.data.release import promote_release, resolve_release
    from src.scripts.advance_data_release import advance_release

    bucket = settings["bucket"]
    selected, manifest = resolve_release(s3, bucket, release_id=result["data_release"])
    if manifest["data_producer_sha256"] != result["request"]["producer_sha"]:
        raise ValueError("Candidate data was built by another producer")
    timeout = settings.get("rollout_timeout", 600)
    if timeout <= 0:
        raise RuntimeError("Insufficient invocation time for a recoverable serving rollout")
    build, content = staged_cache(s3, bucket, result)
    backup_key = run_key(result["run_id"], "activation-backup.json")
    backup, _ = get_json(s3, bucket, backup_key, optional=True)
    if backup is None:
        pointer, _ = get_json(s3, bucket, "models/predictions_cache/current.json", optional=True)
        backup = {
            "snapshot": pointer,
            "task_definition": result["request"]["service_revision"],
            "pointers": {
                key: get_json(s3, bucket, key, optional=True)[0]
                for key in result["request"]["data_pointers"]
            },
        }
        put_json(s3, bucket, backup_key, backup)

    def record_snapshot(pointer):
        settings.get("assert_lease", lambda: None)()
        backup["published_snapshot"] = pointer
        put_json(s3, bucket, backup_key, backup)

    def record_rollout(state):
        backup["rollout"] = state
        backup["activated_task_definition"] = state["expected_task_definition"]
        put_json(s3, bucket, backup_key, backup)

    with tempfile.TemporaryDirectory(prefix="maintenance-activation-") as directory:
        root = Path(directory)
        for name, payload in content.items():
            (root / name).write_bytes(payload)
        pointer = serving_snapshot.publish(s3, bucket, root, build, before_publish=record_snapshot)
        settings.get("assert_lease", lambda: None)()
        promote_release(s3, bucket, selected, manifest)
        outcome = advance_release(
            s3,
            _transport(ecs, settings),
            bucket=bucket,
            cluster=settings["cluster"],
            service=settings["service"],
            release_id=selected,
            expected_task_definition=result["request"]["service_revision"],
            state_path=root / "rollout.json",
            timeout=timeout,
            on_state=record_rollout,
        )
    if (
        not outcome["advanced"]
        and outcome.get("reason") != "running task already pins this release"
    ):
        raise RuntimeError(f"Data activation refused: {outcome.get('reason')}")
    return _weekly_receipt(
        result,
        outcome.get("task_definition", result["request"]["service_revision"]),
        pointer["generation"],
    )


def restore_activation(s3, ecs, settings, result):
    """Restore this run's canonical pointers and rollout, preserving later owners."""
    from src.artifacts import deployment

    bucket = settings["bucket"]
    backup_key = run_key(result["run_id"], "activation-backup.json")
    backup, _ = get_json(s3, bucket, backup_key, optional=True)
    if backup is None:
        return {"restored": False, "reason": "activation never began"}
    # Check all owners before the first restoration write.
    pointer_states = []
    for key, previous in backup["pointers"].items():
        current, etag = get_json(s3, bucket, key, optional=True)
        if current != previous and (
            not current or current.get("release_id") != result["data_release"]
        ):
            raise RuntimeError("A later data publication prevents automatic rollback")
        pointer_states.append((key, previous, current, etag))
    cache_key = "models/predictions_cache/current.json"
    current_pointer, pointer_etag = get_json(s3, bucket, cache_key, optional=True)
    if current_pointer not in (backup["snapshot"], backup.get("published_snapshot")):
        raise RuntimeError("A later cache publication prevents automatic rollback")
    live, container = service_state(ecs, settings["cluster"], settings["service"])
    if container["image"].rsplit(":", 1)[-1] != result["request"]["source_sha"]:
        raise RuntimeError("A later code deployment prevents automatic rollback")
    if live["taskDefinition"] not in {
        backup["task_definition"],
        backup.get("activated_task_definition"),
    }:
        raise RuntimeError("A later data deployment prevents automatic rollback")
    receipt_key = run_key(result["run_id"], "published.json")
    receipt, _ = get_json(s3, bucket, receipt_key, optional=True)
    put_json(s3, bucket, receipt_key, {**(receipt or {}), "rollback_pending": True})
    rollout = backup.get("rollout")
    if rollout:
        rollout = {**rollout, "phase": "rollback-requested"}

        def record_state(state):
            backup["rollout"] = state
            put_json(s3, bucket, backup_key, backup)

        with tempfile.TemporaryDirectory(prefix="maintenance-restore-") as directory:
            deployment.restore_rollout(
                _transport(ecs, settings),
                rollout,
                state_path=Path(directory) / "rollout.json",
                on_state=record_state,
            )
    for key, previous, current, etag in pointer_states:
        if current != previous:
            if previous is None:
                s3.delete_object(Bucket=bucket, Key=key, IfMatch=etag)
            else:
                put_json(s3, bucket, key, previous, IfMatch=etag)
    if current_pointer != backup["snapshot"]:
        if backup["snapshot"] is None:
            s3.delete_object(Bucket=bucket, Key=cache_key, IfMatch=pointer_etag)
        else:
            put_json(s3, bucket, cache_key, backup["snapshot"], IfMatch=pointer_etag)
    put_json(
        s3,
        bucket,
        receipt_key,
        {
            **(receipt or {}),
            "rollback_pending": False,
            "rolled_back": True,
            "rolled_back_at": now_iso(),
        },
    )
    return {"restored": True}


def fetch_public(url):
    with urlopen(url.rstrip("/") + "/api/upcoming_week", timeout=20) as response:
        body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Public forecast exceeds its response limit")
        return json.loads(body)


def verify(s3, ecs, settings, run_id, *, fetcher=fetch_public):
    receipt, _ = get_json(s3, settings["bucket"], run_key(run_id, "published.json"))
    if receipt["mode"] == "active":
        if receipt["kind"] == "weekly":
            live, container = service_state(ecs, settings["cluster"], settings["service"])
            env = {x["name"]: x["value"] for x in container.get("environment", [])}
            primary = next((d for d in live.get("deployments", []) if d["status"] == "PRIMARY"), {})
            if (
                env.get("FF_DATA_RELEASE") != receipt["data_release"]
                or live["taskDefinition"] != receipt["task_definition"]
                or env.get("FF_SERVING_SNAPSHOT_GENERATION") != receipt["snapshot_generation"]
                or primary.get("rolloutState") != "COMPLETED"
                or live.get("desiredCount", 0) <= 0
                or live.get("runningCount") != live.get("desiredCount")
            ):
                raise NotVisibleYet("Waiting for the exact data release to finish ECS rollout")
        else:
            payload = fetcher(settings["url"])
            if payload.get("maintenance", {}).get("run_id") != run_id:
                raise NotVisibleYet("Waiting for public forecast propagation")
            if (
                payload.get("available") is not False
                and payload.get("freshness", {}).get("status") != "fresh"
            ):
                raise ValueError("The public forecast is stale")
    receipt["verified_at"] = now_iso()
    namespace = "shadow" if receipt["mode"] == "shadow" else "status"
    put_json(s3, settings["bucket"], f"maintenance/{namespace}/{receipt['kind']}.json", receipt)
    return receipt


def watchdog(s3, settings, *, fetcher=fetch_public, now=None):
    now = now or datetime.now(UTC)
    problems = []
    for name, limit, time_key in [
        ("source-check", 26 * 3600, "checked_at"),
        ("weekly", 8 * 86400, "verified_at"),
    ]:
        try:
            receipt, _ = get_json(
                s3, settings["bucket"], f"maintenance/status/{name}.json", optional=True
            )
            age = (now - timestamp(receipt[time_key])).total_seconds() if receipt else limit + 1
            healthy = (
                receipt
                and (name != "source-check" or receipt.get("status") == "complete")
                and receipt.get("readiness") != "blocked"
                and -300 <= age <= limit
            )
        except (ValueError, KeyError, TypeError):
            healthy = False
        if not healthy:
            problems.append(f"{name} failed or overdue")
    try:
        payload = fetcher(settings["url"])
        cutoff = payload.get("inputs_fetched_at", payload.get("generated_at", ""))
        offseason = payload.get("available") is False and payload.get("reason") == "offseason"
        age = (now - timestamp(cutoff)).total_seconds()
        if not -300 <= age <= (26 * 3600 if offseason else 4 * 3600):
            problems.append("Forecast inputs overdue")
        if not offseason and (
            not payload.get("available")
            or payload.get("degraded_positions")
            or payload.get("freshness", {}).get("status") != "fresh"
        ):
            problems.append("Forecast unavailable, degraded, or stale")
    except Exception as error:
        problems.append(f"Public forecast check failed: {str(error)[:150]}")
    return {"ok": not problems, "problems": problems, "checked_at": now.isoformat()}


def handler(event, context):
    import boto3

    settings = {
        "bucket": os.environ["ARTIFACT_BUCKET"],
        "table": os.environ["LOCK_TABLE"],
        "cluster": os.environ["ECS_CLUSTER"],
        "service": os.environ["ECS_SERVICE"],
        "url": os.environ["SERVICE_URL"],
        "mode": os.environ["MAINTENANCE_MODE"],
        "rollout_timeout": min(
            600,
            (context.get_remaining_time_in_millis() / 1000 if context is not None else 900) - 180,
        ),
    }
    s3, ecs = boto3.client("s3"), boto3.client("ecs")
    action = event["action"]
    if action == "begin":
        metadata = json.loads(
            (Path(__file__).resolve().parents[2] / "maintenance-image.json").read_text()
        )
        return begin(s3, ecs, settings, event, metadata)
    if action == "finish":
        return finish(s3, boto3.client("dynamodb"), ecs, settings, event["run_id"])
    if action == "verify":
        return verify(s3, ecs, settings, event["run_id"])
    if action == "rollback":
        if settings["mode"] != "active":
            return {"restored": False, "reason": "shadow run"}
        result, _ = get_json(s3, settings["bucket"], run_key(event["run_id"], "result.json"))
        with lease(
            boto3.client("dynamodb"), settings["table"], "serving-publication", seconds=900
        ) as owned:
            owned()
            return restore_activation(s3, ecs, {**settings, "assert_lease": owned}, result)
    if action == "watchdog":
        result = watchdog(s3, settings)
        boto3.client("cloudwatch").put_metric_data(
            Namespace="Fantasy/Maintenance",
            MetricData=[
                {"MetricName": "FreshnessFailure", "Value": int(not result["ok"]), "Unit": "Count"}
            ],
        )
        put_json(s3, settings["bucket"], "maintenance/status/watchdog.json", result)
        return result
    raise ValueError("Unknown maintenance action")
