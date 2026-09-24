"""Run a frozen experiment campaign locally or on reusable Batch allocations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.tuning.campaign_contracts import (
    ROOT,
    execution_environment,
    identity,
    safe_id,
    source_fingerprint,
    validate,
    work_units,
)
from src.tuning.campaign_io import Journal, atomic_json, local_lock, snapshot_local_inputs


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _check_remote_source(sha):
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--exit-code",
            sha,
            "--",
            "src",
            "pyproject.toml",
            "requirements*.txt",
        ],
        check=True,
        capture_output=True,
    )
    untracked = _git("ls-files", "--others", "--exclude-standard", "src")
    if untracked:
        raise ValueError("Commit campaign source files before selecting a Batch image")


def _clients(region):
    import boto3
    from botocore.config import Config

    cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3})
    return {
        name: boto3.client(name, region_name=region, config=cfg) for name in ("s3", "batch", "ecr")
    }


def resolve_workloads(spec, backend):
    resolved = copy.deepcopy(spec)
    for step in resolved["steps"]:
        if step["kind"] != "ab":
            continue
        with tempfile.TemporaryDirectory(prefix="ff-campaign-resolve-") as temporary:
            request, response = Path(temporary) / "request.json", Path(temporary) / "response.json"
            atomic_json(request, step)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.tuning.campaign_worker",
                    "--resolve",
                    str(request),
                    "--backend",
                    backend,
                    "--output",
                    str(response),
                ],
                cwd=ROOT,
                check=True,
                timeout=120,
                stdout=subprocess.DEVNULL,
            )
            step.update(json.loads(response.read_text()))
    return resolved


def freeze(
    spec,
    backend,
    root,
    data_dir,
    *,
    bucket=None,
    region="us-east-1",
    image_sha=None,
    clients=None,
    fresh=False,
):
    from src.data.release import data_producer_hashes, resolve_compatible_release, resolve_release

    code_sha = image_sha or _git("rev-parse", "HEAD")
    expected = data_producer_hashes(ROOT)
    execution_spec = resolve_workloads(spec, backend)
    units = work_units(execution_spec, backend)
    image_uri = None
    if backend == "local":
        dataset_id = snapshot_local_inputs(data_dir, root / "inputs", expected_producer=expected)
        if spec.get("dataset_id") not in (None, dataset_id):
            raise ValueError("Selected local release differs from dataset_id")
    else:
        from src.scripts.resolve_training_image import resolve_ec2

        _check_remote_source(code_sha)
        binding = resolve_ec2(clients["ecr"], code_sha)
        image_uri = binding["image_uri"]
        if spec.get("dataset_id"):
            dataset_id, release = resolve_release(
                clients["s3"], bucket, release_id=spec["dataset_id"]
            )
            if any(release["producer"].get(name) != sha for name, sha in expected.items()):
                raise ValueError("Requested dataset is incompatible with campaign source")
        else:
            dataset_id, _ = resolve_compatible_release(clients["s3"], bucket, expected)
    manifest = {
        "version": 1,
        "id": spec["id"],
        "spec": spec,
        "backend": backend,
        "code_sha": code_sha,
        "source_fingerprint": source_fingerprint(),
        "dataset_id": dataset_id,
        "image_uri": image_uri,
        "bucket": bucket if backend == "batch" else None,
        "region": region,
        "fresh": bool(fresh),
        "execution_environment": execution_environment(),
        "units": units,
        "execution_steps": execution_spec["steps"],
    }
    if backend == "local":
        from src.shared.pipeline import _nn_device

        # Auto selection is resolved once; resuming cannot silently switch devices.
        manifest["execution_environment"]["FF_DEVICE"] = str(_nn_device())
    manifest["manifest_id"] = identity(manifest)
    return manifest


def _definition(batch, resource, image_uri):
    """Clone an existing resource shape into an isolated digest-pinned family."""
    template_name = "ff-training-cpu-job" if resource == "cpu" else "ff-training-job"
    name = f"ff-campaign-{resource}-job"
    templates = [
        d
        for page in batch.get_paginator("describe_job_definitions").paginate(
            jobDefinitionName=template_name, status="ACTIVE"
        )
        for d in page["jobDefinitions"]
    ]
    if not templates:
        raise ValueError(f"Missing resource template: {template_name}")
    template = max(templates, key=lambda d: d["revision"])
    properties = copy.deepcopy(template["containerProperties"])
    properties["image"] = image_uri
    properties["environment"] = [
        item
        for item in properties.get("environment", [])
        if not item["name"].startswith(
            ("FF_MODEL_", "FF_BUILD_", "FF_DATA", "FF_TRAIN_", "FF_LEGACY_", "FF_CAMPAIGN_")
        )
    ]
    for page in batch.get_paginator("describe_job_definitions").paginate(
        jobDefinitionName=name, status="ACTIVE"
    ):
        for definition in page["jobDefinitions"]:
            if definition["containerProperties"] == properties:
                return definition.get("jobDefinitionArn") or f"{name}:{definition['revision']}"
    from src.batch.launch import RETRY_STRATEGY

    registered = batch.register_job_definition(
        jobDefinitionName=name,
        type="container",
        containerProperties=properties,
        retryStrategy=RETRY_STRATEGY,
        platformCapabilities=["EC2"],
    )
    return registered.get("jobDefinitionArn") or f"{name}:{registered['revision']}"


def _find_submitted(batch, queue, job_name):
    rows = [
        row
        for page in batch.get_paginator("list_jobs").paginate(
            jobQueue=queue, filters=[{"name": "JOB_NAME", "values": [job_name]}]
        )
        for row in page.get("jobSummaryList", [])
        if row["jobName"] == job_name
    ]
    if len(rows) > 1:
        raise RuntimeError(f"Ambiguous campaign submission: {job_name}")
    return rows[0]["jobId"] if rows else None


def _unit_complete(unit, journal):
    progress, _ = journal.read(f"units/{unit['id']}/progress.json")
    steps = (progress or {}).get("steps", {})
    return all(
        steps.get(name, {}).get("state") == "SUCCEEDED"
        and journal.outputs_valid(
            journal.root / "units" / unit["id"] / "steps" / name,
            f"units/{unit['id']}/steps/{name}",
            steps[name].get("outputs", {}),
        )
        for name in unit["steps"]
    )


def submit_units(
    manifest,
    journal,
    batch,
    *,
    resume=False,
    gpu_queue="ff-training-queue",
    cpu_queue="ff-cpu-training-queue",
    attempt_timeout=10800,
):
    definitions = {}
    jobs = {}
    for unit in manifest["units"]:
        key = f"units/{unit['id']}/submission.json"
        prior, etag = journal.read(key)
        queue = cpu_queue if unit["resource"] == "cpu" else gpu_queue
        if prior and prior["manifest_id"] != manifest["manifest_id"]:
            raise ValueError("Submission belongs to another campaign identity")
        if prior:
            job_id = prior.get("job_id")
            if not job_id and prior.get("submit_error"):
                if not resume:
                    raise RuntimeError(
                        "Previous submission was rejected; correct the cause and use --resume"
                    )
            elif not job_id:
                job_id = _find_submitted(batch, queue, prior["job_name"])
                if not job_id:
                    raise RuntimeError(
                        "An earlier submission has an uncertain outcome; reconcile its recorded job name before retrying"
                    )
                prior["job_id"] = job_id
                etag = journal.write(key, prior, etag)
            if job_id:
                described = batch.describe_jobs(jobs=[job_id])["jobs"]
                if not described:
                    raise RuntimeError(
                        "Recorded Batch job is unavailable; refusing a blind duplicate"
                    )
                state = described[0]["status"]
                retryable = state == "FAILED" or (
                    state == "SUCCEEDED" and not _unit_complete(unit, journal)
                )
                if not retryable or not resume:
                    jobs[unit["id"]] = job_id
                    continue
        attempt = (prior or {}).get("attempt", 0) + 1
        resource = unit["resource"]
        if prior and prior.get("job_definition"):
            definition = prior["job_definition"]
        else:
            if resource not in definitions:
                definitions[resource] = _definition(batch, resource, manifest["image_uri"])
            definition = definitions[resource]
        job_name = f"ff-campaign-{manifest['manifest_id'][:16]}-{unit['id']}-{attempt}"
        intent = {
            "manifest_id": manifest["manifest_id"],
            "attempt": attempt,
            "job_name": job_name,
            "job_id": None,
            "job_definition": definition,
        }
        # This conditional intent is the submission lock: concurrent launchers
        # cannot both call submit_job for the same attempt.
        etag = journal.write(key, intent, etag)
        request = dict(
            jobName=job_name,
            jobQueue=queue,
            jobDefinition=definition,
            containerOverrides={
                "command": ["--mode", "campaign", "--position", unit["position"]],
                "environment": [
                    {"name": "S3_BUCKET", "value": manifest["bucket"]},
                    {"name": "FF_CAMPAIGN_ID", "value": manifest["id"]},
                    {"name": "FF_CAMPAIGN_UNIT", "value": unit["id"]},
                    {"name": "FF_CAMPAIGN_ATTEMPT", "value": str(attempt)},
                    {"name": "FF_CAMPAIGN_MANIFEST_ID", "value": manifest["manifest_id"]},
                ],
            },
            timeout={"attemptDurationSeconds": attempt_timeout},
        )
        from botocore.exceptions import ClientError

        try:
            response = batch.submit_job(**request)
        except ClientError as exc:
            # A definitive rejection can be retried after fixing its cause.
            # Transport errors and 5xx outcomes remain uncertain and must be
            # reconciled by the stable job name, never blindly duplicated.
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in {
                400,
                401,
                403,
                404,
                413,
            }:
                intent["submit_error"] = exc.response["Error"]["Code"]
                journal.write(key, intent, etag)
            raise
        intent["job_id"] = response["jobId"]
        journal.write(key, intent, etag)
        jobs[unit["id"]] = intent["job_id"]
    return jobs


def status(manifest, journal, batch=None):
    result = {"id": manifest["id"], "manifest_id": manifest["manifest_id"], "units": {}}
    complete = True
    failed = False
    active = False
    for unit in manifest["units"]:
        progress, _ = journal.read(f"units/{unit['id']}/progress.json")
        steps = (progress or {}).get("steps", {})
        states = {name: steps.get(name, {}).get("state", "PENDING") for name in unit["steps"]}
        entry = {"steps": states}
        submitted, _ = journal.read(f"units/{unit['id']}/submission.json")
        if submitted and submitted.get("job_id") and batch is not None:
            jobs = batch.describe_jobs(jobs=[submitted["job_id"]])["jobs"]
            entry["job_state"] = jobs[0]["status"] if jobs else "UNKNOWN"
            failed |= entry["job_state"] == "FAILED"
            active |= entry["job_state"] not in {"FAILED", "SUCCEEDED", "UNKNOWN"}
            if entry["job_state"] == "SUCCEEDED" and not _unit_complete(unit, journal):
                entry["error"] = "Job ended without verified complete outputs"
                failed = True
            complete &= entry["job_state"] == "SUCCEEDED"
        complete &= all(state == "SUCCEEDED" for state in states.values()) and _unit_complete(
            unit, journal
        )
        failed |= any(state == "FAILED" for state in states.values())
        result["units"][unit["id"]] = entry
    result["state"] = (
        "SUCCEEDED" if complete else "RUNNING" if active else "FAILED" if failed else "RUNNING"
    )
    return result


def run_batch_entry(position):
    from src.artifacts.source import image_source_sha
    from src.data.release import download_release
    from src.tuning.campaign_worker import run_unit

    campaign_id = safe_id(os.environ["FF_CAMPAIGN_ID"])
    unit_id = safe_id(os.environ["FF_CAMPAIGN_UNIT"])
    bucket = os.environ["S3_BUCKET"]
    import boto3

    root = Path("/opt/ml/campaign") / campaign_id
    journal = Journal(root, s3=boto3.client("s3"), bucket=bucket, campaign_id=campaign_id)
    manifest, _ = journal.read("manifest.json")
    if manifest is None or manifest["manifest_id"] != os.environ["FF_CAMPAIGN_MANIFEST_ID"]:
        raise RuntimeError("Missing or mismatched campaign manifest")
    if image_source_sha() != manifest["code_sha"]:
        raise RuntimeError("Campaign source differs from the built image")
    unit = next(unit for unit in manifest["units"] if unit["id"] == unit_id)
    if unit["position"] != position:
        raise RuntimeError("Worker position differs from the assigned unit")
    data_dir = root / "inputs"
    download_release(
        journal.s3,
        bucket,
        release_id=manifest["dataset_id"],
        raw_dir=data_dir / "raw",
        splits_dir=data_dir / "splits",
    )
    code = run_unit(
        manifest,
        unit,
        journal,
        data_dir=data_dir,
        directory=root / "units" / unit_id,
        attempt=int(os.environ["FF_CAMPAIGN_ATTEMPT"]),
    )
    raise SystemExit(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--backend", choices=("local", "batch"), default="local")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/campaigns")
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--image-sha")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=10800)
    parser.add_argument("--attempt-timeout", type=int, default=10800)
    args = parser.parse_args(argv)
    if args.wait_timeout < 1 or args.attempt_timeout < 60:
        parser.error(
            "--wait-timeout must be positive; --attempt-timeout must be at least 60 seconds"
        )
    spec = validate(json.loads(args.file.read_text()))
    root = args.output_dir.resolve() / spec["id"]
    if args.dry_run:
        resolved = resolve_workloads(spec, args.backend)
        print(
            json.dumps(
                {
                    "id": spec["id"],
                    "backend": args.backend,
                    "units": work_units(resolved, args.backend),
                    "steps": resolved["steps"],
                    "resolved_image_and_dataset": False,
                },
                indent=2,
            )
        )
        return 0
    clients = _clients(args.region) if args.backend == "batch" else None
    journal = Journal(
        root, s3=clients["s3"] if clients else None, bucket=args.bucket, campaign_id=spec["id"]
    )
    with local_lock(root):
        existing, _ = journal.read("manifest.json")
        if args.status:
            if existing is None:
                raise ValueError("Campaign does not exist")
            print(
                json.dumps(
                    status(existing, journal, clients["batch"] if clients else None), indent=2
                )
            )
            return 0
        if existing is not None:
            if not args.resume:
                raise ValueError("Campaign already exists; use --resume or a new ID")
            if (
                existing["spec"] != spec
                or existing["backend"] != args.backend
                or existing["fresh"] != args.fresh
            ):
                raise ValueError("Campaign settings changed; choose a new ID")
            if args.image_sha and args.image_sha != existing["code_sha"]:
                raise ValueError("Resume cannot change the image source")
            manifest = existing
        else:
            manifest = freeze(
                spec,
                args.backend,
                root,
                args.data_dir,
                bucket=args.bucket,
                region=args.region,
                image_sha=args.image_sha,
                clients=clients,
                fresh=args.fresh,
            )
            journal.immutable("manifest.json", manifest)
        atomic_json(root / "manifest.json", manifest)
        if args.backend == "local":
            from src.tuning.campaign_worker import run_unit

            previous, _ = journal.read("units/local/progress.json")
            return run_unit(
                manifest,
                manifest["units"][0],
                journal,
                data_dir=root / "inputs",
                directory=root / "units/local",
                attempt=(previous or {}).get("attempt", 0) + 1,
            )
        jobs = submit_units(
            manifest,
            journal,
            clients["batch"],
            resume=args.resume,
            attempt_timeout=args.attempt_timeout,
        )
        print(json.dumps({"campaign": manifest["id"], "jobs": jobs}, indent=2))
        if not args.wait:
            return 0
        deadline = time.monotonic() + args.wait_timeout
        while time.monotonic() < deadline:
            current = status(manifest, journal, clients["batch"])
            if current["state"] in {"SUCCEEDED", "FAILED"}:
                print(json.dumps(current, indent=2))
                return 0 if current["state"] == "SUCCEEDED" else 1
            time.sleep(30)
        raise TimeoutError("Campaign remains active; inspect --status or resume later")


if __name__ == "__main__":
    raise SystemExit(main())
