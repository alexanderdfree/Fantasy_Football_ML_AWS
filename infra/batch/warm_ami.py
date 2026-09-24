"""Measure, activate and roll back a pre-pulled AMI with immutable evidence.

Canaries use temporary Spot queues, the existing training launcher and isolated
artifact receipts. Production capacity limits and publication prefixes are never
used as experiment outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TYPES = ("g6.xlarge", "g5.xlarge")
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
_UPLOAD_STATE = None


def save(path, value, *, strict=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)
    if _UPLOAD_STATE is not None:
        try:
            _UPLOAD_STATE(path.name, path.read_bytes())
        except Exception:
            if strict:
                raise
            print(
                f"Cleanup checkpoint could not upload; local evidence remains at {path}",
                file=sys.stderr,
            )


def recipe(sha):
    def read(name):
        return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{sha}:{name}"])

    before, separator, after = read("src/batch/Dockerfile.train").partition(b"COPY src/ src/")
    instructions = [
        line.strip().split(b" ", 1)[0]
        for line in after.splitlines()
        if line.strip() and not line.lstrip().startswith(b"#")
    ]
    if not separator or instructions != [b"ARG", b"RUN", b"ENTRYPOINT"]:
        raise ValueError("Training image layout changed; review the dependency boundary")
    return hashlib.sha256(before + read("src/batch/requirements.txt")).hexdigest()


def layers(ecr, image):
    repository, digest = image.split("/", 1)[1].split("@", 1)
    rows = ecr.batch_get_image(repositoryName=repository, imageIds=[{"imageDigest": digest}])[
        "images"
    ]
    if len(rows) != 1:
        raise ValueError("Image digest is unavailable")
    manifest = json.loads(rows[0]["imageManifest"])
    if "layers" not in manifest:
        raise ValueError("Expected a single linux/amd64 training image")
    return [layer["digest"] for layer in manifest["layers"]]


def freshness(bake, ecr, image_sha=""):
    from src.scripts.resolve_training_image import resolve_ec2

    current = resolve_ec2(ecr, image_sha)
    baked_layers = layers(ecr, bake["image_uri"])
    selected_layers = layers(ecr, current["image_uri"])
    fresh = (
        bake.get("eligible") is True
        and bake["dependency_recipe"] == recipe(current["image_sha"])
        and baked_layers[:-2] == selected_layers[:-2]
    )
    return {"fresh": fresh, "current_image": current, "baked_dependency_layers": baked_layers[:-2]}


def clients(region):
    import boto3
    from botocore.config import Config

    cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3})
    return {
        name: boto3.client(name, region_name=region, config=cfg)
        for name in ("ec2", "ecs", "batch", "ecr", "s3")
    }


def compute_environment(batch, name):
    rows = batch.describe_compute_environments(computeEnvironments=[name])["computeEnvironments"]
    return rows[0] if rows else None


def template(ec2, resources):
    selected = resources.get("launchTemplate", {})
    if selected.get("overrides"):
        raise ValueError("Per-instance launch-template overrides require a separate canary design")
    if not selected:
        return {}, {}
    selector = (
        {"LaunchTemplateId": selected["launchTemplateId"]}
        if selected.get("launchTemplateId")
        else {"LaunchTemplateName": selected["launchTemplateName"]}
    )
    row = ec2.describe_launch_template_versions(
        **selector, Versions=[selected.get("version", "$Default")]
    )["LaunchTemplateVersions"][0]
    return {"launchTemplateId": row["LaunchTemplateId"], "version": str(row["VersionNumber"])}, row[
        "LaunchTemplateData"
    ]


def validate_bake(bake, aws):
    if not bake.get("eligible"):
        raise ValueError("Bake has no verified resident-layer evidence")
    rows = aws["ec2"].describe_images(ImageIds=[bake["source_ami"], bake["candidate_ami"]])[
        "Images"
    ]
    if {row["ImageId"] for row in rows} != {bake["source_ami"], bake["candidate_ami"]}:
        raise ValueError("Source or candidate AMI is unavailable")
    if any(row["State"] != "available" or row["Architecture"] != "x86_64" for row in rows):
        raise ValueError("Both AMIs must be available x86_64 images")
    source = next(row for row in rows if row["ImageId"] == bake["source_ami"])
    if "al2023" not in source["Name"] or "gpu" not in source["Name"]:
        raise ValueError("The control must be the baked AL2023 ECS GPU source")


def wait_until(read, ready, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if ready(value):
            return value
        if isinstance(value, dict) and value.get("status") == "INVALID":
            raise RuntimeError(f"{label}: {value.get('statusReason', 'INVALID')}")
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {label}")


def create_resources(aws, baseline, bake, name, instance_type, arm, state, path):
    ec2, batch = aws["ec2"], aws["batch"]
    _, data = template(ec2, baseline["computeResources"])
    data = {**data, "ImageId": bake["candidate_ami"] if arm == "warm" else bake["source_ami"]}
    owned = {"name": name, "arm": arm, "instance_type": instance_type, "jobs": {}}
    state["resources"].append(owned)
    save(path, state)  # deterministic names survive an interrupted create response
    tag = {"ff-purpose": "warm-ami-canary", "ff-warm-ami-run": state["id"]}
    launch = ec2.create_launch_template(
        LaunchTemplateName=name,
        ClientToken=name,
        LaunchTemplateData=data,
        TagSpecifications=[
            {
                "ResourceType": "launch-template",
                "Tags": [{"Key": k, "Value": v} for k, v in tag.items()],
            }
        ],
    )["LaunchTemplate"]
    owned["launch_template"] = launch["LaunchTemplateId"]
    save(path, state)
    source = baseline["computeResources"]
    resources = {
        key: copy.deepcopy(source[key])
        for key in (
            "type",
            "allocationStrategy",
            "subnets",
            "securityGroupIds",
            "instanceRole",
            "spotIamFleetRole",
            "bidPercentage",
        )
        if key in source
    }
    resources.update(
        minvCpus=0,
        desiredvCpus=0,
        maxvCpus=4,
        instanceTypes=[instance_type],
        tags=tag,
        ec2Configuration=[{"imageType": "ECS_AL2023_NVIDIA"}],
        launchTemplate={"launchTemplateId": launch["LaunchTemplateId"], "version": "1"},
    )
    batch.create_compute_environment(
        computeEnvironmentName=name,
        type="MANAGED",
        state="ENABLED",
        serviceRole=baseline["serviceRole"],
        computeResources=resources,
        tags=tag,
    )
    ce = wait_until(
        lambda: compute_environment(batch, name), lambda v: v and v["status"] == "VALID", 300, name
    )
    owned["cluster"] = ce["ecsClusterArn"]
    batch.create_job_queue(
        jobQueueName=name,
        state="ENABLED",
        priority=1,
        computeEnvironmentOrder=[{"order": 1, "computeEnvironment": name}],
        tags=tag,
    )
    wait_until(
        lambda: batch.describe_job_queues(jobQueues=[name])["jobQueues"],
        lambda values: bool(values) and values[0]["status"] == "VALID",
        300,
        name + " queue",
    )
    save(path, state)
    return owned


def job_definition(aws, bake, state, path):
    from src.scripts.resolve_training_image import resolve_batch

    bound = resolve_batch(
        aws["batch"], aws["s3"], state["bucket"], sha=bake["source_sha"], include_definition=True
    )
    original = aws["batch"].describe_job_definitions(jobDefinitions=[bound["job_definition"]])[
        "jobDefinitions"
    ][0]
    properties = copy.deepcopy(original["containerProperties"])
    properties["image"] = bake["image_uri"]
    properties["environment"] = [
        item
        for item in properties.get("environment", [])
        if not item["name"].startswith(("FF_", "S3_DATA_PREFIX"))
    ]
    response = aws["batch"].register_job_definition(
        jobDefinitionName=state["id"],
        type="container",
        containerProperties=properties,
        platformCapabilities=["EC2"],
        retryStrategy={"attempts": 1},
    )
    state["job_definition"] = response["jobDefinitionArn"]
    save(path, state)
    return {
        "image_sha": bake["source_sha"],
        "gpu_definition": response["jobDefinitionArn"],
        "cpu_definition": response["jobDefinitionArn"],
        "gpu_image": bake["image_uri"],
        "cpu_image": bake["image_uri"],
    }


@contextlib.contextmanager
def environment(values):
    prior = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("FF_"):
            os.environ.pop(key)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def submit(aws, state, path, owned, position, binding):
    from src.batch import launch

    prefix = f"experiments/warm-ami/{state['id']}/{owned['name']}/models"
    values = {
        "FF_LEGACY_RUN_ID": owned["name"],
        "FF_TRAIN_GIT_SHA": state["bake"]["source_sha"],
        "FF_MODEL_S3_PREFIX": prefix,
        "FF_DATA_RELEASE": state["dataset_id"],
        "FF_DATASET_ID": state["dataset_id"],
        "FF_DATA_FORMAT": "data-release-v1",
    }

    class Queue:
        def submit_job(self, **request):
            request.update(
                jobQueue=owned["name"],
                jobName=f"{owned['name']}-{position}",
                retryStrategy={"attempts": 1},
                timeout={"attemptDurationSeconds": 900},
            )
            return aws["batch"].submit_job(**request)

    owned["jobs"][position] = {"job_name": f"{owned['name']}-{position}", "model_prefix": prefix}
    save(path, state)
    with environment(values):
        previous_bucket = launch.S3_BUCKET
        try:
            launch.S3_BUCKET = state["bucket"]
            launch.register_submission_source(aws["s3"])
            _, job_id = launch.submit_job(position, seed=42, batch_client=Queue(), binding=binding)
        finally:
            launch.S3_BUCKET = previous_bucket
    owned["jobs"][position]["job_id"] = job_id
    save(path, state)


def lifecycle(aws, owned, job):
    container = job.get("container", {})
    task_arn = container.get("taskArn")
    if not task_arn:
        return {}
    tasks = aws["ecs"].describe_tasks(cluster=owned["cluster"], tasks=[task_arn])["tasks"]
    if not tasks or not tasks[0].get("pullStoppedAt"):
        return {}
    task = tasks[0]
    host = aws["ecs"].describe_container_instances(
        cluster=owned["cluster"], containerInstances=[task["containerInstanceArn"]]
    )["containerInstances"][0]
    instance = aws["ec2"].describe_instances(InstanceIds=[host["ec2InstanceId"]])["Reservations"][
        0
    ]["Instances"][0]
    pull_start, pull_end = task["pullStartedAt"].timestamp(), task["pullStoppedAt"].timestamp()
    result = {
        "instance_id": instance["InstanceId"],
        "instance_type": instance["InstanceType"],
        "ami": instance["ImageId"],
        "pull_seconds": pull_end - pull_start,
        "queue_provision_seconds": pull_start - job["createdAt"] / 1000,
        "task_arn": task_arn,
        "pull_started_at": pull_start,
        "pull_stopped_at": pull_end,
    }
    if job.get("startedAt"):
        result["container_start_seconds"] = job["startedAt"] / 1000 - pull_end
    if job.get("stoppedAt"):
        result.update(
            total_seconds=(job["stoppedAt"] - job["createdAt"]) / 1000,
            execution_seconds=(job["stoppedAt"] - job["startedAt"]) / 1000,
        )
    return result


def wait_jobs(aws, state, path, arms, timeout):
    from src.artifacts.receipts import download_run_artifact

    deadline = time.monotonic() + timeout
    pending = [
        (owned, pos, row)
        for owned in arms
        for pos, row in owned["jobs"].items()
        if row.get("status") != "SUCCEEDED"
    ]
    while pending and time.monotonic() < deadline:
        jobs = {
            j["jobId"]: j
            for j in aws["batch"].describe_jobs(jobs=[row["job_id"] for _, _, row in pending])[
                "jobs"
            ]
        }
        remaining = []
        for owned, pos, row in pending:
            job = jobs[row["job_id"]]
            row["status"] = job["status"]
            row["lifecycle"] = {**row.get("lifecycle", {}), **lifecycle(aws, owned, job)}
            if job.get("stoppedAt") and job.get("startedAt"):
                row["lifecycle"].update(
                    total_seconds=(job["stoppedAt"] - job["createdAt"]) / 1000,
                    execution_seconds=(job["stoppedAt"] - job["startedAt"]) / 1000,
                )
            if job["status"] == "FAILED":
                save(path, state)
                raise RuntimeError(
                    f"Canary {owned['name']}/{pos} failed: {job.get('statusReason', '')}"
                )
            if job["status"] == "SUCCEEDED":
                with tempfile.TemporaryDirectory(prefix="warm-ami-receipt-") as directory:
                    row["metrics"] = download_run_artifact(
                        aws["s3"],
                        state["bucket"],
                        row["model_prefix"],
                        state["bake"]["source_sha"],
                        pos,
                        owned["name"],
                        Path(directory) / "model.tar.gz",
                        expected_dataset_id=state["dataset_id"],
                    )
            else:
                remaining.append((owned, pos, row))
        save(path, state)
        pending = remaining
        if pending:
            time.sleep(15)
    if pending:
        raise TimeoutError("Canary remained queued/running beyond its measurement budget")


def same_metrics(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_metrics(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_metrics(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5)
    return a == b


def metric_values(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key.endswith(("_metrics", "_ranking")) or key == "cohorts"
    }


def valid_metrics(metrics):
    models = [value for key, value in metrics.items() if key.endswith("_metrics")]
    return bool(models) and all(
        isinstance(value.get("total", {}).get(name), (int, float))
        and math.isfinite(value["total"][name])
        for value in models
        for name in ("mae", "rmse")
    )


def assess(state):
    errors, timings, seen_hosts = [], {}, set()
    for kind in TYPES:
        deltas, total_deltas, covered = [], [], set()
        for pair in range(3):
            arms = {
                r["arm"]: r
                for r in state["resources"]
                if r["instance_type"] == kind and r["pair"] == pair
            }
            if set(arms) != {"cold", "warm"}:
                errors.append(f"{kind}/{pair}: missing paired trial")
                continue
            for arm, resource in arms.items():
                row = resource["jobs"].get("RB", {})
                timing = row.get("lifecycle", {})
                expected_ami = state["bake"]["candidate_ami" if arm == "warm" else "source_ami"]
                if (
                    row.get("status") != "SUCCEEDED"
                    or timing.get("ami") != expected_ami
                    or timing.get("instance_type") != kind
                ):
                    errors.append(f"{kind}/{pair}/{arm}: missing successful runtime/AMI evidence")
                host = timing.get("instance_id")
                if not host or host in seen_hosts:
                    errors.append(f"{kind}/{pair}/{arm}: cold host was absent or reused")
                seen_hosts.add(host)
            cold, warm = arms["cold"]["jobs"], arms["warm"]["jobs"]
            if all(
                all(
                    isinstance(rows.get("RB", {}).get("lifecycle", {}).get(field), (int, float))
                    and math.isfinite(rows["RB"]["lifecycle"][field])
                    and rows["RB"]["lifecycle"][field] >= 0
                    for field in ("pull_seconds", "total_seconds")
                )
                for rows in (cold, warm)
            ):
                deltas.append(
                    cold["RB"]["lifecycle"]["pull_seconds"]
                    - warm["RB"]["lifecycle"]["pull_seconds"]
                )
                total_deltas.append(
                    cold["RB"]["lifecycle"]["total_seconds"]
                    - warm["RB"]["lifecycle"]["total_seconds"]
                )
            for pos in cold.keys() | warm.keys():
                a, b = cold.get(pos, {}).get("metrics", {}), warm.get(pos, {}).get("metrics", {})
                if (
                    not valid_metrics(a)
                    or not valid_metrics(b)
                    or not same_metrics(metric_values(a), metric_values(b))
                ):
                    errors.append(f"{kind}/{pair}/{pos}: metrics or cohort parity failed")
                else:
                    covered.add(pos)
            if not cold.get("RB", {}).get("metrics", {}).get("gpu_name") or not warm.get(
                "RB", {}
            ).get("metrics", {}).get("gpu_name"):
                errors.append(f"{kind}/{pair}: GPU execution evidence missing")
        if covered != set(POSITIONS):
            errors.append(f"{kind}: incomplete six-position comparison")
        if len(deltas) != 3 or statistics.median(deltas) < 60:
            errors.append(f"{kind}: median pull saving below 60 seconds or fewer than three pairs")
        if len(total_deltas) != 3 or statistics.median(total_deltas) < 0:
            errors.append(f"{kind}: total turnaround did not improve")
        timings[kind] = {
            "paired_pull_savings_seconds": deltas,
            "paired_total_savings_seconds": total_deltas,
        }
    return {"passed": not errors, "errors": errors, "timings": timings}


def cleanup(aws, state, path):
    batch, ec2 = aws["batch"], aws["ec2"]
    for owned in state["resources"]:
        if owned.get("cleaned"):
            continue
        name = owned["name"]
        ce = compute_environment(batch, name)
        queues = batch.describe_job_queues(jobQueues=[name])["jobQueues"]
        for resource in [ce, *queues]:
            if resource and resource.get("tags", {}).get("ff-warm-ami-run") != state["id"]:
                raise ValueError(f"Refusing to clean resources not owned by this canary: {name}")
        if queues:
            for row in owned["jobs"].values():
                if not row.get("job_id"):
                    found = [
                        j
                        for page in batch.get_paginator("list_jobs").paginate(
                            jobQueue=name,
                            filters=[{"name": "JOB_NAME", "values": [row["job_name"]]}],
                        )
                        for j in page.get("jobSummaryList", [])
                        if j["jobName"] == row["job_name"]
                    ]
                    for job in found:
                        if job["status"] not in {"SUCCEEDED", "FAILED"}:
                            batch.terminate_job(
                                jobId=job["jobId"], reason="Owned AMI canary cleanup"
                            )
                else:
                    jobs = batch.describe_jobs(jobs=[row["job_id"]])["jobs"]
                    if jobs and jobs[0]["status"] not in {"SUCCEEDED", "FAILED"}:
                        batch.terminate_job(jobId=row["job_id"], reason="Owned AMI canary cleanup")
            batch.update_job_queue(jobQueue=name, state="DISABLED")
            wait_until(
                lambda name=name: batch.describe_job_queues(jobQueues=[name])["jobQueues"],
                lambda q: not q or q[0]["status"] == "VALID",
                300,
                name + " disable",
            )
            batch.delete_job_queue(jobQueue=name)
            wait_until(
                lambda name=name: batch.describe_job_queues(jobQueues=[name])["jobQueues"],
                lambda q: not q,
                600,
                name + " delete",
            )
        if ce:
            batch.update_compute_environment(computeEnvironment=name, state="DISABLED")
            wait_until(
                lambda name=name: compute_environment(batch, name),
                lambda c: not c or c["status"] == "VALID",
                300,
                name + " disable",
            )
            batch.delete_compute_environment(computeEnvironment=name)
            wait_until(
                lambda name=name: compute_environment(batch, name),
                lambda c: not c,
                600,
                name + " delete",
            )
        if not owned.get("launch_template"):
            recovered = ec2.describe_launch_templates(
                Filters=[{"Name": "launch-template-name", "Values": [name]}]
            )["LaunchTemplates"]
            if recovered:
                tags = {tag["Key"]: tag["Value"] for tag in recovered[0].get("Tags", [])}
                if tags.get("ff-warm-ami-run") != state["id"]:
                    raise ValueError("Refusing to delete an unowned launch template")
                owned["launch_template"] = recovered[0]["LaunchTemplateId"]
        if owned.get("launch_template"):
            ec2.delete_launch_template(LaunchTemplateId=owned["launch_template"])
        owned["cleaned"] = True
        save(path, state, strict=False)


def canary(aws, bake, path, bucket, dataset_id, timeout, smoke=False):
    from src.data.release import resolve_compatible_release, resolve_release
    from src.scripts.wait_data_release import producer_hashes_at_revision

    if path.exists():
        raise ValueError("Evidence path already exists; inspect/clean it before choosing a new run")
    validate_bake(bake, aws)
    baseline = compute_environment(aws["batch"], "ff-gpu-spot")
    if baseline["computeResources"]["type"] != "SPOT" or baseline["status"] != "VALID":
        raise ValueError("The live fleet must be a valid Spot environment")
    expected = producer_hashes_at_revision(bake["source_sha"])
    selected, release = (
        resolve_release(aws["s3"], bucket, release_id=dataset_id)
        if dataset_id
        else resolve_compatible_release(aws["s3"], bucket, expected)
    )
    if any(release["producer"].get(name) != sha for name, sha in expected.items()):
        raise ValueError("Canary data differs from the image's producer")
    state = {
        "version": 1,
        "id": "ff-ami-" + uuid.uuid4().hex[:12],
        "bake": bake,
        "bucket": bucket,
        "dataset_id": selected,
        "baseline": baseline,
        "resources": [],
        "smoke": smoke,
    }
    save(path, state)
    try:
        binding = job_definition(aws, bake, state, path)
        for kind in TYPES[:1] if smoke else TYPES:
            for pair in range(1 if smoke else 3):
                arms = []
                for arm in ("cold", "warm"):
                    name = f"{state['id']}-{kind.split('.')[0]}-{pair}-{arm}"
                    owned = create_resources(aws, baseline, bake, name, kind, arm, state, path)
                    owned["pair"] = pair
                    arms.append(owned)
                for owned in arms:
                    submit(aws, state, path, owned, "RB", binding)
                wait_jobs(aws, state, path, arms, timeout)
                if not smoke:
                    for pos in [("QB", "WR"), ("TE", "K"), ("DST",)][pair]:
                        for owned in arms:
                            submit(aws, state, path, owned, pos, binding)
                    wait_jobs(aws, state, path, arms, timeout)
                cleanup(aws, state, path)
        state["assessment"] = assess(state)
        save(path, state)
        print(json.dumps(state["assessment"], indent=2))
    finally:
        cleanup(aws, state, path)
        if state.get("job_definition"):
            aws["batch"].deregister_job_definition(jobDefinition=state["job_definition"])
    return 0 if smoke or state["assessment"]["passed"] else 1


def activate(aws, evidence, output, rollback=False):
    batch, ec2 = aws["batch"], aws["ec2"]
    current = compute_environment(batch, "ff-gpu-spot")
    if current["status"] != "VALID" or current["computeResources"]["desiredvCpus"] != 0:
        raise RuntimeError(
            "Wait for the GPU fleet to become valid and idle before changing its AMI"
        )
    if aws["ecs"].list_tasks(cluster=current["ecsClusterArn"], desiredStatus="RUNNING")["taskArns"]:
        raise RuntimeError(
            "GPU jobs are running; leave them undisturbed and retry after they finish"
        )
    previous, data = template(ec2, current["computeResources"])
    if rollback:
        if previous != evidence["selected_template"]:
            raise ValueError(
                "The active template changed since activation; inspect before rollback"
            )
        target = evidence["previous_template"]
    else:
        if evidence.get("smoke") or not assess(evidence)["passed"]:
            raise ValueError("Full paired canary gates have not passed")
        validate_bake(evidence["bake"], aws)
        if not freshness(evidence["bake"], aws["ecr"])["fresh"]:
            raise ValueError("Current training dependencies differ; rebuild and canary first")
        if any(
            row.get("imageIdOverride")
            for row in current["computeResources"].get("ec2Configuration", [])
        ):
            raise ValueError("A compute-resource AMI override would mask the launch template")
        if previous:
            row = ec2.create_launch_template_version(
                LaunchTemplateId=previous["launchTemplateId"],
                SourceVersion=previous["version"],
                LaunchTemplateData={"ImageId": evidence["bake"]["candidate_ami"]},
            )["LaunchTemplateVersion"]
            target = {
                "launchTemplateId": row["LaunchTemplateId"],
                "version": str(row["VersionNumber"]),
            }
        else:
            row = ec2.create_launch_template(
                LaunchTemplateName="ff-warm-" + uuid.uuid4().hex[:12],
                LaunchTemplateData={**data, "ImageId": evidence["bake"]["candidate_ami"]},
            )["LaunchTemplate"]
            target = {
                "launchTemplateId": row["LaunchTemplateId"],
                "version": str(row["LatestVersionNumber"]),
            }
    receipt = {
        "previous_environment": current,
        "previous_template": previous,
        "selected_template": target,
        "candidate_ami": evidence.get("candidate_ami")
        if rollback
        else evidence["bake"]["candidate_ami"],
        "rollback": rollback,
    }
    save(output, receipt)
    batch.update_compute_environment(
        computeEnvironment="ff-gpu-spot",
        computeResources={"launchTemplate": target},
        updatePolicy={"terminateJobsOnUpdate": False, "jobExecutionTimeoutMinutes": 180},
    )
    final = wait_until(
        lambda: compute_environment(batch, "ff-gpu-spot"),
        lambda c: c["status"] == "VALID" and template(ec2, c["computeResources"])[0] == target,
        1800,
        "GPU AMI rollout",
    )
    for key in ("type", "minvCpus", "maxvCpus", "instanceTypes", "subnets", "securityGroupIds"):
        if final["computeResources"][key] != current["computeResources"][key]:
            raise RuntimeError(f"Unexpected capacity/network drift during activation: {key}")
    if template(ec2, final["computeResources"])[0] != target:
        raise RuntimeError("Batch did not select the pinned launch-template version")
    receipt["verified"] = True
    save(output, receipt)


def main(argv=None):
    global _UPLOAD_STATE
    _UPLOAD_STATE = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "canary", "assess", "activate", "rollback", "cleanup")
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--data-release", default="")
    parser.add_argument("--image-sha", default="")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint-prefix",
        help="Optional experiments/warm-ami/ S3 prefix for controller recovery",
    )
    args = parser.parse_args(argv)
    value = json.loads(args.input.read_text())
    if args.action == "assess":
        print(json.dumps(assess(value), indent=2))
        return 0 if assess(value)["passed"] else 1
    if args.action in {"canary", "activate", "rollback"} and (
        args.output is None or args.output.exists()
    ):
        parser.error("Choose a new --output path to preserve existing evidence")
    if args.timeout < 60:
        parser.error("--timeout must be at least 60 seconds")
    aws = clients(args.region)
    if args.checkpoint_prefix:
        if not args.checkpoint_prefix.startswith(
            "experiments/warm-ami/"
        ) or ".." in args.checkpoint_prefix.split("/"):
            parser.error("Controller checkpoints must use an experiments/warm-ami/ prefix")

        def upload(name, body):
            aws["s3"].put_object(
                Bucket=args.bucket,
                Key=f"{args.checkpoint_prefix.rstrip('/')}/{name}",
                Body=body,
                ContentType="application/json",
            )

        _UPLOAD_STATE = upload
    if args.action == "check":
        result = freshness(value, aws["ecr"], args.image_sha)
        print(json.dumps(result, indent=2))
        return 0 if result["fresh"] else 1
    if args.action == "canary":
        return canary(
            aws, value, args.output, args.bucket, args.data_release, args.timeout, args.smoke
        )
    if args.action == "cleanup":
        cleanup(aws, value, args.input)
    else:
        activate(aws, value, args.output, rollback=args.action == "rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
