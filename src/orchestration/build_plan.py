"""Select verified data and pin a reviewable Batch execution plan before submission."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path

from src.artifacts.intent import reserve_intent, validate_intent
from src.artifacts.model_sync import load_publication_manifest
from src.artifacts.source import read_source, register_source
from src.orchestration.datasets import (
    DATA_RELEASE_FORMAT,
    LEGACY_DATASET_FORMAT,
    DatasetError,
    content_id,
    load_dataset,
    put_immutable_json,
    read_json,
    require_id,
    select_dataset,
    source_identity,
)

PROVENANCE_ENV = {
    "dataset_id": "FF_DATASET_ID",
    "data_release": "FF_DATA_RELEASE",
    "data_format": "FF_DATA_FORMAT",
    "build_plan_id": "FF_BUILD_PLAN_ID",
    "git_sha": "FF_TRAIN_GIT_SHA",
    "image_id": "FF_TRAIN_IMAGE_ID",
}


def provenance() -> dict[str, str]:
    return {name: os.environ[env] for name, env in PROVENANCE_ENV.items() if os.environ.get(env)}


def resolve_job_definition(batch, reference: str) -> dict:
    """Use the shared actual-image resolver and preserve its immutable revision."""
    from src.scripts.resolve_training_image import resolve_batch, resolve_definition

    if ":" in reference:
        selected = resolve_definition(batch, reference, include_image=True)
    else:
        selected = resolve_batch(batch, None, "", name=reference, include_definition=True)
    return {"arn": selected["job_definition"], "image": selected["image"]}


def resolve_execution_source(batch, code_sha: str, gpu_definition: str, cpu_definition: str | None):
    """Capture revision ARNs before waiting for data; infer manual image identity.

    A dispatch without an explicit image SHA means the latest active GPU
    definition, not necessarily the runner checkout's newer Git commit.
    """
    gpu = resolve_job_definition(batch, gpu_definition)
    cpu = resolve_job_definition(batch, cpu_definition) if cpu_definition else None
    if not code_sha:
        match = re.search(r":([0-9a-f]{40})$", gpu["image"])
        if match is None:
            raise DatasetError("Cannot infer code SHA: Batch image needs a full Git SHA tag")
        code_sha = match.group(1)
    for definition in (gpu, cpu):
        if definition and not definition["image"].endswith(":" + code_sha):
            raise DatasetError("Resolved Batch image tag does not match the requested code SHA")
    return code_sha, gpu["arn"], cpu["arn"] if cpu else None


def create_plan(
    s3,
    batch,
    bucket: str,
    *,
    dataset_id: str,
    source_id: str,
    code_sha: str,
    gpu_definition: str,
    cpu_definition: str | None = None,
    positions: list[str],
    seed: int,
    run_id: str | None = None,
    model_prefix: str = "models",
    data_format: str = DATA_RELEASE_FORMAT,
    repo: Path = Path("."),
) -> tuple[str, dict]:
    model_prefix = model_prefix.strip("/")
    read_source(s3, bucket, model_prefix, code_sha)
    dataset = load_dataset(s3, bucket, dataset_id, data_format=data_format)
    if dataset["source_id"] != source_id:
        raise DatasetError("Build plan dataset/source mismatch")
    if data_format == DATA_RELEASE_FORMAT:
        from src.data.release import producer_fingerprint
        from src.scripts.wait_data_release import producer_hashes_at_revision

        expected = producer_hashes_at_revision(code_sha, repo_root=repo)
        if (
            not expected
            or producer_fingerprint(expected) != source_id
            or any(
                dataset["release_manifest"]["producer"].get(name) != digest
                for name, digest in expected.items()
            )
        ):
            raise DatasetError("Data release producer differs from the selected training image")
    definitions = {"gpu": resolve_job_definition(batch, gpu_definition)}
    if cpu_definition:
        definitions["cpu"] = resolve_job_definition(batch, cpu_definition)
    for definition in definitions.values():
        if not definition["image"].endswith(":" + code_sha):
            raise DatasetError("Resolved Batch image tag does not match the requested code SHA")
    run_id = run_id or uuid.uuid4().hex
    intents = {}
    for position in positions:
        previous = load_publication_manifest(s3, bucket, model_prefix, position)
        intents[position] = reserve_intent(
            s3,
            bucket,
            model_prefix,
            position,
            code_sha,
            dataset_id,
            run_id,
            publication_revision=(previous or {}).get("rollback_epoch"),
        )
    plan = {
        "schema_version": 2 if data_format == DATA_RELEASE_FORMAT else 1,
        "dataset_id": dataset_id,
        "source_id": source_id,
        "git_sha": code_sha,
        "job_definitions": definitions,
        "positions": positions,
        "seed": seed,
        "run_id": run_id,
        "model_prefix": model_prefix,
        "intents": intents,
        "publication_revisions": {
            position: descriptor["publication_revision"] for position, descriptor in intents.items()
        },
    }
    if data_format == DATA_RELEASE_FORMAT:
        plan["data_format"] = DATA_RELEASE_FORMAT
    plan_id = content_id(plan)
    put_immutable_json(s3, bucket, f"build-plans/{plan_id}.json", plan)
    return plan_id, plan


def load_plan(s3, bucket: str, plan_id: str) -> dict:
    require_id(plan_id, "build plan ID")
    plan = read_json(s3, bucket, f"build-plans/{plan_id}.json")
    if plan.get("schema_version") not in {1, 2} or content_id(plan) != plan_id:
        raise DatasetError("Build plan identity/schema mismatch")
    require_id(plan.get("dataset_id"), "dataset ID")
    plan_data_format(plan)
    return plan


def plan_data_format(plan: dict) -> str:
    """Interpret retained plans without modifying their content-addressed body."""
    if (
        plan.get("schema_version") == 1
        and plan.get("data_format", LEGACY_DATASET_FORMAT) == LEGACY_DATASET_FORMAT
    ):
        return LEGACY_DATASET_FORMAT
    if plan.get("schema_version") == 2 and plan.get("data_format") == DATA_RELEASE_FORMAT:
        return DATA_RELEASE_FORMAT
    raise DatasetError("Build plan has an unsupported data format")


def verify_plan_data_environment(plan: dict) -> None:
    if os.environ.get("FF_DATASET_ID") != plan["dataset_id"]:
        raise DatasetError("Training dataset ID does not match build plan")
    release_id = os.environ.get("FF_DATA_RELEASE")
    if plan_data_format(plan) == DATA_RELEASE_FORMAT:
        if release_id != plan["dataset_id"]:
            raise DatasetError("FF_DATA_RELEASE and FF_DATASET_ID must match the build plan")
    elif release_id:
        raise DatasetError("Legacy dataset plans cannot also select FF_DATA_RELEASE")


def verify_training_plan(s3, bucket: str, *, position: str, seed: int, branch: str) -> dict | None:
    """Validate the plan/env contract; legacy no-ID operator runs stay explicit."""
    plan_id = os.environ.get("FF_BUILD_PLAN_ID")
    if not plan_id:
        identified_run = bool(os.environ.get("FF_LEGACY_RUN_ID"))
        dataset = os.environ.get("FF_DATASET_ID")
        release = os.environ.get("FF_DATA_RELEASE")
        if identified_run and dataset:
            require_id(dataset, "data release ID")
            if release != dataset:
                raise DatasetError("FF_DATA_RELEASE and FF_DATASET_ID must match")
        if os.environ.get("FF_REQUIRE_BUILD_PLAN") == "1" or (dataset and not identified_run):
            raise DatasetError("FF_BUILD_PLAN_ID is required for identified dataset training")
        return None
    plan = load_plan(s3, bucket, plan_id)
    verify_plan_data_environment(plan)
    if os.environ.get("FF_TRAIN_GIT_SHA") != plan["git_sha"]:
        raise DatasetError("Training code SHA does not match build plan")
    prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")
    if plan.get("model_prefix") != prefix:
        raise DatasetError("Training publication prefix does not match build plan")
    if position not in plan["positions"] or seed != plan["seed"]:
        raise DatasetError("Training position/seed does not match build plan")
    selected = (
        ["cpu"] if branch in {"cpu", "merge"} else ["gpu"] if branch == "nn" else ["gpu", "cpu"]
    )
    images = {plan["job_definitions"].get(role, {}).get("image") for role in selected}
    image = os.environ.get("FF_TRAIN_IMAGE_ID")
    if not image or image not in images:
        raise DatasetError("Training image reference does not match build plan")
    intent = plan.get("intents", {}).get(position)
    if (
        not isinstance(intent, dict)
        or intent.get("source_sha") != plan["git_sha"]
        or intent.get("dataset_id") != plan["dataset_id"]
        or intent.get("run_id") != plan["run_id"]
        or intent.get("position") != position
        or position not in plan.get("publication_revisions", {})
        or intent.get("publication_revision") != plan["publication_revisions"][position]
    ):
        raise DatasetError("Training publication intent does not match build plan")
    validate_intent(s3, bucket, prefix, intent)
    os.environ["FF_DATA_FORMAT"] = plan_data_format(plan)
    return plan


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model-prefix", default=os.environ.get("FF_MODEL_S3_PREFIX", "models"))
    parser.add_argument(
        "--code-sha", default="", help="Infer the selected image's SHA when omitted"
    )
    parser.add_argument("--gpu-definition", required=True)
    parser.add_argument("--cpu-definition")
    parser.add_argument("--positions", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import boto3

    s3, batch = boto3.client("s3"), boto3.client("batch")
    code_sha, gpu_definition, cpu_definition = resolve_execution_source(
        batch, args.code_sha, args.gpu_definition, args.cpu_definition
    )
    register_source(s3, args.bucket, args.model_prefix, code_sha, args.repo)
    source_id = source_identity(args.repo, code_sha)
    from src.scripts.wait_data_release import producer_hashes_at_revision

    dataset_id = select_dataset(
        s3,
        args.bucket,
        source_id,
        timeout=args.timeout,
        expected_hashes=producer_hashes_at_revision(code_sha, repo_root=args.repo),
    )
    plan_id, plan = create_plan(
        s3,
        batch,
        args.bucket,
        dataset_id=dataset_id,
        source_id=source_id,
        code_sha=code_sha,
        gpu_definition=gpu_definition,
        cpu_definition=cpu_definition,
        positions=args.positions,
        seed=args.seed,
        run_id=args.run_id,
        model_prefix=args.model_prefix,
        repo=args.repo,
    )
    output = {"build_plan_id": plan_id, **plan}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"build_plan_id={plan_id}\ndataset_id={dataset_id}\ndata_release={dataset_id}\ngit_sha={code_sha}\n"
            )
    print(json.dumps({"build_plan_id": plan_id, "dataset_id": dataset_id, "git_sha": code_sha}))


if __name__ == "__main__":
    main()
