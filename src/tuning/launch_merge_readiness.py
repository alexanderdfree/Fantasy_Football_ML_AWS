"""Gate a bounded merge-validation fleet through the existing Batch A/B launcher.

The smoke and full phases have separate Actions jobs and immutable namespaces.
Full execution always rechecks smoke evidence; neither phase trains locally,
cancels jobs, or publishes production artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

SPEC = "src.tuning.ab_merge_readiness"
SPEC_PATH = "src/tuning/ab_merge_readiness.py"
CAMPAIGN_PREFIX = "experiments/merge-readiness/20260911T181252Z"
RESULTS_PREFIX = f"{CAMPAIGN_PREFIX}/ab_runs"
VARIANTS = ("baseline", "corrected")
SEEDS = (42, 123, 7)


@dataclass(frozen=True)
class Campaign:
    nn_image_sha: str
    combined_image_sha: str
    nn_data_prefix: str
    combined_data_prefix: str
    nn_data_release: str
    combined_data_release: str
    replay_sha256: str

    @classmethod
    def from_json(cls, raw: str) -> Campaign:
        campaign = cls(**json.loads(raw))
        for name in ("nn_image_sha", "combined_image_sha"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(campaign, name)):
                raise ValueError(f"{name} must be an explicit full image SHA")
        for name in ("nn_data_release", "combined_data_release", "replay_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(campaign, name)):
                raise ValueError(f"{name} must be an explicit SHA256")
        for name in ("nn_data_prefix", "combined_data_prefix"):
            prefix = getattr(campaign, name)
            if not prefix.startswith(f"{CAMPAIGN_PREFIX}/data/") or any(
                part in ("", ".", "..") for part in prefix.split("/")
            ):
                raise ValueError(f"{name} must identify isolated campaign data")
        if campaign.nn_data_prefix == campaign.combined_data_prefix:
            raise ValueError("NN and combined inputs require separate data prefixes")
        return campaign


@dataclass(frozen=True)
class Run:
    kind: str
    phase: str
    run_id: str
    image_sha: str
    data_prefix: str
    data_release: str
    positions: tuple[str, ...]
    seeds: tuple[int, ...]

    @property
    def cell_keys(self) -> list[str]:
        return [f"{p}-{v}-{s}" for p in self.positions for v in VARIANTS for s in self.seeds]


def phase_runs(campaign: Campaign, run_id: str, phase: str) -> list[Run]:
    if phase not in ("smoke", "full") or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id
    ):
        raise ValueError("Invalid phase or immutable run ID")
    return [
        Run(
            kind=kind,
            phase=phase,
            run_id=f"{run_id}-{phase}-{kind}",
            image_sha=getattr(campaign, f"{kind}_image_sha"),
            data_prefix=getattr(campaign, f"{kind}_data_prefix"),
            data_release=getattr(campaign, f"{kind}_data_release"),
            positions=("QB",)
            if kind == "combined"
            else (("WR",) if phase == "smoke" else ("QB", "RB", "WR", "TE", "DST")),
            seeds=(42,) if phase == "smoke" else SEEDS,
        )
        for kind in ("nn", "combined")
    ]


def verify_sources(campaign: Campaign) -> str:
    """The launcher and both explicitly selected images must resolve one spec."""
    expected = Path(SPEC_PATH).read_bytes()
    for sha in (campaign.nn_image_sha, campaign.combined_image_sha):
        actual = subprocess.run(
            ["git", "show", f"{sha}:{SPEC_PATH}"], capture_output=True, check=True
        ).stdout
        if actual != expected:
            raise ValueError(f"Merge-readiness spec differs in image source {sha}")
    return hashlib.sha256(expected).hexdigest()


def launch_command(campaign: Campaign, run: Run, *, wait_timeout: int, attempt_timeout: int):
    return [
        sys.executable,
        "-m",
        "src.tuning.launch_ab",
        "--spec",
        SPEC,
        "--image-sha",
        run.image_sha,
        "--data-prefix",
        run.data_prefix,
        "--run-id",
        run.run_id,
        "--s3-prefix",
        RESULTS_PREFIX,
        "--positions",
        *run.positions,
        "--seeds",
        *map(str, run.seeds),
        "--only",
        *VARIANTS,
        "--cuda-graph",
        "auto",
        "--wait",
        "true",
        "--wait-timeout",
        str(wait_timeout),
        "--attempt-timeout",
        str(attempt_timeout),
        "--max-cells",
        str(len(run.cell_keys)),
        "--env",
        "FF_AMP_DTYPE=fp32",
        "--env",
        "FF_AB_STACKED=0",
        "--env",
        "FF_FEATURE_CACHE_DISABLE=1",
        "--env",
        "FF_MERGE_READINESS_QB_REPLAY=data/raw/qb_pregame_replay.parquet",
        "--env",
        f"FF_MERGE_READINESS_QB_REPLAY_SHA256={campaign.replay_sha256}",
    ]


def launch_run(campaign: Campaign, run: Run, *, wait_timeout: int, attempt_timeout: int) -> None:
    environment = os.environ.copy()
    # Each launcher pins its selected release against the actual image's producer
    # hashes. Never inherit a different previously bound dataset into this process.
    for name in ("FF_DATASET_ID", "FF_DATA_FORMAT", "FF_DATA_RELEASE"):
        environment.pop(name, None)
    environment["FF_DATA_RELEASE"] = run.data_release
    environment["FF_S3_BUCKET"] = os.environ.get("S3_BUCKET", "ff-predictor-training")
    print(f"Launching {run.run_id}: {len(run.cell_keys)} cells", flush=True)
    subprocess.run(
        launch_command(campaign, run, wait_timeout=wait_timeout, attempt_timeout=attempt_timeout),
        env=environment,
        check=True,
    )


def _json_object(s3, bucket: str, key: str) -> dict:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        data = body.read(4 * 1024 * 1024 + 1)
    finally:
        body.close()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError(f"Oversized evidence JSON: {key}")
    return json.loads(data)


def image_digests(ecr, campaign: Campaign) -> dict[str, str]:
    """Fail closed on mutable image tags; never equate a source tag with a digest."""
    digests = {}
    for kind in ("nn", "combined"):
        sha = getattr(campaign, f"{kind}_image_sha")
        details = ecr.describe_images(repositoryName="ff-training", imageIds=[{"imageTag": sha}])[
            "imageDetails"
        ]
        if len(details) != 1 or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", details[0].get("imageDigest", "")
        ):
            raise ValueError(f"Cannot resolve one ECR digest for {sha}")
        digests[kind] = details[0]["imageDigest"]
    return digests


def _write_controller_evidence(s3, bucket, key, value):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(value, sort_keys=True, indent=2).encode(),
        ContentType="application/json",
        IfNoneMatch="*",
    )


def _verified_manifest(s3, bucket: str, prefix: str) -> dict:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []) if "/manifest-" in obj["Key"])
    if len(keys) != 1:
        raise ValueError(
            f"Expected one immutable readiness manifest at {prefix}, found {len(keys)}"
        )
    key = keys[0]
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        raw = body.read(4 * 1024 * 1024 + 1)
    finally:
        body.close()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > 4 * 1024 * 1024 or key != f"{prefix}manifest-{digest}.json":
        raise ValueError(f"Readiness manifest content hash mismatch: {key}")
    return json.loads(raw)


def _verify_frame(s3, bucket: str, prefix: str, frame: dict) -> None:
    location, expected = frame["location"], frame["sha256"]
    if frame["n_rows"] <= 0 or not location.startswith(f"s3://{bucket}/{prefix}"):
        raise ValueError("Missing or incorrectly scoped evaluation artifact")
    key = location[len(f"s3://{bucket}/") :]
    digest = hashlib.sha256()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
    finally:
        body.close()
    if digest.hexdigest() != expected:
        raise ValueError(f"Evaluation artifact content hash mismatch: {key}")


def verify_cell(s3, bucket: str, campaign: Campaign, run: Run, cell_key: str) -> None:
    position, variant, seed = cell_key.split("-")
    identity = {"position": position, "variant": variant, "seed": int(seed)}
    root = f"{RESULTS_PREFIX}/{run.run_id}/"
    cell = _json_object(s3, bucket, f"{root}cells/{cell_key}.json")
    if cell.get("ok") is not True or any(cell.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Unavailable or failed validation cell: {run.run_id}/{cell_key}")
    if cell.get("provenance", {}).get("git_sha") != run.image_sha:
        raise ValueError(f"Cell image provenance mismatch: {cell_key}")
    metrics = cell.get("metrics", {}).get("readiness", {})
    for name in ("native_rows", "nn_full_step_capture", "attn_nn_full_step_capture"):
        if not isinstance(metrics.get(name), (int, float)) or metrics[name] <= 0:
            raise ValueError(f"Missing observed {name}: {cell_key}")
    if metrics.get("required_cohorts_available") != 4:
        raise ValueError(f"Required cohort evidence unavailable: {cell_key}")
    prefix = f"{root}readiness/{cell_key}/"
    manifest = _verified_manifest(s3, bucket, prefix)
    expected = {
        **identity,
        "schema_version": 1,
        "image_sha": run.image_sha,
        "data_release": run.data_release,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Readiness manifest identity mismatch: {cell_key}")
    capture = manifest.get("cuda_capture", {})
    if any(
        capture.get(key) is not True
        for key in ("required", "enabled_gate", "full_step_enabled_gate")
    ):
        raise ValueError(f"CUDA capture gates did not activate: {cell_key}")
    for family in ("nn", "attn_nn"):
        if not any(
            event.get("family") == family
            and str(event.get("device", "")).startswith("cuda")
            and event.get("use_amp") is False
            and all(
                event.get(key) is True
                for key in ("returned_true", "graph_present", "capturable_loss")
            )
            for event in capture.get("events", [])
        ):
            raise ValueError(f"No observed FP32 full-step CUDA capture for {family}: {cell_key}")
    _verify_frame(s3, bucket, prefix, manifest["evaluations"]["native"])
    if position == "QB":
        if metrics.get("pregame_rows", 0) <= 0:
            raise ValueError(f"QB pregame replay unavailable: {cell_key}")
        if manifest.get("replay_input", {}).get("sha256") != campaign.replay_sha256:
            raise ValueError(f"QB replay input differs from the common pin: {cell_key}")
        _verify_frame(s3, bucket, prefix, manifest["evaluations"]["pregame_replay"])


def verify_phase(s3, bucket: str, campaign: Campaign, runs: list[Run]) -> None:
    cells = [(run, key) for run in runs for key in run.cell_keys]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: verify_cell(s3, bucket, campaign, *item), cells))
    print(f"Verified {len(cells)} cells, capture events and evaluation artifacts", flush=True)


def run_phase(
    campaign, *, phase, run_id, s3, ecr, bucket, spec_sha256, wait_timeout, attempt_timeout
):
    runs = phase_runs(campaign, run_id, phase)
    evidence_prefix = f"{RESULTS_PREFIX}/controller/{run_id}/"
    requested = {
        "campaign": asdict(campaign),
        "spec_sha256": spec_sha256,
        "image_digests": image_digests(ecr, campaign),
    }
    if phase == "full":
        original = _json_object(s3, bucket, f"{evidence_prefix}request.json")
        smoke = _json_object(s3, bucket, f"{evidence_prefix}smoke-complete.json")
        if original != requested or smoke != {"verified_cells": 4, **requested}:
            raise ValueError(
                "Smoke source, image digest or campaign pins changed before full fanout"
            )
        # A successful Actions dependency alone cannot certify GPU/replay evidence.
        verify_phase(s3, bucket, campaign, phase_runs(campaign, run_id, "smoke"))
    for run in runs:
        existing = s3.list_objects_v2(
            Bucket=bucket, Prefix=f"{RESULTS_PREFIX}/{run.run_id}/", MaxKeys=1
        )
        if existing.get("Contents"):
            raise ValueError(f"Run namespace already exists; use a new run ID: {run.run_id}")
    if image_digests(ecr, campaign) != requested["image_digests"]:
        raise ValueError("ECR image digests changed before submission")
    if phase == "smoke":
        _write_controller_evidence(s3, bucket, f"{evidence_prefix}request.json", requested)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                launch_run,
                campaign,
                run,
                wait_timeout=wait_timeout,
                attempt_timeout=attempt_timeout,
            )
            for run in runs
        ]
        for future in futures:
            future.result()  # Failure/timeout never falls through to another phase.
    verify_phase(s3, bucket, campaign, runs)
    if image_digests(ecr, campaign) != requested["image_digests"]:
        raise ValueError("ECR image digests changed while validation jobs were running")
    _write_controller_evidence(
        s3,
        bucket,
        f"{evidence_prefix}{phase}-complete.json",
        {"verified_cells": sum(len(run.cell_keys) for run in runs), **requested},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, help="Explicit Campaign fields as JSON")
    parser.add_argument("--run-id", required=True, help="Unique ID shared by smoke and full phases")
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--wait-timeout", type=int, default=18000)
    parser.add_argument("--attempt-timeout", type=int, default=10800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.wait_timeout <= 18000 or not 60 <= args.attempt_timeout <= 18000:
        parser.error("Wait/attempt timeouts must fit the six-hour Actions job")
    campaign = Campaign.from_json(args.config_json)
    runs = phase_runs(campaign, args.run_id, args.phase)
    if args.dry_run:
        for run in runs:
            print(
                json.dumps(
                    {
                        "run": run.run_id,
                        "cells": run.cell_keys,
                        "data_release": run.data_release,
                        "command": launch_command(
                            campaign,
                            run,
                            wait_timeout=args.wait_timeout,
                            attempt_timeout=args.attempt_timeout,
                        ),
                    }
                )
            )
        return
    spec_sha256 = verify_sources(campaign)
    import boto3

    bucket = os.environ.get("S3_BUCKET", "ff-predictor-training")
    run_phase(
        campaign,
        phase=args.phase,
        run_id=args.run_id,
        bucket=bucket,
        s3=boto3.client("s3"),
        ecr=boto3.client("ecr"),
        spec_sha256=spec_sha256,
        wait_timeout=args.wait_timeout,
        attempt_timeout=args.attempt_timeout,
    )


if __name__ == "__main__":
    main()
