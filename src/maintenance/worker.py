"""Portable Fargate entrypoints for inference and staged historical releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.maintenance.coordination import lease
from src.maintenance.storage import (
    FORMATS,
    MODELS,
    POSITIONS,
    assert_models_current,
    get_json,
    json_bytes,
    now_iso,
    put_json,
    run_key,
    timestamp,
)


def validate_forecast(payload: dict) -> None:
    from src.artifacts.upcoming_transfer import decode_artifact
    from src.contracts.upcoming_status import freshness

    # Keep the producer at least as strict as the current serving reader.
    decode_artifact(json_bytes(payload))
    if payload.get("available") is False:
        if payload.get("reason") != "offseason":
            raise ValueError("Only verified offseason may publish an unavailable forecast")
        age = (datetime.now(UTC) - timestamp(payload["generated_at"])).total_seconds()
        if not -300 <= age <= 4 * 3600:
            raise ValueError("Verified offseason result is not fresh")
        return
    if freshness(payload)["status"] != "fresh":
        raise ValueError("Forecast inputs are not fresh")
    if payload.get("degraded_positions"):
        raise ValueError("A required model position is unavailable")
    if not isinstance(payload.get("season"), int) or not 1 <= payload.get("week", 0) <= 22:
        raise ValueError("Forecast has no valid NFL slate")
    expected = None
    roster = payload.get("sources", {}).get("roster", {})
    if roster.get("failed_teams"):
        raise ValueError("A scheduled team is missing its roster")
    teams = set(roster.get("covered_teams", []))
    for scoring in FORMATS:
        rows = payload.get("scoring", {}).get(scoring, [])
        if set(row["position"] for row in rows) != set(POSITIONS):
            raise ValueError(f"Missing forecast positions for {scoring}")
        keys = [(r["player_id"], r["position"], r["team"]) for r in rows]
        if len(set(keys)) != len(keys) or (expected is not None and set(keys) != expected):
            raise ValueError("Forecast scoring formats disagree or have duplicate players")
        expected = set(keys)
        if teams and any(
            {r["team"] for r in rows if r["position"] == pos} != teams for pos in POSITIONS
        ):
            raise ValueError("Forecast does not cover every scheduled team and position")
        for row in rows:
            if any(
                not isinstance(row.get(model), (int, float)) or not math.isfinite(row[model])
                for model in MODELS
            ):
                raise ValueError(f"Non-finite model output: {row['position']}")


def hydrate_models(s3, bucket: str, pins: dict, root: Path) -> None:
    from src.artifacts.model_sync import _extract_tarball, record_synced_model_key

    if set(pins) != set(POSITIONS):
        raise ValueError("A maintenance run must pin all six model positions")
    for pos in POSITIONS:
        pin = pins[pos]
        response = s3.get_object(Bucket=bucket, Key=pin["artifact"]["key"])
        stream = response["Body"]
        try:
            body = stream.read()
        finally:
            stream.close()
        if len(body) != pin["artifact"]["bytes"]:
            raise ValueError(f"Model artifact size mismatch: {pos}")
        directory = root / "src" / pos.lower() / "outputs" / "models"
        _extract_tarball(body, directory)
        (directory.parent / ".manifest-etag").write_text(pin["etag"])
        record_synced_model_key(root, pos, pin["artifact"]["key"])


def _record(s3, bucket, run_id, name, body, content_type):
    key = run_key(run_id, name)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    return {"key": key, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def produce_forecast(s3, bucket, request, *, root=Path("."), run=subprocess.run):
    from src.data.release import download_release
    from src.maintenance.readiness import live_checks

    download_release(s3, bucket, release_id=request["data_release"])
    hydrate_models(s3, bucket, request["models"], root)
    env = os.environ.copy()
    env.update(
        FF_DATA_RELEASE=request["data_release"],
        FF_DEVICE="cpu",
        FF_MODEL_S3_BUCKET="",
        GITHUB_SHA=request["source_sha"],
        NFLREADPY_CACHE="off",
    )
    # The existing CLI owns live-cache isolation. It builds locally; only this
    # job's validated publication stage has permission to advance production.
    run([sys.executable, "-m", "src.prediction.upcoming"], env=env, check=True)
    payload = json.loads((root / "data/serving_cache/upcoming_week.json").read_text())
    payload["maintenance"] = {
        "run_id": request["run_id"],
        "data_release": request["data_release"],
        "model_keys": {p: pin["artifact"]["key"] for p, pin in request["models"].items()},
        "source_sha": request["source_sha"],
    }
    validate_forecast(payload)
    checked = live_checks(payload)
    if any(c["readiness"] == "blocked" for c in checked.values()):
        raise ValueError("Required live source readiness/coverage checks failed")
    artifact = _record(
        s3, bucket, request["run_id"], "upcoming_week.json", json_bytes(payload), "application/json"
    )
    return {
        "forecast": artifact,
        "live_source_checks": checked,
        "generated_at": payload["generated_at"],
        "inputs_fetched_at": payload.get("inputs_fetched_at", payload["generated_at"]),
    }


def prepare_release(s3, bucket, request, *, root=Path("."), run=subprocess.run):
    from src.data.release import download_release, publish_release

    if any((root / "data" / name).exists() for name in ("raw", "splits")):
        raise ValueError("Historical rebuild requires a clean task filesystem")
    env = os.environ.copy()
    env.pop("FF_DATA_RELEASE", None)
    env.update(
        GITHUB_SHA=request["source_sha"],
        NFLREADPY_CACHE="off",
        FF_CAPTURE_PROVIDER_SOURCES="data/raw/provider_sources",
    )
    run([sys.executable, "-m", "src.data.maintenance_build"], env=env, check=True)
    # No current/index pointer moves until cache validation and compatible activation.
    selected = publish_release(s3, bucket, promote=False)
    # Install the sealed release markers before cache construction. Historical
    # dependencies must be replayed from this candidate, never fetched again.
    download_release(s3, bucket, release_id=selected)
    hydrate_models(s3, bucket, request["models"], root)
    staging = root / "maintenance-cache"
    env.update(FF_MODEL_S3_BUCKET=bucket, FF_DATA_RELEASE=selected, FF_DEVICE="cpu")
    run(
        [
            sys.executable,
            "-m",
            "src.prediction.build_snapshot",
            "--stage-directory",
            str(staging),
        ],
        env=env,
        check=True,
    )
    from src.artifacts.serving_snapshot import FILES

    cache = {
        "build": json.loads((staging / "build.json").read_text()),
        "files": {
            name: _record(
                s3,
                bucket,
                request["run_id"],
                name,
                (staging / name).read_bytes(),
                "application/octet-stream",
            )
            for name in FILES
        },
    }
    return {"data_release": selected, "cache": cache}


def execute(s3, dynamo, *, bucket, table, request, root=Path(".")):
    run_id = request["run_id"]
    metadata = json.loads((root / "maintenance-image.json").read_text())
    if (
        metadata["source_sha"] != request["source_sha"]
        or metadata["data_producer_sha256"] != request["producer_sha"]
    ):
        raise ValueError("Worker image and request provenance disagree")
    result_key = run_key(run_id, "result.json")
    existing, _ = get_json(s3, bucket, result_key, optional=True)
    if existing is not None:
        return existing  # Task retry/explicit resume reuses this validated result.
    with lease(dynamo, table, f"worker:{request['kind']}", seconds=7200) as assert_lease:
        check = None
        if request.get("check_sources"):
            from src.maintenance.sources import check_sources

            namespace = "shadow" if request.get("mode") == "shadow" else "status"
            status_key = f"maintenance/{namespace}/source-check.json"
            previous, _ = get_json(s3, bucket, status_key, optional=True)
            check = check_sources(previous)
            put_json(s3, bucket, run_key(run_id, "sources.json"), check)
        if request["kind"] == "inference":
            result = produce_forecast(s3, bucket, request, root=root)
        elif request["kind"] == "weekly":
            result = prepare_release(s3, bucket, request, root=root)
        else:
            raise ValueError("Unknown worker kind")
        if check is not None:
            from src.maintenance.readiness import complete_report

            payload = None
            if request["kind"] == "inference":
                payload, _ = get_json(s3, bucket, result["forecast"]["key"])
            check = complete_report(check, root / "data/raw", payload=payload)
            check["data_release"] = result.get("data_release", request.get("data_release"))
            put_json(s3, bucket, run_key(run_id, "sources.json"), check)
            # A historical-only correction check cannot stand in for the daily
            # check of actual live/expert inputs.
            if request["kind"] == "inference":
                put_json(s3, bucket, status_key, check)
            if check["readiness"] == "blocked":
                raise ValueError("Required historical source schema/coverage checks failed")
        assert_lease()
        assert_models_current(s3, bucket, request["models"])
        result.update(run_id=run_id, kind=request["kind"], completed_at=now_iso(), request=request)
        put_json(s3, bucket, result_key, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-key", required=True)
    args = parser.parse_args()
    import boto3

    bucket, table = os.environ["FF_MODEL_S3_BUCKET"], os.environ["FF_MAINTENANCE_LOCK_TABLE"]
    region = os.environ["AWS_REGION"]
    s3 = boto3.client("s3", region_name=region)
    request, _ = get_json(s3, bucket, args.request_key)
    if args.request_key != run_key(request["run_id"], "request.json"):
        raise ValueError("Request key and run identity disagree")
    result = execute(
        s3,
        boto3.client("dynamodb", region_name=region),
        bucket=bucket,
        table=table,
        request=request,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "kind": result["kind"],
                "completed_at": result["completed_at"],
            }
        )
    )


if __name__ == "__main__":
    main()
