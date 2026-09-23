"""Write reviewable Batch registration/submission JSON; never calls AWS."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

NAME = "ff-cv-fixture-diagnostic"


def payloads(template: dict, image: str, release: str, run_id: str) -> dict:
    if not re.fullmatch(r".+/ff-training:[0-9a-f]{40}@sha256:[0-9a-f]{64}", image):
        raise ValueError("Diagnostic image must have a full Git tag and immutable digest")
    if template["jobDefinitionName"] != "ff-training-cpu-job":
        raise ValueError("Expected the production CPU template, not the GPU definition")
    if not re.fullmatch(r"[0-9a-f]{64}", release):
        raise ValueError("Frozen data release SHA-256 is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe diagnostic run ID")
    container = copy.deepcopy(template["containerProperties"])
    if any(r["type"] == "GPU" for r in container.get("resourceRequirements", [])):
        raise ValueError("Diagnostic CPU template unexpectedly requests a GPU")
    container["image"] = image
    container.pop("resourceRequirements", None)
    container["vcpus"], container["memory"] = 4, 7500
    environment = {
        "FF_DEVICE": "cpu",
        "FF_AMP_DTYPE": "fp32",
        "REQUIRE_GPU": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LGBM_N_JOBS": "1",
        "LOKY_MAX_CPU_COUNT": "4",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    }
    container["environment"] = [{"name": k, "value": v} for k, v in environment.items()]
    # ENTRYPOINT belongs to the disposable image; never inherit a production
    # training command that might publish models or benchmark history.
    container["command"] = []
    definition = {
        "jobDefinitionName": NAME,
        "type": template["type"],
        "containerProperties": container,
        "retryStrategy": {"attempts": 1},
        "timeout": {"attemptDurationSeconds": 1800},
    }
    for key in ("platformCapabilities", "propagateTags"):
        if key in template:
            definition[key] = template[key]
    result = {"definition.json": definition}
    for position in ("WR", "RB", "DST", "UNIT"):
        result[f"submit-{position.lower()}.json"] = {
            "jobName": f"cv-smoke-{position.lower()}-{run_id}",
            "jobQueue": "ff-cpu-training-queue",
            # The operator must replace this with the just-registered ARN;
            # it intentionally cannot submit against an unpinned latest name.
            "jobDefinition": "REPLACE_WITH_REGISTERED_DIAGNOSTIC_ARN",
            "retryStrategy": {"attempts": 1},
            "timeout": {"attemptDurationSeconds": 7200 if position == "UNIT" else 1800},
            "containerOverrides": {
                "vcpus": 4,
                "memory": 7500,
                "command": [
                    "--position",
                    position,
                    "--data-release",
                    release,
                    "--result-prefix",
                    f"diagnostics/cv-smoke/{run_id}",
                ],
                "environment": container["environment"],
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text())["jobDefinitions"][0]
    prepared = payloads(template, args.image, args.data_release, args.run_id)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in prepared.items():
        (args.output / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": list(prepared), "aws_calls": 0}))


if __name__ == "__main__":
    main()
