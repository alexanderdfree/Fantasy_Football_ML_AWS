"""Temporary, publication-free image cache comparison for the CI ROI draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def output(name, value):
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"{name}<<CI_VALUE\n{value}\nCI_VALUE\n")


def prepare(args):
    source = args.source.resolve()
    parent = git(source, "rev-parse", "HEAD")
    if args.sample:
        # Change a copied, non-executable source file on a disposable checkout.
        # Record the actual resulting commit rather than mislabeling an image.
        (source / "src/.ci-cache-probe").write_text("warm source-only build\n")
        git(source, "add", "src/.ci-cache-probe")
        git(
            source,
            "-c",
            "user.name=CI cache benchmark",
            "-c",
            "user.email=ci-cache@example.invalid",
            "commit",
            "-qm",
            "Local cache probe",
        )
    sha = git(source, "rev-parse", "HEAD")
    scope = f"ci-roi-{os.environ['GITHUB_RUN_ID']}-{args.variant}-{args.image}"
    cache_from = f"type=gha,version=2,scope={scope}"
    cache_to = f"{cache_from},mode=max"
    if args.variant == "baseline":
        cache_from += "\ntype=local,src=/tmp/.buildx-cache"
        cache_to += "\ntype=local,dest=/tmp/.buildx-cache-new,mode=max"
    output("source_sha", sha)
    output("cache_from", cache_from)
    output("cache_to", cache_to)
    Path("measurement.json").write_text(
        json.dumps(
            {
                "image": args.image,
                "variant": args.variant,
                "sample": args.sample,
                "parent_sha": parent,
                "source_sha": sha,
                "run_id": os.environ["GITHUB_RUN_ID"],
                "runner_arch": os.environ["RUNNER_ARCH"],
                "export": "local OCI; excludes production ECR push and deployment",
            },
            indent=2,
        )
    )


def inspect_image(args):
    result = json.loads(Path("measurement.json").read_text())
    packages = {}
    source_files = {}
    stamped = None
    root = "opt/ml/code/" if result["image"] == "training" else "app/"
    with tarfile.open(args.archive) as archive:

        def read_json(name):
            with archive.extractfile(name) as stream:
                return json.load(stream)

        descriptor = read_json("index.json")["manifests"][0]
        while True:
            manifest = read_json("blobs/" + descriptor["digest"].replace(":", "/"))
            if "layers" in manifest:
                break
            descriptor = manifest["manifests"][0]
        for layer in manifest["layers"]:
            with (
                archive.extractfile("blobs/" + layer["digest"].replace(":", "/")) as stream,
                tarfile.open(fileobj=stream, mode="r|*") as entries,
            ):
                for member in entries:
                    if not member.isfile():
                        continue
                    name = member.name.removeprefix("./").lstrip("/")
                    is_metadata = name.endswith(".dist-info/METADATA")
                    is_source = name.startswith(root + "src/") and "__pycache__" not in name
                    is_stamp = name == root + ".training-source-sha"
                    if not (is_metadata or is_source or is_stamp):
                        continue
                    data = entries.extractfile(member).read()
                    if is_metadata:
                        fields = {}
                        for line in data.decode(errors="replace").splitlines():
                            if not line:
                                break
                            if line.startswith(("Name: ", "Version: ")):
                                key, value = line.split(": ", 1)
                                fields[key] = value
                        if "Name" in fields and "Version" in fields:
                            packages[fields["Name"].lower().replace("_", "-")] = fields["Version"]
                    if is_source:
                        source_files[name.removeprefix(root)] = hashlib.sha256(data).hexdigest()
                    if is_stamp:
                        stamped = data.decode().strip()
    assert "src/serving/app.py" in source_files
    if result["image"] == "training":
        assert stamped == result["source_sha"], (stamped, result["source_sha"])
        assert {"torch", "lightgbm", "scikit-learn"} <= packages.keys()
    else:
        assert "torch" not in packages
        assert {"flask", "gunicorn"} <= packages.keys()
    result.update(
        packages=packages,
        source_files=source_files,
        source_stamp=stamped,
        archive_bytes=args.archive.stat().st_size,
        smoke="Dockerfile smoke passed; final exported contents and source stamp verified",
    )
    Path("measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "image",
                    "variant",
                    "sample",
                    "source_sha",
                    "source_stamp",
                    "archive_bytes",
                    "smoke",
                )
            }
        )
    )


def reject_invalid_provenance(args):
    # The small metadata target must still reject invalid provenance itself.
    for invalid in ("", "not-a-full-sha"):
        completed = subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--target",
                "source-metadata",
                "--build-arg",
                f"TRAIN_GIT_SHA={invalid}",
                "--file",
                str(args.source / "src/batch/Dockerfile.train"),
                str(args.source),
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "AssertionError: TRAIN_GIT_SHA must be the full built Git SHA" in completed.stderr
    print("Missing and malformed provenance rejected by the Docker build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "inspect", "reject-invalid"))
    parser.add_argument("--source", type=Path, default=Path("source"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--image", choices=("training", "serving"))
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    {"prepare": prepare, "inspect": inspect_image, "reject-invalid": reject_invalid_provenance}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
