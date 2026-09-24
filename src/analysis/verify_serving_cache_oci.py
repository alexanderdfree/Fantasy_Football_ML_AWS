"""One-off, fixed-source native serving-cache validation; no fitting or deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

SOURCE_SHA = "fe368b1ae6cadafd94bb67f7ffd92499266188bb"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect_image(args):
    assert platform.machine() in {"aarch64", "arm64"}, "Native ARM64 worker required"
    actual_sha = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual_sha == SOURCE_SHA
    assert not subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True
    ).strip(), "Source checkout changed"
    sources, dependencies, package_metadata = {}, {}, {}
    with tarfile.open(args.archive) as archive:

        def blob(descriptor):
            algorithm, checksum = descriptor["digest"].split(":", 1)
            assert algorithm == "sha256"
            data = archive.extractfile(f"blobs/sha256/{checksum}").read()
            assert sha(data) == checksum
            return data

        index = json.load(archive.extractfile("index.json"))
        descriptor = index["manifests"][0]
        manifest = json.loads(blob(descriptor))
        while "layers" not in manifest:
            descriptor = manifest["manifests"][0]
            manifest = json.loads(blob(descriptor))
        config = json.loads(blob(manifest["config"]))
        assert config["os"] == "linux" and config["architecture"] == "arm64"
        for layer in manifest["layers"]:
            algorithm, checksum = layer["digest"].split(":", 1)
            assert algorithm == "sha256"
            with (
                archive.extractfile(f"blobs/sha256/{checksum}") as stream,
                tarfile.open(fileobj=stream, mode="r|*") as members,
            ):
                for member in members:
                    name = member.name.removeprefix("./").lstrip("/")
                    path = PurePosixPath(name)
                    assert ".." not in path.parts
                    if path.name.startswith(".wh."):
                        removed = (
                            path.parent
                            if path.name == ".wh..wh..opq"
                            else path.with_name(path.name[4:])
                        )
                        for inventory in (sources, dependencies, package_metadata):
                            for key in list(inventory):
                                if key == str(removed) or key.startswith(str(removed) + "/"):
                                    del inventory[key]
                        continue
                    if (
                        not member.isfile()
                        or "__pycache__" in path.parts
                        or path.suffix in {".pyc", ".pyo"}
                    ):
                        continue
                    is_source = name.startswith("app/src/")
                    is_dependency = "/site-packages/" in name
                    if not (is_source or is_dependency):
                        continue
                    data = members.extractfile(member).read()
                    if is_source:
                        expected = args.source / name.removeprefix("app/")
                        assert expected.is_file() and sha(expected.read_bytes()) == sha(data), name
                        sources[name] = sha(data)
                    if is_dependency:
                        dependencies[name] = sha(data)
                    if name.endswith(".dist-info/METADATA"):
                        fields = {}
                        for line in data.decode(errors="replace").splitlines():
                            if not line:
                                break
                            if line.startswith(("Name: ", "Version: ")):
                                key, value = line.split(": ", 1)
                                fields[key] = value
                        package_metadata[name] = fields
    packages = {
        fields["Name"].lower().replace("_", "-"): fields["Version"]
        for fields in package_metadata.values()
    }
    assert {"flask", "gunicorn", "pandas", "numpy"} <= packages.keys()
    assert "torch" not in packages and "app/src/serving/app.py" in sources
    metadata = json.loads(args.metadata.read_text())
    assert metadata["containerimage.config.digest"] == manifest["config"]["digest"]
    materials = metadata["buildx.build.provenance"]["materials"]
    assert materials
    receipt = {
        "arm": args.arm,
        "source_sha": actual_sha,
        "native_machine": platform.machine(),
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "source_files": sources,
        "dependency_files": dependencies,
        "packages": packages,
        "runtime_config": {
            key: config.get(key) for key in ("os", "architecture", "variant", "config")
        },
        "build_materials": materials,
        "image_digest": metadata["containerimage.digest"],
        "config_digest": manifest["config"]["digest"],
        "created": config.get("created"),
        "history": config.get("history"),
        "rootfs": config.get("rootfs"),
        "smoke": "Successful Dockerfile build includes check-serving-runtime.py --require-absent import/API smoke",
        "excluded": [
            "Generated Python bytecode",
            "Build timestamps/history/provenance are recorded separately from runtime configuration",
        ],
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "source_sha": actual_sha,
                "source_files": len(sources),
                "dependency_files": len(dependencies),
                "packages": len(packages),
            }
        )
    )


def compare(args):
    receipts = {}
    for path in args.receipts.rglob("receipt-*.json"):
        value = json.loads(path.read_text())
        assert value["arm"] not in receipts
        receipts[value["arm"]] = value
    assert set(receipts) == {"legacy", "candidate"}
    left, right = receipts["legacy"], receipts["candidate"]
    for value in receipts.values():
        assert value["source_sha"] == SOURCE_SHA
        assert value["native_machine"] in {"aarch64", "arm64"}
        assert value["verifier_sha256"] == sha(Path(__file__).read_bytes())
    for field in (
        "source_sha",
        "source_files",
        "dependency_files",
        "packages",
        "runtime_config",
        "build_materials",
    ):
        assert left[field] == right[field], f"Parity failed: {field}"
    result = {
        "ok": True,
        "source_sha": SOURCE_SHA,
        "source_files_equal": True,
        "dependency_files_equal": True,
        "runtime_config_equal": True,
        "packages_equal": True,
        "build_materials_equal": True,
        "image_digest_equal": left["image_digest"] == right["image_digest"],
        "config_digest_equal": left["config_digest"] == right["config_digest"],
        "created": {arm: value["created"] for arm, value in receipts.items()},
        "limit": "Semantic inventory/config parity and import/API smoke; no whole-image bitwise or performance claim. Full receipts retain build metadata differences.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "compare"))
    parser.add_argument("--arm", choices=("legacy", "candidate"))
    for name in ("source", "archive", "metadata", "receipts", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    required = (
        ("source", "archive", "metadata", "arm", "output")
        if args.mode == "inspect"
        else ("receipts", "output")
    )
    if any(getattr(args, name) is None for name in required):
        parser.error(f"{args.mode} requires {', '.join(required)}")
    (inspect_image if args.mode == "inspect" else compare)(args)


if __name__ == "__main__":
    main()
