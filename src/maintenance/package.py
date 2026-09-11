"""Build the control Lambda zip with matching immutable producer provenance."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.maintenance.image_metadata import build

CONTROL_FILES = (
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/release.py",
    "src/artifacts/__init__.py",
    "src/artifacts/model_sync.py",
    "src/artifacts/serving_snapshot.py",
    "src/artifacts/deployment.py",
    "src/scripts/__init__.py",
    "src/scripts/advance_data_release.py",
    "src/scripts/wait_data_release.py",
    "src/maintenance/__init__.py",
    "src/maintenance/control.py",
    "src/maintenance/storage.py",
    "src/maintenance/coordination.py",
)


def package(source_sha, output, *, root=Path(".")):
    metadata = build(source_sha, root)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in CONTROL_FILES:
            archive.write(root / name, name)
        archive.writestr("maintenance-image.json", json.dumps(metadata, sort_keys=True))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = package(args.source_sha, args.output)
    print(
        json.dumps(
            {"source_sha": metadata["source_sha"], "producer_sha": metadata["data_producer_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
