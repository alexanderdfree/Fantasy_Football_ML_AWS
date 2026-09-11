"""Bake immutable source/producer identity into the maintenance image and Lambda bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data.release import data_producer_hashes, producer_fingerprint


def build(source_sha: str, root=Path(".")) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("A complete source commit SHA is required")
    producers = data_producer_hashes(root)
    return {
        "source_sha": source_sha,
        "producer": producers,
        "data_producer_sha256": producer_fingerprint(producers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", default="maintenance-image.json")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build(args.source_sha), sort_keys=True))


if __name__ == "__main__":
    main()
