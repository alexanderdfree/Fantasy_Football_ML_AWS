"""Export the shared wire definition used by the browser (or check freshness)."""

import argparse
import json
from pathlib import Path

from src.contracts.api import API_CONTRACT

OUTPUT = Path(__file__).resolve().parents[1] / "serving/frontend/src/api-contract.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository output root")
    args = parser.parse_args()
    output = args.root / "src/serving/frontend/src/api-contract.json"
    rendered = json.dumps(API_CONTRACT, indent=2, sort_keys=True) + "\n"
    if args.check:
        if output.read_text() != rendered:
            raise SystemExit("Browser API contract is stale: python -m src.contracts.export")
    else:
        output.write_text(rendered)


if __name__ == "__main__":
    main()
