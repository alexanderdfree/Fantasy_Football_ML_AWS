"""Generate existing environment manifests from the canonical pyproject table."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def render_manifests(project: dict) -> dict[str, str]:
    config = project["tool"]["ffp"]["dependencies"]
    runtime = project["project"]["dependencies"]
    tools = project["dependency-groups"]["tools"]
    pins = {}
    for requirement in [*runtime, *tools]:
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group()
        if name in pins and pins[name] != requirement:
            raise ValueError(f"Conflicting requirements for {name}")
        pins[name] = requirement

    def group(name):
        return [pins[package] for package in config[name]]

    header = "# Generated from pyproject.toml; run python -m src.packaging.dependencies.\n"
    cpu = [f"--extra-index-url {config['cpu-index']}", f"torch=={config['torch-version']}"]
    cuda = [
        f"--extra-index-url {config['cuda-index']}",
        f"torch=={config['torch-version']}+{config['cuda-variant']}",
    ]
    rows = {
        "requirements.txt": runtime,
        "requirements-serving.txt": group("serving"),
        "requirements-dev.txt": ["-r requirements.txt", *cpu, *tools],
        "requirements-gpu.txt": ["-r requirements.txt", *cuda, *tools],
        "src/batch/requirements.txt": [*cuda, *group("batch")],
    }
    return {name: header + "\n".join(lines) + "\n" for name, lines in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository containing pyproject.toml"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    project = tomllib.loads((args.root / "pyproject.toml").read_text())
    stale = []
    for relative, rendered in render_manifests(project).items():
        path = args.root / relative
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                stale.append(relative)
        else:
            path.write_text(rendered)
    if stale:
        raise SystemExit("Stale generated dependency manifests: " + ", ".join(stale))


if __name__ == "__main__":
    main()
