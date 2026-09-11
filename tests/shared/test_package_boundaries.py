"""Focused import-direction and generated dependency checks for new boundaries."""

import ast
import json
import os
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

import pytest

from src.packaging.dependencies import render_manifests

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_dependency_manifests_are_generated_from_one_source():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name, rendered in render_manifests(project).items():
        assert (ROOT / name).read_text() == rendered, f"Regenerate {name} from pyproject.toml"
    cpu = render_manifests(project)["requirements-dev.txt"]
    gpu = render_manifests(project)["requirements-gpu.txt"]
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in cpu
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in gpu
    version = project["tool"]["ffp"]["dependencies"]["torch-version"]
    assert f"torch=={version}\n" in cpu
    assert f"torch=={version}+cu130\n" in gpu


def test_separated_packages_do_not_import_execution_or_transport_layers():
    for package in ("evaluation", "contracts", "packaging"):
        for path in (ROOT / "src" / package).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for module in modules:
                    if module.startswith("src."):
                        assert module.startswith(f"src.{package}."), (
                            f"{path}: forbidden dependency {module}"
                        )


def test_evaluation_metrics_import_without_torch_plotting_or_analysis_cli():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.evaluation.metrics; assert 'torch' not in sys.modules; assert 'matplotlib.pyplot' not in sys.modules; assert 'src.analysis.cohort_analysis' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr[-2000:]


def test_experiment_metrics_do_not_reintroduce_analysis_entrypoint_dependency():
    for path in (ROOT / "src/tuning").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "src.analysis.cohort_analysis":
                assert not (
                    {alias.name for alias in node.names} & {"available_models", "per_model_metrics"}
                )


@pytest.mark.timeout(120)
def test_installed_wheel_contract_export_targets_requested_repository(tmp_path):
    from src.contracts.api import API_CONTRACT

    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
            str(wheelhouse),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    wheel = next(wheelhouse.glob("*.whl"))
    installed = tmp_path / "installed"
    # uv-managed macOS interpreters locate libpython relative to the executable;
    # copying that executable into the test venv breaks its dynamic-library path.
    venv.EnvBuilder(with_pip=True, system_site_packages=True, symlinks=os.name != "nt").create(
        installed
    )
    scripts = installed / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert install.returncode == 0, install.stderr[-2000:]
    repository = tmp_path / "consumer-repository"
    output = repository / "src/serving/frontend/src/api-contract.json"
    output.parent.mkdir(parents=True)
    command = scripts / ("ff-contract-export.exe" if os.name == "nt" else "ff-contract-export")
    for cwd, args in (
        (repository, []),
        (repository, ["--check"]),
        (tmp_path, ["--root", str(repository), "--check"]),
    ):
        result = subprocess.run(
            [str(command), *args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(output.read_text()) == API_CONTRACT


def test_serving_import_does_not_load_training_or_experiment_orchestration():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.serving.app; "
            "forbidden = [name for name in sys.modules "
            "if name.startswith(('src.tuning.', 'src.batch.', 'src.orchestration.')) "
            "or name in {'src.shared.pipeline', 'src.shared.training', 'src.training.effects'}]; "
            "assert not forbidden, forbidden",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr[-2000:]
