"""CI image metadata preserves provenance and branch publication boundaries."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _steps(workflow, job):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return document["jobs"][job]["steps"]


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/codex/test-build"])
@pytest.mark.parametrize("pull_through", [True, False])
def test_training_metadata_pins_checkout_and_keeps_branch_off_latest(tmp_path, ref, pull_through):
    steps = _steps("batch-image.yml", "build-and-push")
    metadata = next(step for step in steps if step.get("id") == "build")
    build = next(step for step in steps if step.get("uses", "").startswith("docker/build-push"))
    register = next(
        step for step in steps if step.get("name") == "Register new job definition revision"
    )
    assert steps.index(metadata) < steps.index(build) < steps.index(register)
    assert register["env"]["IMAGE_URI"] == "${{ steps.build.outputs.image_uri }}"
    assert "github.ref == 'refs/heads/main'" in register["if"]
    assert build["with"]["tags"] == "${{ steps.build.outputs.tags }}"
    assert "TRAIN_GIT_SHA=${{ steps.build.outputs.source_sha }}" in build["with"]["build-args"]
    assert (
        "PULL_THROUGH_PREFIX=${{ steps.build.outputs.pull_through_prefix }}"
        in build["with"]["build-args"]
    )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "source",
    )
    sha = git("rev-parse", "HEAD")
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    (bin_path / "aws").write_text("#!/bin/sh\necho 1\n" if pull_through else "#!/bin/sh\nexit 1\n")
    (bin_path / "aws").chmod(0o755)
    output = tmp_path / "outputs"
    env = dict(
        os.environ,
        PATH=f"{bin_path}:{os.environ['PATH']}",
        ECR_REGISTRY="registry.example",
        ECR_REPOSITORY="training",
        IMAGE_TAG="b" * 40,
        TRAIN_GIT_SHA="not-the-checkout",
        GITHUB_REF=ref,
        GITHUB_OUTPUT=str(output),
        AWS_REGION="us-east-1",
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", metadata["run"]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    values, tags = output.read_text().split("tags<<TAGS\n")
    values = dict(line.split("=", 1) for line in values.splitlines())
    assert values["source_sha"] == sha
    assert values["image_uri"] == "registry.example/training:" + "b" * 40
    assert values["pull_through_prefix"] == ("registry.example/dockerhub/" if pull_through else "")
    expected_tags = [values["image_uri"]]
    if ref == "refs/heads/main":
        expected_tags.append("registry.example/training:latest")
    assert tags.splitlines() == [*expected_tags, "TAGS"]


def test_image_caches_do_not_share_architecture_or_export_twice():
    scopes = set()
    for workflow, job, platform in [
        ("batch-image.yml", "build-and-push", "linux/amd64"),
        ("deploy.yml", "deploy", "linux/arm64"),
    ]:
        steps = _steps(workflow, job)
        build = next(step for step in steps if step.get("uses", "").startswith("docker/build-push"))
        options = build["with"]
        assert options["platforms"] == platform
        assert options["context"] == "."
        assert options["push"] is True
        assert options["provenance"] is False
        cache_from = dict(part.split("=", 1) for part in options["cache-from"].split(","))
        cache_to = dict(part.split("=", 1) for part in options["cache-to"].split(","))
        assert cache_from["type"] == cache_to["type"] == "gha"
        assert cache_from["version"] == cache_to["version"] == "2"
        assert cache_to["mode"] == "max"
        assert cache_from["scope"] == cache_to["scope"]
        assert cache_from["scope"].endswith(platform.replace("/", "-"))
        scopes.add(cache_from["scope"])
        assert all(not step.get("uses", "").startswith("actions/cache") for step in steps)
    assert len(scopes) == 2
