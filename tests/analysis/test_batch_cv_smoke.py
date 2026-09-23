"""No-fit checks for the disposable Batch-only CV test entrypoint."""

from __future__ import annotations

import pytest

from src.analysis import batch_cv_smoke as smoke
from src.analysis.prepare_batch_cv_smoke import payloads

pytestmark = pytest.mark.unit


def test_tiny_config_is_the_only_ignored_assignment():
    original = "POSITION_CONFIG = {'loss': 'huber'}\nCONFIG_TINY = {'epochs': 1}\n"
    tiny_changed = original.replace("'epochs': 1", "'epochs': 2")
    production_changed = original.replace("'loss': 'huber'", "'loss': 'mse'")
    assert smoke.without_tiny_hash(original) == smoke.without_tiny_hash(tiny_changed)
    assert smoke.without_tiny_hash(original) != smoke.without_tiny_hash(production_changed)


def test_dataset_producer_mismatch_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        smoke.validate_data_producers(
            {"producer": {"src/wr/config.py": "changed"}},
            {"baseline_data_producers": {"src/wr/config.py": "expected"}},
        )


def test_junit_counts_include_skips_and_errors(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        "<testsuites><testsuite><testcase/><testcase><skipped/></testcase>"
        "<testcase><failure/></testcase><testcase><error/></testcase></testsuite></testsuites>"
    )
    assert smoke.junit_counts(path) == {"tests": 4, "skipped": 1, "failures": 1, "errors": 1}


def test_runtime_rejects_local_fitting(monkeypatch):
    monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="require AWS Batch"):
        smoke.main(
            [
                "--position",
                "WR",
                "--data-release",
                "a" * 64,
                "--result-prefix",
                "diagnostics/cv-smoke/test",
            ]
        )


def test_dry_run_never_imports_boto3(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)
    assert (
        smoke.main(
            [
                "--position",
                "RB",
                "--data-release",
                "a" * 64,
                "--result-prefix",
                "diagnostics/cv-smoke/test",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "tests/rb/test_run_cv_pipeline.py" in capsys.readouterr().out


def test_prepared_jobs_keep_production_definition_untouched():
    template = {
        "jobDefinitionName": "ff-training-cpu-job",
        "type": "container",
        "containerProperties": {"image": "production", "vcpus": 4, "memory": 7500},
    }
    result = payloads(
        template, "example/ff-training:" + "a" * 40 + "@sha256:" + "b" * 64, "c" * 64, "test"
    )
    assert template["containerProperties"]["image"] == "production"
    assert result["definition.json"]["jobDefinitionName"] == "ff-cv-fixture-diagnostic"
    assert result["submit-wr.json"]["jobDefinition"] == "REPLACE_WITH_REGISTERED_DIAGNOSTIC_ARN"
    assert result["submit-rb.json"]["containerOverrides"]["command"][1] == "RB"
    assert result["submit-dst.json"]["containerOverrides"]["command"][1] == "DST"
    assert smoke.EXPECTED_TESTS == {"WR": 14, "RB": 11, "DST": 3}
    assert result["submit-unit.json"]["timeout"]["attemptDurationSeconds"] == 7200


def test_unit_allows_platform_skips_but_not_empty_or_failed_suites():
    counts = {"tests": 10, "failures": 0, "errors": 0, "skipped": 2}
    assert smoke.suite_passed("UNIT", 0, counts)
    assert not smoke.suite_passed("UNIT", 1, counts)
    assert not smoke.suite_passed("UNIT", 0, {**counts, "errors": 1})
    assert not smoke.suite_passed("UNIT", 0, {**counts, "tests": 2})
    assert not smoke.suite_passed("UNIT", 0, {})
    assert not smoke.suite_passed("WR", 0, {**counts, "tests": 14})


def test_test_process_cannot_inherit_production_credentials_or_publish(tmp_path):
    parent = {
        "AWS_ACCESS_KEY_ID": "real",
        "AWS_SECRET_ACCESS_KEY": "real",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/credentials",
        "AWS_WEB_IDENTITY_TOKEN_FILE": "/token",
        "AWS_PROFILE": "production",
        "FF_MODEL_S3_BUCKET": "production",
        "S3_BUCKET": "production",
        "FF_DEVICE": "cpu",
        "FF_AMP_DTYPE": "fp32",
    }
    env = smoke.isolated_test_environment(
        parent, tmp_path, "diagnostics/cv-smoke/test", target="UNIT"
    )
    assert parent["AWS_ACCESS_KEY_ID"] == "real"
    assert env["AWS_ACCESS_KEY_ID"] == env["AWS_SECRET_ACCESS_KEY"] == "testing"
    assert "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" not in env
    assert "AWS_WEB_IDENTITY_TOKEN_FILE" not in env and "AWS_PROFILE" not in env
    assert env["AWS_EC2_METADATA_DISABLED"] == "true"
    assert env["FF_MODEL_S3_BUCKET"] == ""
    assert "FF_S3_BUCKET" not in env and "S3_BUCKET" not in env
    assert "FF_MODEL_S3_PREFIX" not in env
    assert env["FF_BENCHMARK_SYNC_INTERVAL_S"] == "0"
    cv_env = smoke.isolated_test_environment(
        parent, tmp_path, "diagnostics/cv-smoke/test", target="WR"
    )
    assert cv_env["FF_MODEL_S3_PREFIX"] == "diagnostics/cv-smoke/test/unpublished-models"


def test_workflow_exports_only_tracked_files_and_sanitized_git(tmp_path):
    import subprocess

    workflow = (smoke.ROOT / ".github/workflows/batch-image.yml").read_text()
    section = workflow.split("- name: Export tracked diagnostic fixtures", 1)[1]
    script = section.split("run: |\n", 1)[1].split("\n      - name:", 1)[0]
    script = "\n".join(line[10:] for line in script.splitlines())
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.txt").write_text("tracked fixture\n")
    (tmp_path / "untracked-private.txt").write_text("not exported\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "http.extraHeader",
            "AUTHORIZATION: test-only",
        ],
        check=True,
    )
    subprocess.run(["bash", "-c", script], cwd=tmp_path, check=True)
    exported = tmp_path / ".diagnostic-checkout"
    assert (exported / "tracked.txt").is_file()
    assert not (exported / "untracked-private.txt").exists()
    for path in ("config", "logs", "hooks", "objects/info/alternates"):
        assert not (exported / ".git" / path).exists()
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"])
    assert subprocess.check_output(["git", "-C", str(exported), "rev-parse", "HEAD"]) == head
