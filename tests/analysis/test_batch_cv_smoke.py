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
