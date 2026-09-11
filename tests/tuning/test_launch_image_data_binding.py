"""Operator entrypoints bind remote image, immutable revision, and compatible data."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data.release import DataReleaseError

pytestmark = pytest.mark.unit
IMAGE_A = "a" * 40
CHECKOUT_B = "b" * 40
RELEASE_A = "c" * 64
RELEASE_B = "d" * 64
LAUNCHERS = ("launch_tune", "launch_ablate_scheduler", "launch_ab", "launch_ablate")


def configure(monkeypatch, name):
    for name_ in ("FF_DATA_RELEASE", "FF_DATASET_ID", "FF_DATA_FORMAT"):
        monkeypatch.delenv(name_, raising=False)
    module = importlib.import_module(f"src.tuning.{name}")
    batch, s3 = MagicMock(), MagicMock()
    batch.submit_job.return_value = {"jobId": "pinned-job"}
    monkeypatch.setattr("boto3.client", lambda service, **kw: {"batch": batch, "s3": s3}[service])
    argv = [name, "--positions", "RB", "--wait", "false"]
    if name in {"launch_tune", "launch_ablate_scheduler"}:
        binding = {
            "image_sha": IMAGE_A,
            "gpu_definition": "ff-training-job:14",
            "cpu_definition": "",
        }
        resolver = MagicMock(return_value=binding)
        monkeypatch.setattr(module, "resolve_launch_binding", resolver)
        expected_definition = binding["gpu_definition"]
    else:
        argv += ["--image-sha", IMAGE_A, "--skip-image-check", "--seeds", "42"]
        monkeypatch.setattr(module, "_git_head_sha", lambda: CHECKOUT_B)
        monkeypatch.setattr(module, "resolve_job_definition", lambda sha, client: "ff-ab-job:14")
        resolver = MagicMock(return_value={"image_sha": IMAGE_A, "job_definition": "ff-ab-job:14"})
        monkeypatch.setattr(module, "resolve_definition", resolver)
        expected_definition = "ff-ab-job:14"
        if name == "launch_ab":
            argv += ["--spec", "src.tuning.ab_example"]
        else:
            argv += ["--mod", "example.ablation"]
            fake_module = SimpleNamespace(
                ABLATION_NAME="example",
                BASELINE="baseline",
                VARIANTS={"baseline": ()},
                print_summary=lambda *a: True,
            )
            monkeypatch.setattr(module, "load_ablation_module", lambda _: fake_module)
    monkeypatch.setattr(sys, "argv", argv)
    return module, batch, s3, resolver, expected_definition


@pytest.mark.parametrize("name", LAUNCHERS)
def test_actual_image_source_is_checked_then_frozen_binding_is_submitted(monkeypatch, name):
    module, batch, s3, resolver, definition = configure(monkeypatch, name)
    calls = []

    def pin(client, **kwargs):
        assert client is s3
        assert kwargs["source_ref"] == IMAGE_A  # remote A, never the launcher checkout B
        batch.submit_job.assert_not_called()
        calls.append(kwargs)
        monkeypatch.setenv("FF_DATA_RELEASE", RELEASE_A)
        # Another build can advance the mutable globals after compatibility was
        # checked. The already-resolved revision must still reach Batch.
        monkeypatch.setattr(module, "JOB_DEFINITION", "new-latest-job", raising=False)
        monkeypatch.setattr(module, "JOB_DEFINITION_REVISION", "99", raising=False)
        return RELEASE_A

    monkeypatch.setattr(module, "pin_data_release", pin)
    module.main()
    assert len(calls) == 1
    resolver.assert_called_once()
    if name in {"launch_tune", "launch_ablate_scheduler"}:
        assert resolver.call_args.kwargs["gpu_only"] is True
    call = batch.submit_job.call_args.kwargs
    assert call["jobDefinition"] == definition
    env = {e["name"]: e["value"] for e in call["containerOverrides"]["environment"]}
    assert env["FF_TRAIN_GIT_SHA"] == IMAGE_A
    assert env["FF_DATA_RELEASE"] == RELEASE_A
    assert env["FF_DATASET_ID"] == RELEASE_A
    assert env["FF_DATA_FORMAT"] == "data-release-v1"


@pytest.mark.parametrize("name", LAUNCHERS)
def test_incompatible_explicit_data_pin_prevents_every_submission(monkeypatch, name):
    module, batch, _s3, _resolver, _definition = configure(monkeypatch, name)
    monkeypatch.setenv("FF_DATA_RELEASE", RELEASE_B)

    def reject(_client, **kwargs):
        assert kwargs["source_ref"] == IMAGE_A
        raise DataReleaseError("Selected data release is incompatible with remote image A")

    monkeypatch.setattr(module, "pin_data_release", reject)
    with pytest.raises(DataReleaseError, match="incompatible"):
        module.main()
    batch.submit_job.assert_not_called()


@pytest.mark.parametrize("name", ["launch_ab", "launch_ablate"])
def test_ablation_job_definition_must_match_requested_image(monkeypatch, name):
    module, batch, _s3, resolver, _definition = configure(monkeypatch, name)
    resolver.return_value["image_sha"] = CHECKOUT_B
    pin = MagicMock()
    monkeypatch.setattr(module, "pin_data_release", pin)
    with pytest.raises(RuntimeError, match="differs from the requested source SHA"):
        module.main()
    pin.assert_not_called()
    batch.submit_job.assert_not_called()


@pytest.mark.parametrize("name", ["launch_ab", "launch_ablate"])
def test_collect_only_needs_no_image_or_data_preflight(monkeypatch, name):
    module, batch, s3, resolver, _definition = configure(monkeypatch, name)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--collect-only", "--run-id", "existing-run"])
    client = MagicMock(
        side_effect=lambda service, **kw: s3 if service == "s3" else pytest.fail(service)
    )
    monkeypatch.setattr("boto3.client", client)
    monkeypatch.setattr(module, "load_run_manifest", lambda *a, **kw: None)
    monkeypatch.setattr(module, "collect_results", lambda *a, **kw: [])
    report_name = "_aggregate_and_report" if name == "launch_ab" else "_report"
    monkeypatch.setattr(module, report_name, lambda *a, **kw: 0)
    pin = MagicMock()
    monkeypatch.setattr(module, "pin_data_release", pin)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    resolver.assert_not_called()
    pin.assert_not_called()
    batch.submit_job.assert_not_called()


@pytest.mark.parametrize("name", ["launch_ab", "launch_ablate"])
def test_ablation_dry_run_stays_offline(monkeypatch, name):
    module, _batch, _s3, resolver, _definition = configure(monkeypatch, name)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run"])
    monkeypatch.setattr("boto3.client", lambda *a, **kw: pytest.fail("dry-run contacted AWS"))
    pin = MagicMock()
    monkeypatch.setattr(module, "pin_data_release", pin)
    module.main()
    resolver.assert_not_called()
    pin.assert_not_called()
