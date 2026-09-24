import copy
import json
from unittest.mock import Mock

import pytest

from src.tuning import campaign
from src.tuning.campaign_contracts import identity, validate, work_units
from src.tuning.campaign_io import Journal, local_lock

pytestmark = pytest.mark.unit


def spec():
    return validate(
        {
            "version": 1,
            "id": "example",
            "steps": [
                {
                    "id": "ab",
                    "kind": "ab",
                    "spec": "src.tuning.ab_example",
                    "positions": ["RB"],
                    "options": {"seeds": [42], "stacked_seeds": False},
                },
                {
                    "id": "nn",
                    "kind": "nn_tune",
                    "positions": ["RB", "WR"],
                    "options": {"n_trials": 1},
                },
                {
                    "id": "trees",
                    "kind": "lgbm_tune",
                    "positions": ["RB"],
                    "options": {"n_trials": 1},
                },
                {"id": "bench", "kind": "benchmark", "positions": ["WR"]},
            ],
        }
    )


def manifest():
    value = {
        "id": "example",
        "spec": spec(),
        "backend": "batch",
        "bucket": "bucket",
        "image_uri": "registry/image@sha256:" + "a" * 64,
        "units": work_units(spec(), "batch"),
    }
    value["manifest_id"] = identity(value)
    return value


def test_campaign_groups_work_by_position_and_resource_without_extra_allocations():
    units = work_units(spec(), "batch")
    assert [(u["id"], u["steps"]) for u in units] == [
        ("RB-gpu", ["ab", "nn"]),
        ("WR-gpu", ["nn", "bench"]),
        ("RB-cpu", ["trees"]),
    ]
    assert len(work_units(spec(), "local")) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(id="../escape"),
        lambda s: s["steps"][0].update(options={"command": "anything"}),
        lambda s: s["steps"][0].update(env={"FF_MODEL_S3_PREFIX": "models"}),
        lambda s: s["steps"][1].update(options={"n_trials": 0}),
        lambda s: s["steps"][0].update(options={"seeds": [42, 42]}),
        lambda s: s["steps"].append(copy.deepcopy(s["steps"][0])),
        lambda s: s.update(data_prefix="mutable/other"),
    ],
)
def test_invalid_campaign_fails_before_submission(change):
    value = spec()
    change(value)
    with pytest.raises(ValueError):
        validate(value)


def test_journal_rejects_identity_changes_and_stale_writers(tmp_path):
    journal = Journal(tmp_path)
    journal.immutable("manifest.json", {"id": "one"})
    journal.immutable("manifest.json", {"id": "one"})
    with pytest.raises(ValueError, match="identity changed"):
        journal.immutable("manifest.json", {"id": "two"})
    token = journal.write("progress.json", {"state": "RUNNING"})
    journal.write("progress.json", {"state": "SUCCEEDED"}, token)
    with pytest.raises(RuntimeError, match="concurrently"):
        journal.write("progress.json", {"state": "FAILED"}, token)


def test_local_campaign_has_one_active_controller(tmp_path):
    with (
        local_lock(tmp_path),
        pytest.raises(RuntimeError, match="already running"),
        local_lock(tmp_path),
    ):
        pytest.fail("lock was acquired twice")


def test_batch_resume_attaches_active_jobs_and_reuses_pinned_definition(tmp_path, monkeypatch):
    journal = Journal(tmp_path)
    batch = Mock()
    batch.submit_job.side_effect = [{"jobId": f"job-{i}"} for i in range(4)]
    batch.describe_jobs.side_effect = lambda jobs: {
        "jobs": [{"jobId": jobs[0], "status": "RUNNING"}]
    }
    definition = Mock(side_effect=lambda batch, resource, image: f"pinned-{resource}:1")
    monkeypatch.setattr(campaign, "_definition", definition)
    original = campaign.submit_units(manifest(), journal, batch)
    assert len(original) == 3
    assert definition.call_count == 2
    assert campaign.submit_units(manifest(), journal, batch, resume=True) == original
    assert batch.submit_job.call_count == 3
    monkeypatch.setattr(campaign, "_unit_complete", lambda unit, journal: True)
    batch.describe_jobs.side_effect = lambda jobs: {
        "jobs": [{"jobId": jobs[0], "status": "FAILED" if jobs[0] == "job-0" else "SUCCEEDED"}]
    }
    again = campaign.submit_units(manifest(), journal, batch, resume=True)
    assert again["RB-gpu"] == "job-3"
    assert batch.submit_job.call_args.kwargs["jobDefinition"] == "pinned-gpu:1"
    assert definition.call_count == 2


def test_uncertain_submission_is_not_blindly_duplicated(tmp_path, monkeypatch):
    journal = Journal(tmp_path)
    value = manifest()
    journal.write(
        "units/RB-gpu/submission.json",
        {"manifest_id": value["manifest_id"], "attempt": 1, "job_name": "pending", "job_id": None},
    )
    batch = Mock()
    monkeypatch.setattr(campaign, "_find_submitted", lambda *args: None)
    with pytest.raises(RuntimeError, match="uncertain"):
        campaign.submit_units(value, journal, batch, resume=True)
    batch.submit_job.assert_not_called()


def test_dry_run_does_not_create_aws_clients(tmp_path, monkeypatch, capsys):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(spec()))
    monkeypatch.setattr(campaign, "resolve_workloads", lambda value, backend: value)
    client = Mock(side_effect=AssertionError("AWS called"))
    monkeypatch.setattr(campaign, "_clients", client)
    assert campaign.main(["--file", str(path), "--backend", "batch", "--dry-run"]) == 0
    assert not json.loads(capsys.readouterr().out)["resolved_image_and_dataset"]
    client.assert_not_called()
