import copy
import io
import json
from unittest.mock import Mock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.analysis.practice_cutoffs import load_snapshots
from src.artifacts.practice_archive import archive_refresh
from src.data.practice_reports import PracticeReport

pytestmark = pytest.mark.unit


@pytest.fixture
def inputs():
    time = "2026-09-23T12:00:00+00:00"
    report = PracticeReport(
        {"a": 1.0},
        {"fetched_at": time},
        [
            {
                "player_id": "a",
                "season": 2026,
                "week": 3,
                "coverage": "reported",
                "source": "NFL.com",
                "observed_at": time,
                "reported_at": None,
                "injury_descriptions": ["knee"],
                "practice_status": 1.0,
            }
        ],
    )
    forecast = {
        "available": True,
        "season": 2026,
        "week": 3,
        "generated_at": time,
        "input_signature": "model-and-input-identity",
        "scoring": {
            "ppr": [{"player_id": "a", "position": "RB", "team": "BAL", "ridge_pred": 10.0}]
        },
    }
    slate = pd.DataFrame({"recent_team": ["BAL"], "kickoff": ["2026-09-27T17:00:00Z"]})
    return report, forecast, slate


def test_complete_content_addressed_files_are_idempotent_and_do_not_mutate_forecast(
    tmp_path, inputs
):
    report, forecast, slate = inputs
    before = copy.deepcopy(forecast)
    path = archive_refresh(
        report, forecast, slate, directory=tmp_path, available_at=report.metadata["fetched_at"]
    )
    assert (
        archive_refresh(
            report, forecast, slate, directory=tmp_path, available_at=report.metadata["fetched_at"]
        )
        == path
    )
    assert len(list(tmp_path.rglob("*.json"))) == 1
    snapshot = next(load_snapshots(tmp_path))
    assert snapshot["practice"]["observations"][0]["reported_at"] is None
    assert snapshot["forecast"]["input_signature"] == before["input_signature"]
    assert forecast == before
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(load_snapshots(tmp_path))


def test_s3_uses_conditional_creation_and_checks_an_existing_copy(tmp_path, inputs):
    report, forecast, slate = inputs
    s3 = Mock()
    path = archive_refresh(
        report,
        forecast,
        slate,
        directory=tmp_path,
        bucket="test",
        s3=s3,
        available_at=report.metadata["fetched_at"],
    )
    assert s3.put_object.call_args.kwargs["IfNoneMatch"] == "*"
    assert s3.put_object.call_args.kwargs["Body"] == path.read_bytes()
    s3.put_object.side_effect = ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
    s3.get_object.return_value = {"Body": io.BytesIO(path.read_bytes())}
    archive_refresh(
        report,
        forecast,
        slate,
        directory=tmp_path,
        bucket="test",
        s3=s3,
        available_at=report.metadata["fetched_at"],
    )
    s3.get_object.return_value = {"Body": io.BytesIO(b"{}")}
    with pytest.raises(RuntimeError, match="disagrees"):
        archive_refresh(
            report,
            forecast,
            slate,
            directory=tmp_path,
            bucket="test",
            s3=s3,
            available_at=report.metadata["fetched_at"],
        )


def test_revision_creates_new_evidence_and_week_mismatch_fails(tmp_path, inputs):
    report, forecast, slate = inputs
    first = archive_refresh(report, forecast, slate, directory=tmp_path)
    original = first.read_bytes()
    report.observations[0]["injury_descriptions"] = ["illness"]
    second = archive_refresh(report, forecast, slate, directory=tmp_path)
    assert first != second
    assert first.read_bytes() == original
    assert json.loads(second.read_bytes())["practice"]["observations"][0][
        "injury_descriptions"
    ] == ["illness"]
    report.observations[0]["week"] = 2
    with pytest.raises(ValueError, match="different weeks"):
        archive_refresh(report, forecast, slate, directory=tmp_path)
