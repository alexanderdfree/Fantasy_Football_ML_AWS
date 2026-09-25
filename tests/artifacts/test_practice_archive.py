import copy
import io
import json
from unittest.mock import Mock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.analysis.practice_cutoffs import load_snapshots
from src.artifacts.practice_archive import archive_refresh, build_cohort_context
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


def test_pregame_cohorts_use_prior_season_and_archived_reference_not_current_actuals():
    from src.shared.evaluation_cohorts import REFERENCE_VERSION

    prior = pd.DataFrame(
        {
            "player_id": [f"p{i:02d}" for i in range(26)],
            "position": "RB",
            "season": 2025,
            "week": 1,
            "rushing_yards": [i * 10.0 for i in range(26)],
            "season_type": "REG",
        }
    )
    for column in (
        "rushing_tds",
        "receiving_tds",
        "receiving_yards",
        "receptions",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ):
        prior[column] = 0.0
    current = prior.iloc[[0, 25]].copy()
    current["season"] = 2026
    current["week"] = 3
    current["rushing_yards"] = [99999, 0]  # These outcomes must be ignored.
    current["is_returning_from_absence"] = [0, 1]
    current["game_status"] = [1.0, 0.5]
    reference = current[["player_id", "position", "season", "week"]].assign(
        reference_rank=[1, 30], reference_version=REFERENCE_VERSION
    )
    frame = pd.concat([prior, current], ignore_index=True)
    context = build_cohort_context(frame, 2026, 3, reference=reference)
    labels = {row["player_id"]: row for row in context["players"]}
    assert labels["p00"]["elite_top24"] is False
    assert labels["p25"]["elite_top24"] is True
    assert labels["p00"]["weekly_reference_top24"] is True
    assert labels["p25"]["weekly_reference_top24"] is False
    assert labels["p25"]["returning"] is True
    missing = build_cohort_context(
        frame.drop(columns="sack_fumbles_lost"), 2026, 3, reference=reference.iloc[:0]
    )
    assert all(row["elite_top24"] is None for row in missing["players"])
    assert all(row["weekly_reference_top24"] is None for row in missing["players"])
