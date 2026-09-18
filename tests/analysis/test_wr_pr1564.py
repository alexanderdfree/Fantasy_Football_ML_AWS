"""No-fit checks for historical-input identity and intervention isolation."""

import hashlib
import io

import pandas as pd
import pytest

from src.analysis.wr_pr1564_inputs import apply_depth, selected_inventory
from src.analysis.wr_pr1564_recovery import recover_object
from src.analysis.wr_pr1564_report import load, shapley
from src.tuning.ab_wr_pr1564 import _immutable_put, _metrics

pytestmark = pytest.mark.unit


class Source:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def get_object(self, **kwargs):
        self.requests.append(kwargs)
        return {"Body": io.BytesIO(self.content), "VersionId": "archived-version"}


def test_recovery_pins_version_and_rejects_changed_archive(tmp_path):
    original = b"archived input"
    task = {
        "key": "data/train.parquet",
        "version": "old-version",
        "sha256": hashlib.sha256(original).hexdigest(),
        "bytes": len(original),
        "destination": "baseline/train.parquet",
    }
    source = Source(b"newer bytes")
    record = recover_object(source, "bucket", task, tmp_path)
    assert not record["ok"]
    assert source.requests[0]["VersionId"] == "old-version"
    assert not (tmp_path / task["destination"]).exists()
    source.content = original
    record = recover_object(source, "bucket", task, tmp_path)
    assert record["ok"]
    assert (tmp_path / task["destination"]).read_bytes() == original


def test_depth_intervention_does_not_change_training_or_other_positions():
    frame = pd.DataFrame(
        {
            "player_id": ["a", "a", "b", "c"],
            "season": [2024, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "position": ["WR", "WR", "QB", "WR"],
            "depth_chart_rank": [2.0, 2.0, 2.0, 2.0],
            "is_top_available": [1.0, 0.0, 0.0, 1.0],
        }
    )
    lookup = pd.Series([1.0], index=pd.MultiIndex.from_tuples([("a", 2025, 1)]))
    result = apply_depth(frame, lookup)
    assert result.depth_chart_rank.tolist() == [2.0, 1.0, 2.0, -1.0]
    pd.testing.assert_frame_equal(
        result.drop(columns="depth_chart_rank"), frame.drop(columns="depth_chart_rank")
    )
    assert frame.depth_chart_rank.eq(2).all()


def test_inventory_includes_key_features_and_added_rows():
    left = pd.DataFrame(
        {"player_id": ["a"], "season": [2025], "week": [1], "depth_chart_rank": [3.0]}
    )
    right = pd.concat(
        [left.assign(depth_chart_rank=1), left.assign(player_id="b")], ignore_index=True
    )
    result = selected_inventory(left, right, ["week", "depth_chart_rank"])
    assert result["changed"] == {"depth_chart_rank": 1}
    assert result["added"] == [["b", 2025, 1]]


def test_metrics_keep_fixed_truth_and_subset():
    frame = pd.DataFrame({"actual": [0.0, 10.0, 1000.0]})
    for family in ("ridge", "nn", "attn_nn", "lgbm"):
        frame[f"pred_{family}_total"] = [2.0, 8.0, 0.0]
    block = _metrics(frame, pd.Series([True, True, False]))
    assert block["n"] == 2
    assert all(
        values == {"mae": 2.0, "rmse": 2.0, "bias": 0.0} for values in block["models"].values()
    )


def test_existing_evidence_cannot_be_overwritten():
    class Conflict(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class Existing(Source):
        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"
            raise Conflict()

    source = Existing(b"saved")
    _immutable_put(source, "bucket", "key", b"saved")
    with pytest.raises(ValueError, match="Refusing to replace"):
        _immutable_put(source, "bucket", "key", b"different")


def test_attribution_allocates_interactions_and_reconciles():
    import itertools

    values = {
        (r, a, d): 3 * r + 2 * a + 4 * d + 6 * r * d
        for r, a, d in itertools.product((0, 1), repeat=3)
    }
    assert shapley(values) == {"source": 6.0, "availability": 2.0, "depth": 7.0}


def test_incomplete_experiment_cannot_produce_a_conclusion(tmp_path):
    with pytest.raises(ValueError, match="Incomplete factorial"):
        load(tmp_path)
