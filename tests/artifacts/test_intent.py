"""Delayed jobs and retries cannot relabel older inputs as newer publication intent."""

import io
import json

import pytest
from botocore.exceptions import ClientError

from src.artifacts import intent

pytestmark = pytest.mark.unit
SHA = "a" * 40
DATA_A, DATA_B = "b" * 64, "c" * 64


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.revision = 0
        self.interleave = None

    def get_object(self, *, Key, **_):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, etag = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, *, Key, Body, IfMatch=None, IfNoneMatch=None, **_):
        if self.interleave is not None:
            callback, self.interleave = self.interleave, None
            callback()
        current = self.objects.get(Key)
        if (IfNoneMatch == "*" and current is not None) or (
            IfMatch is not None and (current is None or current[1] != IfMatch)
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.revision += 1
        self.objects[Key] = Body, str(self.revision)


def reserve(s3, run="a", data=DATA_A, position="QB", sha=SHA):
    return intent.reserve_intent(s3, "bucket", "models", position, sha, data, run)


def valid(s3, descriptor):
    return intent.validate_intent(s3, "bucket", "models", descriptor)


def test_older_dataset_completion_cannot_become_latest():
    s3 = MemoryS3()
    old = reserve(s3, "old", DATA_A)
    new = reserve(s3, "new", DATA_B)
    assert old["sequence"] == 1 and new["sequence"] == 2
    assert not valid(s3, old)
    assert valid(s3, new)
    # A restarted old workflow keeps its sequence even after a newer dispatch.
    assert reserve(s3, "old", DATA_A) == old
    assert not valid(s3, old)
    assert s3.revision == 2


def test_run_id_cannot_be_reused_for_different_data():
    s3 = MemoryS3()
    first = reserve(s3)
    with pytest.raises(RuntimeError, match="different inputs"):
        reserve(s3, data=DATA_B)
    assert valid(s3, first)


def test_competing_reservations_retry_without_losing_a_binding():
    s3 = MemoryS3()
    other = []
    s3.interleave = lambda: other.append(reserve(s3, "other", DATA_B))
    ours = reserve(s3, "ours", DATA_A)
    assert other[0]["sequence"] == 1
    assert ours["sequence"] == 2
    assert not valid(s3, other[0]) and valid(s3, ours)


def test_duplicate_concurrent_run_gets_one_reservation():
    s3 = MemoryS3()
    other = []
    s3.interleave = lambda: other.append(reserve(s3, "same", DATA_A))
    ours = reserve(s3, "same", DATA_A)
    assert ours == other[0]
    assert ours["sequence"] == 1
    assert s3.revision == 1


def test_intents_are_separate_per_position_and_source():
    s3 = MemoryS3()
    qb = reserve(s3)
    rb = reserve(s3, position="RB")
    other_source = reserve(s3, sha="d" * 40)
    assert all(valid(s3, d) and d["sequence"] == 1 for d in (qb, rb, other_source))
    assert intent.intent_key("/models/", "QB", SHA) == intent.intent_key("models", "QB", SHA)


def test_retry_keeps_pre_training_revision_after_manual_rollback():
    s3 = MemoryS3()
    old = intent.reserve_intent(
        s3, "bucket", "models", "QB", SHA, DATA_A, "run", publication_revision="before"
    )
    retry = intent.reserve_intent(
        s3, "bucket", "models", "QB", SHA, DATA_A, "run", publication_revision="rollback"
    )
    assert retry == old
    assert retry["publication_revision"] == "before"
    assert s3.revision == 1


def test_legacy_data_is_explicitly_unidentified():
    s3 = MemoryS3()
    descriptor = reserve(s3, data=None)
    assert descriptor["dataset_id"] is None
    assert valid(s3, descriptor)


def test_unregistered_and_forged_descriptors_fail_closed():
    s3 = MemoryS3()
    descriptor = reserve(s3)
    for changes in ({"dataset_id": DATA_B}, {"run_id": "unknown"}, {"sequence": 999}):
        with pytest.raises(RuntimeError, match="immutable binding"):
            valid(s3, {**descriptor, **changes})
    with pytest.raises(RuntimeError, match="immutable binding"):
        valid(MemoryS3(), descriptor)
    with pytest.raises(RuntimeError, match="integer sequence"):
        valid(s3, {**descriptor, "sequence": True})


def test_corrupt_counter_is_not_overwritten():
    s3 = MemoryS3()
    descriptor = reserve(s3)
    key = intent.intent_key("models", "QB", SHA)
    ledger = json.loads(s3.objects[key][0])
    ledger["latest_sequence"] = 20
    corrupt = json.dumps(ledger).encode()
    s3.objects[key] = corrupt, "broken"
    with pytest.raises(RuntimeError, match="Malformed"):
        reserve(s3, "next")
    with pytest.raises(RuntimeError, match="Malformed"):
        valid(s3, descriptor)
    assert s3.objects[key][0] == corrupt
