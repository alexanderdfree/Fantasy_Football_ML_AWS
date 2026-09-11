"""Release source order comes from actual Git/image ancestry, never finish time."""

import io
import json
import subprocess

import pytest
from botocore.exceptions import ClientError

from src.artifacts import source

pytestmark = pytest.mark.unit


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Key, Body, IfNoneMatch, **_):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def get_object(self, *, Key, **_):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def history(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", *args],
            cwd=tmp_path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "--initial-branch=main")
    git("commit", "--allow-empty", "-m", "root")
    parent = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-m", "newer")
    newer = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", newer)
    return tmp_path, git, parent, newer


def test_registration_order_is_ancestry_even_when_old_finishes_last(history, monkeypatch):
    repo, _, parent, newer = history
    s3 = MemoryS3()
    new = source.register_source(s3, "b", "models", newer, repo)
    old = source.register_source(s3, "b", "models", parent, repo)
    assert new == {"source_sha": newer, "source_order": 2, "lineage": [newer, parent]}
    assert old == {"source_sha": parent, "source_order": 1, "lineage": [parent]}
    monkeypatch.setattr(source, "image_source_sha", lambda: newer)
    assert source.load_source(s3, "b", "models", newer) == new
    assert source.read_source(s3, "b", "models", parent) == old
    with pytest.raises(RuntimeError, match="disagrees with the actual"):
        source.load_source(s3, "b", "models", parent)
    assert source.register_source(s3, "b", "models", newer, repo) == new


def test_registration_refuses_to_rewrite_an_immutable_record(history):
    repo, _, _, newer = history
    s3 = MemoryS3()
    key = source.source_key("models", newer)
    s3.objects[key] = b'{"source_order":999}'
    with pytest.raises(RuntimeError, match="disagrees with Git ancestry"):
        source.register_source(s3, "b", "models", newer, repo)
    assert s3.objects[key] == b'{"source_order":999}'


def test_branch_source_requires_sandbox_prefix(history):
    repo, git, parent, newer = history
    git("checkout", "-b", "experiment", parent)
    git("commit", "--allow-empty", "-m", "branch")
    branch = git("rev-parse", "HEAD")
    s3 = MemoryS3()
    for prefix in ("models", "/models/"):
        with pytest.raises(RuntimeError, match="first-parent history"):
            source.register_source(s3, "b", prefix, branch, repo)
    record = source.register_source(s3, "b", "experiments/branch", branch, repo)
    assert record["lineage"] == [branch, parent]
    assert newer not in record["lineage"]


def test_source_keys_follow_runtime_prefix_normalization():
    assert source.source_key("/models/", "a" * 40) == source.source_key("models", "a" * 40)


def test_shallow_history_cannot_establish_order(history):
    repo, _, _, newer = history
    (repo / ".git/shallow").write_text(newer + "\n")
    with pytest.raises(RuntimeError, match="full Git history"):
        source.register_source(MemoryS3(), "b", "models", newer, repo)


def test_runtime_request_cannot_relabel_baked_image(history, monkeypatch):
    repo, _, parent, newer = history
    monkeypatch.setenv("FF_TRAIN_GIT_SHA", parent)
    assert source.image_source_sha(root=repo) == newer
    (repo / ".training-source-sha").write_text(newer + "\n")
    assert source.image_source_sha(root=repo) == newer
    monkeypatch.setattr(source, "image_source_sha", lambda: newer)
    with pytest.raises(RuntimeError, match="disagrees with the actual"):
        source.load_source(MemoryS3(), "b", "models", parent)


@pytest.mark.parametrize("value", ["latest", "abcdef0", "z" * 40, None])
def test_source_identity_requires_full_git_sha(value):
    with pytest.raises(RuntimeError, match="full 40-character"):
        source.source_key("models", value)


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"source_sha": "a" * 40, "source_order": 2, "lineage": ["a" * 40]},
        {"source_sha": "a" * 40, "source_order": 2, "lineage": ["a" * 40, "a" * 40]},
        {"source_sha": "a" * 40, "source_order": 1, "lineage": [123]},
    ],
)
def test_malformed_source_fails_closed(record, monkeypatch):
    sha = "a" * 40
    s3 = MemoryS3()
    s3.objects[source.source_key("models", sha)] = json.dumps(record).encode()
    monkeypatch.setattr(source, "image_source_sha", lambda: sha)
    with pytest.raises(RuntimeError, match="Malformed"):
        source.load_source(s3, "b", "models", sha)


def test_missing_source_fails_before_training(monkeypatch):
    sha = "a" * 40
    monkeypatch.setattr(source, "image_source_sha", lambda: sha)
    with pytest.raises(RuntimeError, match="register the built image SHA"):
        source.load_source(MemoryS3(), "b", "models", sha)
