"""Artifact retention remains non-destructive until it coordinates with writers."""

from unittest.mock import Mock

import pytest

from src.shared.artifact_gc import prune

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("manifest", [None, {}, {"stable": {"key": "old"}, "history": ["old"]}])
def test_prune_retains_every_object_without_s3_requests(manifest):
    s3 = Mock()
    with pytest.warns(RuntimeWarning, match="pruning is disabled"):
        assert prune(s3, "bucket", "models", "QB", manifest) == []
    assert s3.mock_calls == []


def test_stale_publisher_cannot_delete_concurrent_promotion():
    # A captured its manifest before B promoted. Collection based on A's
    # snapshot formerly deleted B, despite the live pointer now naming B.
    objects = {"A": b"artifact A", "B": b"artifact B", "in-flight-C": b"partial"}
    s3 = Mock()
    s3.delete_objects.side_effect = lambda **_: objects.clear()
    stale_a = {"stable": {"key": "A"}, "history": ["A"]}
    live_b = {"stable": {"key": "B"}, "history": ["B", "A"]}
    with pytest.warns(RuntimeWarning, match="pruning is disabled"):
        assert prune(s3, "bucket", "models", "QB", stale_a) == []
    assert objects[live_b["stable"]["key"]] == b"artifact B"
    assert objects["in-flight-C"] == b"partial"
    assert s3.mock_calls == []
