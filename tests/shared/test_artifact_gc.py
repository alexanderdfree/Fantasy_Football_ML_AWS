"""Retention must never use a stale manifest to sweep concurrent uploads."""

from unittest.mock import Mock

import pytest

from src.shared.artifact_gc import prune

pytestmark = pytest.mark.unit


def key(name, pos="QB"):
    return f"models/{pos}/releases/history/{name}/model.tar.gz"


def test_only_explicit_retired_keys_are_deleted():
    s3 = Mock()
    s3.delete_objects.return_value = {}
    manifest = {"schema_version": 3, "history": [key("active")], "retired": [key("old")]}
    assert prune(s3, "b", "models", "QB", manifest) == [key("old")]
    s3.get_paginator.assert_not_called()
    s3.delete_objects.assert_called_once_with(
        Bucket="b", Delete={"Objects": [{"Key": key("old")}], "Quiet": True}
    )


def test_active_stable_and_other_namespaces_cannot_be_deleted():
    s3 = Mock()
    manifest = {
        "schema_version": 3,
        "stable": {"key": key("stable")},
        "retired": [key("stable"), key("other", "RB"), "models/QB/history/old/model.tar.gz"],
    }
    assert prune(s3, "b", "models", "QB", manifest) == []
    s3.delete_objects.assert_not_called()


def test_legacy_manifest_has_no_cleanup_authority():
    s3 = Mock()
    assert prune(s3, "b", "models", "QB", {"schema_version": 2, "history": []}) == []
    assert not s3.mock_calls


def test_delete_errors_are_reported():
    s3 = Mock()
    s3.delete_objects.return_value = {"Errors": [{"Key": key("old"), "Code": "AccessDenied"}]}
    with pytest.raises(RuntimeError, match="retention failed"):
        prune(s3, "b", "models", "QB", {"schema_version": 3, "retired": [key("old")]})
