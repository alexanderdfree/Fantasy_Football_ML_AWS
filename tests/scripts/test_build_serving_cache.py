"""Import + guard smoke tests for the off-container serving-cache builder.

Operator CLI (src/scripts/build_serving_cache.py): an import-smoke test makes
signature/import drift fail the unit shard instead of surfacing only when the
refresh-splits workflow runs it. The no-bucket guard is exercised directly — it
returns before importing torch/serving, so it stays fast and dependency-light.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

_MODULE = "src.scripts.build_serving_cache"


def test_module_imports_and_exposes_main():
    mod = importlib.import_module(_MODULE)
    assert callable(mod.main)


def test_main_refuses_without_s3_bucket(monkeypatch):
    """No FF_MODEL_S3_BUCKET -> exit 1 before any S3 sync / heavy import."""
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    mod = importlib.import_module(_MODULE)
    assert mod.main() == 1
