"""The temporary image comparison rejects semantic drift, without building images."""

import argparse
import copy
import json
from pathlib import Path

import pytest

from src.analysis import verify_serving_cache_oci as verifier

pytestmark = pytest.mark.unit


def receipts(tmp_path, changed=None):
    base = {
        "source_sha": verifier.SOURCE_SHA,
        "native_machine": "aarch64",
        "verifier_sha256": verifier.sha(Path(verifier.__file__).read_bytes()),
        "source_files": {"app/src/serving/app.py": "source"},
        "dependency_files": {"site-packages/package.py": "dependency"},
        "packages": {"flask": "3.1.3"},
        "runtime_config": {
            "config": {"Cmd": ["gunicorn"], "Env": ["FF_ALLOW_RUNTIME_INFERENCE=0"]}
        },
        "build_materials": [{"digest": "base"}],
    }
    for arm in ("legacy", "candidate"):
        item = copy.deepcopy(base)
        item.update(
            arm=arm,
            image_digest=f"image-{arm}",
            config_digest=f"config-{arm}",
            created=f"time-{arm}",
        )
        if arm == "candidate" and changed:
            item.update(changed)
        (tmp_path / f"receipt-{arm}.json").write_text(json.dumps(item))
    return argparse.Namespace(receipts=tmp_path, output=tmp_path / "parity.json")


def test_equivalent_runtime_records_keep_different_build_metadata_explicit(tmp_path):
    args = receipts(tmp_path)
    verifier.compare(args)
    result = json.loads(args.output.read_text())
    assert result["ok"] and result["runtime_config_equal"]
    assert not result["image_digest_equal"] and not result["config_digest_equal"]
    assert result["created"] == {"legacy": "time-legacy", "candidate": "time-candidate"}


@pytest.mark.parametrize(
    "changed",
    [
        {"source_sha": "unreviewed-source"},
        {"native_machine": "x86_64"},
        {"verifier_sha256": "other-verifier"},
        {"source_files": {"app/src/serving/app.py": "changed"}},
        {"dependency_files": {"site-packages/package.py": "changed"}},
        {"packages": {"flask": "different"}},
        {"runtime_config": {"config": {"Cmd": ["other-process"]}}},
        {"build_materials": [{"digest": "different-base"}]},
    ],
)
def test_semantic_or_identity_drift_never_publishes_success(tmp_path, changed):
    args = receipts(tmp_path, changed)
    with pytest.raises(AssertionError):
        verifier.compare(args)
    assert not args.output.exists()
