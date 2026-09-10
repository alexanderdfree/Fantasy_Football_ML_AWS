"""The actual review CLI must report ordered input/architecture list changes."""

import copy
import json

import pytest

from src.scripts import feature_manifest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "change", ["feature_order", "duplicate_layer", "feature_replacement", "unchanged"]
)
def test_diff_cli_reports_list_order_and_multiplicity(monkeypatch, capsys, change):
    original = feature_manifest.build_manifest()
    updated = copy.deepcopy(original)
    expected_path = "features"
    if change == "feature_order":
        features = updated["QB"]["features"]
        assert features[0] != features[1]
        features[0], features[1] = features[1], features[0]
    elif change == "duplicate_layer":
        layers = updated["QB"]["nn"]["backbone_layers"]
        layers.append(layers[-1])
        expected_path = "nn.backbone_layers"
    elif change == "feature_replacement":
        updated["QB"]["features"][0] += "_replacement"
    monkeypatch.setattr(feature_manifest, "build_manifest", lambda: updated)
    monkeypatch.setattr(feature_manifest, "_git_show_snapshot", lambda ref: original)
    assert feature_manifest.main(["--diff", "HEAD"]) == 0
    output = capsys.readouterr().out
    if change == "unchanged":
        assert "no feature-manifest changes" in output
    else:
        assert "no feature-manifest changes" not in output
        assert "[QB]" in output and f"{expected_path}:" in output


def test_snapshot_check_still_rejects_order_drift_and_identifies_it(tmp_path, monkeypatch, capsys):
    original = feature_manifest.build_manifest()
    updated = copy.deepcopy(original)
    updated["QB"]["features"].reverse()
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(feature_manifest, "build_manifest", lambda: updated)
    assert feature_manifest.main(["--manifest-path", str(path)]) == 1
    output = capsys.readouterr().err
    assert "FEATURE MANIFEST DRIFT" in output
    assert "features:" in output
