"""Read-only development reporting uses synthetic manifests, never fitted models."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.analysis.model_default_repair_report import ORIGINS, SEEDS, VARIANTS, main, summarize
from src.prediction.bundle import canonical_json

pytestmark = pytest.mark.unit


def records():
    rows = []
    for origin in ORIGINS:
        for seed in SEEDS:
            for variant in VARIANTS["numerical"]:
                metrics = {
                    f"{cohort}:{family}": {
                        "mae": 2.0 - (variant != "baseline" and family in ("nn", "attn_nn")) * 0.1,
                        "rmse": 3.0 - (variant != "baseline" and family in ("nn", "attn_nn")) * 0.1,
                        "n": 24,
                    }
                    for cohort in ("all", "elite_top24")
                    for family in ("ridge", "lgbm", "nn", "attn_nn")
                }
                rows.append(
                    {
                        "position": "WR",
                        "origin": origin,
                        "seed": seed,
                        "variant": variant,
                        "source_sha": "a" * 40,
                        "data_release": "d" * 64,
                        "gpu": "L4",
                        "tf32_matmul": True,
                        "execution": {
                            "seed": seed,
                            "regime": "eager",
                            "device": "cuda:0",
                            "overrides": {"FF_AMP_DTYPE": "fp32"},
                            "run_id": variant,
                        },
                        "prepared_hashes": {"X": "hash"},
                        "row_hash": "rows",
                        "truth_hash": "truth",
                        "batch_job_id": "batch",
                        "inference_parity_passed": True,
                        "trainers": [
                            {"family": family, "device": "cuda:0", "amp": False, "graph": True}
                            for family in ("nn", "attn_nn")
                        ],
                        "metrics": metrics,
                        "cohorts": {
                            "elite_top24": {
                                "status": "available",
                                "cohort_hash": "elite",
                                "n": 24,
                                "models": {
                                    family: metrics[f"elite_top24:{family}"]
                                    for family in ("nn", "attn_nn")
                                },
                            }
                        },
                    }
                )
    return rows


def test_paired_report_never_authorizes_merge_or_mutates_input():
    data = records()
    before = deepcopy(data)
    report = summarize(data, "numerical")
    assert data == before
    assert not report["provenance_errors"]
    assert report["development_qualified"] == ["corrected", "stable_numeric"]
    assert report["merge_authorized"] is False
    delta = report["comparisons"][0]["deltas"]["all:mae"]
    assert delta["mean"] == pytest.approx(-0.1)
    assert delta["std"] == 0
    assert len(delta["paired"]) == 3


@pytest.mark.parametrize(
    "change", ["missing_seed", "cohort", "hardware", "control", "parity", "regression"]
)
def test_incomplete_or_regressing_development_evidence_cannot_qualify(change):
    data = records()
    if change == "missing_seed":
        data.pop()
    elif change == "cohort":
        data[-1]["cohorts"]["elite_top24"]["cohort_hash"] = "different"
    elif change == "hardware":
        data[-1]["gpu"] = "A10G"
    elif change == "control":
        data[-1]["metrics"]["all:ridge"]["rmse"] += 1
    elif change == "parity":
        data[-1]["inference_parity_passed"] = False
    else:
        data[-1]["metrics"]["all:attn_nn"]["mae"] += 1
    assert "stable_numeric" not in summarize(data, "numerical")["development_qualified"]


def test_duplicate_manifest_is_rejected():
    data = records()
    with pytest.raises(ValueError, match="Duplicate immutable"):
        summarize(data + [data[0]], "numerical")


@pytest.mark.parametrize("cohort", ["all", "elite_top24"])
@pytest.mark.parametrize("family", ["ridge", "nn", "attn_nn"])
def test_report_rejects_model_specific_row_filtering(cohort, family):
    data = records()
    data[-1]["metrics"][f"{cohort}:{family}"]["n"] = 1
    result = summarize(data, "numerical")
    assert not result["development_qualified"]
    assert any("samples" in reason for reason in result["provenance_errors"])


@pytest.mark.parametrize("change", ["dropout", "tf32", "graph", "missing"])
def test_report_rejects_incompatible_execution(change):
    data = records()
    row = data[-1]
    if change == "dropout":
        row["execution"]["overrides"]["FF_FORCE_DROPOUT_ZERO"] = "1"
    elif change == "tf32":
        row["tf32_matmul"] = False
    elif change == "graph":
        row["trainers"][0]["graph"] = False
    else:
        del row["execution"]
    result = summarize(data, "numerical")
    assert not result["development_qualified"]
    assert any("execution settings" in reason for reason in result["provenance_errors"])


def test_cli_verifies_content_addresses_and_retains_manifest_proofs(tmp_path, monkeypatch):
    for i, row in enumerate(records()):
        digest = hashlib.sha256(canonical_json(row).encode()).hexdigest()
        (tmp_path / f"{i}-{digest}-manifest.json").write_text(json.dumps(row))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--kind",
            "numerical",
            "--input-dir",
            str(tmp_path),
            "--pattern",
            "*-manifest.json",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text())
    assert len(report["manifest_proofs"]) == 18
    assert not report["merge_authorized"]
    manifest = next(tmp_path.glob("*-manifest.json"))
    row = json.loads(manifest.read_text())
    row["batch_job_id"] = "tampered"
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="content-address mismatch"):
        main()


def test_archived_protocol_and_decisions_are_preserved():
    root = Path(__file__).resolve().parents[2] / "todo/model-default-repair"
    assert hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() == (
        "caae4d8eea1ba16b8a4dcf6baf7fcec07db2b150950ae3f7ef0244b6018984e3"
    )
    for kind, count in (("weights", 36), ("selector", 12), ("numerical", 18)):
        report = json.loads((root / f"evidence/{kind}.json").read_text())
        assert report["records"] == count
        assert report["development_qualified"] == []
        assert report["merge_authorized"] is False
