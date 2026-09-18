"""No-fit tests for the read-only confirmation gate, independent of model PRs."""

import json
import sys

import pytest

from src.tuning.repair_gate import FAMILIES, POSITIONS, SEEDS, main, promotion_gate

pytestmark = pytest.mark.unit


def records():
    return [
        {
            "position": position,
            "origin": origin,
            "seed": seed,
            "variant": variant,
            "metrics": {
                f"{cohort}:{family}": {
                    "mae": 2.0 - (variant == "candidate") * 0.1,
                    "rmse": 3.0 - (variant == "candidate") * 0.1,
                }
                for cohort in ("all", "elite_top24", "weekly_reference_top24")
                for family in FAMILIES
            },
            "source_sha": "a" * 40,
            "data_release": "d" * 64,
            "gpu": "L4",
            "prepared_hashes": {"X": "hash"},
            "row_hash": "rows",
            "truth_hash": "truth",
            "batch_job_id": "batch",
            "inference_parity_passed": True,
            "frozen_candidate_sha256": "f" * 64,
            "cohorts": {
                c: {"status": "available", "n": 24, "cohort_hash": "cohort"}
                for c in ("elite_top24", "weekly_reference_top24")
            },
        }
        for position in POSITIONS
        for origin in (2024, 2025)
        for seed in SEEDS
        for variant in ("baseline", "candidate")
    ]


@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_each_model_must_improve_both_metrics_in_each_confirmation_year(metric):
    data = records()
    affected = {p: list(FAMILIES) for p in POSITIONS}
    assert promotion_gate(data, candidate="candidate", affected=affected)["passed"]
    for row in data:
        if row["position"] == "WR" and row["origin"] == 2024 and row["variant"] == "candidate":
            row["metrics"]["all:attn_nn"][metric] = 2.0 if metric == "mae" else 3.0
    report = promotion_gate(data, candidate="candidate", affected=affected)
    assert not report["passed"]
    assert any(f"WR/2024/attn_nn/all/{metric}" in r for r in report["reasons"])


@pytest.mark.parametrize("cohort", ["elite_top24", "weekly_reference_top24"])
@pytest.mark.parametrize("metric", ["mae", "rmse"])
def test_protected_cohorts_may_tie_but_may_not_regress(cohort, metric):
    data = records()
    affected = {p: list(FAMILIES) for p in POSITIONS}
    original = 2.0 if metric == "mae" else 3.0
    for row in data:
        if row["variant"] == "candidate":
            row["metrics"][f"{cohort}:attn_nn"][metric] = original
    assert promotion_gate(data, candidate="candidate", affected=affected)["passed"]
    data[-1]["metrics"][f"{cohort}:attn_nn"][metric] += 0.00001
    assert not promotion_gate(data, candidate="candidate", affected=affected)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "cohort",
        "hardware",
        "duplicate",
        "parity",
        "control",
        "freeze",
        "mixed_source",
        "legacy_data",
        "nonfinite",
        "missing_metric",
        "development",
    ],
)
def test_missing_or_incompatible_evidence_fails_closed(change):
    data = records()
    affected = {p: list(FAMILIES) for p in POSITIONS}
    if change == "missing":
        data.pop()
    elif change == "cohort":
        data[-1]["cohorts"]["weekly_reference_top24"]["status"] = "partial"
    elif change == "hardware":
        data[-1]["gpu"] = "A10G"
    elif change == "duplicate":
        data.append(data[-1])
    elif change == "parity":
        data[-1]["inference_parity_passed"] = False
    elif change == "freeze":
        data[-1]["frozen_candidate_sha256"] = "e" * 64
    elif change == "mixed_source":
        for row in data[-2:]:
            row["source_sha"] = "b" * 40
    elif change == "legacy_data":
        for row in data:
            row["data_release"] = "legacy"
    elif change == "nonfinite":
        data[-1]["metrics"]["all:ridge"]["mae"] = float("nan")
    elif change == "missing_metric":
        del data[-1]["metrics"]["all:ridge"]
    elif change == "development":
        for row in data:
            row["origin"] -= 2
    else:
        affected["WR"].remove("ridge")
    assert not promotion_gate(data, candidate="candidate", affected=affected)["passed"]


def test_cli_writes_failure_report_and_returns_nonzero(tmp_path, monkeypatch):
    source, affected, output = (tmp_path / name for name in ("records", "affected", "report"))
    source.write_text("[]")
    affected.write_text(json.dumps({"WR": ["nn"]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair_gate",
            str(source),
            "--candidate",
            "candidate",
            "--affected",
            str(affected),
            "--output",
            str(output),
        ],
    )
    assert main() == 1
    assert not json.loads(output.read_text())["passed"]
