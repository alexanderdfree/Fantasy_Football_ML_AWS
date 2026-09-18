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
                    "n": 24,
                }
                for cohort in ("all", "elite_top24", "weekly_reference_top24")
                for family in FAMILIES
            },
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
            "trainers": [
                {"family": family, "device": "cuda:0", "amp": False, "graph": True}
                for family in ("nn", "attn_nn")
            ],
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


@pytest.mark.parametrize("cohort", ["all", "elite_top24", "weekly_reference_top24"])
@pytest.mark.parametrize("both", [False, True])
def test_filtered_model_rows_cannot_pass_even_if_counts_match(cohort, both):
    data = records()
    affected = {p: list(FAMILIES) for p in POSITIONS}
    for row in data:
        if both or row["variant"] == "candidate":
            row["metrics"][f"{cohort}:attn_nn"]["n"] = 1
    result = promotion_gate(data, candidate="candidate", affected=affected)
    assert not result["passed"]
    assert any("incomplete model samples" in reason for reason in result["reasons"])


@pytest.mark.parametrize("change", ["dropout", "dtype", "tf32", "graph", "seed", "missing"])
def test_execution_differences_cannot_be_attributed_to_candidate(change):
    data = records()
    row = data[-1]
    if change in ("dropout", "dtype"):
        key, value = (
            ("FF_FORCE_DROPOUT_ZERO", "1") if change == "dropout" else ("FF_AMP_DTYPE", "bf16")
        )
        row["execution"]["overrides"][key] = value
    elif change == "tf32":
        row["tf32_matmul"] = False
    elif change == "graph":
        row["trainers"][0]["graph"] = False
    elif change == "seed":
        row["execution"]["seed"] = 99
    else:
        del row["execution"]
    result = promotion_gate(
        data, candidate="candidate", affected={p: list(FAMILIES) for p in POSITIONS}
    )
    assert not result["passed"]
    assert any("execution settings" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "missing", ["metrics", "null_metrics", "null_reference", "null_model", "null_trainers"]
)
def test_cli_writes_rejection_for_incomplete_evidence(tmp_path, monkeypatch, missing):
    data = records()
    row = data[-1]
    if missing == "metrics":
        del row["metrics"]
    elif missing == "null_metrics":
        row["metrics"] = None
    elif missing == "null_reference":
        row["metrics"]["all:ridge"] = None
    elif missing == "null_model":
        row["metrics"]["elite_top24:attn_nn"] = None
    else:
        row["trainers"] = None
    source, affected, output = (tmp_path / name for name in ("records", "affected", "report"))
    source.write_text(json.dumps(data))
    affected.write_text(json.dumps({p: list(FAMILIES) for p in POSITIONS}))
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
    report = json.loads(output.read_text())
    assert report["passed"] is False
    assert report["reasons"]
