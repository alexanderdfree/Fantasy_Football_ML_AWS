"""Pure orchestration/math checks: these tests never train or fit a model."""

from copy import deepcopy

import pandas as pd
import pytest

from src.tuning.model_default_repair import configure, split_origin
from src.tuning.repair_gate import FAMILIES, POSITIONS, SEEDS, promotion_gate
from src.tuning.repair_selection import SelectionTrace

pytestmark = pytest.mark.unit


def test_original_stop_freezes_legacy_anchor():
    trace = SelectionTrace(2)
    for epoch, scores in enumerate(
        [(2, 4, 5), (1, 3, 4), (2, 2.5, 3.8), (3, 2.4, 3.7), (0.1, 1, 2)], 1
    ):
        trace.observe(epoch, *scores)
    report = trace.finish()
    assert report["legacy"]["epoch"] == 2
    assert report["stop_epochs"]["legacy"] == 4
    assert report["guarded"]["epoch"] == 5


def test_guard_uses_ppr_mae_and_never_claims_baseline_is_improvement():
    trace = SelectionTrace(1)
    trace.observe(1, 2, 3, 4)
    trace.observe(2, 3, 3.5, 3)
    trace.observe(3, 4, 3.2, 3.1)
    trace.observe(4, 1, 1, 1)  # after both policies stopped
    report = trace.finish()
    assert report["legacy"]["epoch"] == 1
    assert report["rmse"]["epoch"] == 2
    assert report["guarded"] is None
    assert not report["guarded_qualifies"]


def test_nonfinite_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        SelectionTrace(2).observe(1, 1, float("nan"), 1)


def test_count_reference_normalizes_and_gradients_match_finite_differences():
    import torch

    from src.analysis.repair_count_diagnostics import stable_ztnb_reference

    y = torch.arange(1, 129, dtype=torch.float64)
    mu = torch.full_like(y, 2.0)
    alpha = torch.full_like(y, -0.7)
    assert torch.allclose(
        stable_ztnb_reference(y, mu, alpha).exp().sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-10,
    )
    mu = torch.tensor([0.01, 0.5, 8.0], dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor([-5.0, -0.7, 1.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1.0, 2.0, 20.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda m, a: stable_ztnb_reference(y, m, a), (mu, alpha))


def test_aws_only_training_guard(monkeypatch):
    monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="AWS Batch"):
        configure({}, arm="baseline", mode="wr")


@pytest.mark.parametrize("position,floor", [("WR", 2013), ("K", 2015), ("DST", 2013)])
def test_native_and_skill_origin_slicing(position, floor):
    frame = pd.DataFrame(
        {"season": list(range(2012, 2026)), "player_id": "p", "week": 1, "season_type": "REG"}
    )
    train, val, test = split_origin(
        (frame.iloc[:8], frame.iloc[8:10], frame.iloc[10:]), 2022, position=position
    )
    assert train.season.tolist() == list(range(floor, 2021))
    assert val.season.tolist() == [2021]
    assert test.season.tolist() == [2022]


def records():
    output = []
    for position in POSITIONS:
        for origin in (2024, 2025):
            for seed in SEEDS:
                for variant in ("baseline", "candidate"):
                    changed = variant == "candidate"
                    metrics = {
                        f"{cohort}:{family}": {
                            "mae": 2.0 - changed * 0.1,
                            "rmse": 3.0 - changed * 0.1,
                        }
                        for cohort in ("all", "elite_top24", "weekly_reference_top24")
                        for family in FAMILIES
                    }
                    output.append(
                        dict(
                            position=position,
                            origin=origin,
                            seed=seed,
                            variant=variant,
                            metrics=metrics,
                            source_sha="sha",
                            data_release="release",
                            gpu="L4",
                            prepared_hashes={"X": "hash"},
                            row_hash="rows",
                            truth_hash="truth",
                            batch_job_id="batch",
                            inference_parity_passed=True,
                            frozen_candidate_sha256="freeze",
                            cohorts={
                                c: dict(status="available", n=24, cohort_hash="cohort")
                                for c in ("elite_top24", "weekly_reference_top24")
                            },
                        )
                    )
    return output


def test_gate_requires_both_metrics_every_model_and_year():
    data = records()
    affected = {p: list(FAMILIES) for p in POSITIONS}
    assert promotion_gate(data, candidate="candidate", affected=affected)["passed"]
    for row in data:
        if row["position"] == "WR" and row["origin"] == 2024 and row["variant"] == "candidate":
            row["metrics"]["all:attn_nn"]["mae"] = 2.0
    report = promotion_gate(data, candidate="candidate", affected=affected)
    assert not report["passed"]
    assert any("WR/2024/attn_nn/all/mae" in r for r in report["reasons"])


@pytest.mark.parametrize(
    "change", ["missing", "cohort", "hardware", "duplicate", "parity", "control"]
)
def test_missing_or_incompatible_evidence_fails_closed(change):
    data = deepcopy(records())
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
    else:
        affected["WR"].remove("ridge")
    assert not promotion_gate(data, candidate="candidate", affected=affected)["passed"]
