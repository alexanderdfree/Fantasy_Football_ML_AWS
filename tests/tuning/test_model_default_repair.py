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
    trace.observe(4, 1, 3.4, 1)  # still violates the frozen anchor's PPR MAE
    report = trace.finish()
    assert report["legacy"]["epoch"] == 1
    assert report["rmse"]["epoch"] == 2
    assert report["guarded"] is None
    assert not report["guarded_qualifies"]


def test_guarded_policy_searches_full_declared_budget_after_other_policies_stop():
    trace = SelectionTrace(1)
    trace.observe(1, 1, 3, 4)
    trace.observe(2, 2, 3.5, 3)
    trace.observe(3, 3, 3.6, 3.2)
    trace.observe(4, 0.5, 2.8, 2.9)
    report = trace.finish()
    assert report["legacy"]["epoch"] == 1
    assert report["stop_epochs"] == {"legacy": 2, "rmse": 3}
    assert report["guarded"]["epoch"] == 4
    assert report["guarded_search_epochs"] == 4


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


def test_digest_pin_preserves_source_identity_and_rejects_ambiguous_images():
    from src.scripts.resolve_training_image import _image_sha
    from src.tuning.launch_ab import _swap_image_tag

    sha, digest = "a" * 40, "sha256:" + "b" * 64
    image = f"registry/repo:{sha}@{digest}"
    assert _image_sha({"containerProperties": {"image": image}}) == sha
    assert _swap_image_tag(image, "c" * 40) == "registry/repo:" + "c" * 40
    for invalid in (f"registry/repo@{digest}", f"registry/repo:{sha}@sha256:short"):
        with pytest.raises(ValueError):
            _image_sha({"containerProperties": {"image": invalid}})


def test_batch_clone_pins_digest_without_changing_production_definition():
    from unittest.mock import MagicMock

    from src.tuning import launch_ab

    batch = MagicMock()
    batch.describe_job_definitions.side_effect = [
        {
            "jobDefinitions": [
                {
                    "revision": 8,
                    "type": "container",
                    "containerProperties": {
                        "image": "registry/repo:" + "a" * 40,
                        "jobRoleArn": "original-role",
                    },
                }
            ]
        },
        {"jobDefinitions": []},
    ]
    batch.register_job_definition.return_value = {"revision": 9}
    digest = "sha256:" + "b" * 64
    assert launch_ab.resolve_job_definition("c" * 40, batch, image_digest=digest) == "ff-ab-job:9"
    registered = batch.register_job_definition.call_args.kwargs
    assert registered["jobDefinitionName"] == "ff-ab-job"
    assert registered["containerProperties"]["image"] == "registry/repo:" + "c" * 40 + "@" + digest
    assert registered["containerProperties"]["jobRoleArn"] == "original-role"


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


def test_kicker_provider_imputes_with_origin_ceiling_and_keeps_history(monkeypatch):
    from src.dst import run_pipeline as dst_runner
    from src.k import data as kicker_data
    from src.k import run_pipeline as kicker_runner
    from src.training.contracts import DatasetSplits
    from src.tuning import model_default_repair as repair

    frame = pd.DataFrame(
        {"season": list(range(2015, 2026)), "season_type": "REG", "player_id": "p", "week": 1}
    )
    kick_history = object()
    seen = []

    def fake(cfg):
        seen.append(kicker_data._TRAIN_MAX_SEASON)
        return DatasetSplits(
            frame.iloc[:5],
            frame.iloc[5:6],
            frame.iloc[6:],
            {"attn_history_builder_fn": kick_history},
        )

    original_ceiling = kicker_data._TRAIN_MAX_SEASON
    monkeypatch.setattr(repair, "STATE", {"origin": 2022})
    monkeypatch.setattr(kicker_runner, "provide_dataset", fake)
    monkeypatch.setattr(dst_runner, "provide_dataset", dst_runner.provide_dataset)
    repair.install_native_origins()
    result = kicker_runner.provide_dataset({})
    assert seen == [2020]
    assert original_ceiling == kicker_data._TRAIN_MAX_SEASON
    assert result.bindings["attn_history_builder_fn"] is kick_history
    assert result.test.season.tolist() == [2022]


def test_protected_cohort_metrics_keep_sub_rounding_regressions(tmp_path):
    from types import SimpleNamespace

    from src.training.context import RunContext, use_context
    from src.tuning.model_default_repair import precise_cohorts

    prior = pd.DataFrame(
        {
            "player_id": [f"p{i:02}" for i in range(30)],
            "season": 2021,
            "week": 1,
            "season_type": "REG",
            "receiving_yards": 10.0,
            "receiving_tds": 0.0,
            "receptions": 1.0,
            "fumbles_lost": 0.0,
            "fantasy_points": 2.0,
        }
    )
    frame = prior.assign(season=2022)
    for family in FAMILIES:
        frame[f"pred_{family}_total"] = 2.0000001
        for target in ("receiving_yards", "receiving_tds", "receptions", "fumbles_lost"):
            frame[f"pred_{family}_{target}"] = frame[target] + (
                0.0000001 if target == "receptions" else 0
            )
    result = SimpleNamespace(prepared=SimpleNamespace(train=prior.iloc[:0], val=prior))
    with use_context(RunContext(tmp_path / "output", tmp_path / "data")):
        blocks = precise_cohorts(result, frame, "WR")
    assert 0 < blocks["elite_top24"]["models"]["nn"]["mae"] < 1e-6
    assert blocks["weekly_reference_top24"]["status"] == "unavailable"


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
