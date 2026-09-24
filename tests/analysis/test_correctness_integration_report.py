"""Synthetic records only; no model/scaler fitting or network calls."""

import hashlib
import io
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis import audit_development_report as audit
from src.analysis import correctness_integration_report as report

pytestmark = pytest.mark.unit


def plan():
    return {
        "schema": "correctness-integration-plan/v1",
        "source_sha": "1" * 40,
        "baseline_sha": "2" * 40,
        "data_release": "3" * 64,
        "image_digest": "sha256:" + "4" * 64,
        "candidate_heads": {"1608": "5" * 40, "1613": "6" * 40},
        "runs": [
            {
                "prefix": f"ab_runs/correctness/{kind}-{year}",
                "positions": positions,
                "seeds": list(report.SEEDS),
                "origin": year,
            }
            for year in report.ORIGINS
            for kind, positions in (
                ("affected", list(report.AFFECTED)),
                ("qb", ["QB"]),
                ("native", ["K", "DST"]),
            )
        ],
    }


def cell(position, year, seed, arm):
    changed = position in report.AFFECTED and arm != "baseline"
    row = {
        f"pred_{family}_x": [2.0 if changed and family == "attn_nn" else 1.0]
        for family in audit.FAMILIES
    }
    record = {
        "position": position,
        "origin": year,
        "seed": seed,
        "variant": arm,
        "source_sha": "1" * 40,
        "data_release": "3" * 64,
        "batch_job_id": f"{position}-{year}",
        "gpu": "NVIDIA L4",
        "tf32_matmul": True,
        "tf32_cudnn": True,
        "row_hash": "a" * 64,
        "truth_hash": "b" * 64,
        "prepared_hashes": {"X_train": "c" * 64},
        "execution": {
            "regime": "eager",
            "device": "cuda:0",
            "seed": seed,
            "overrides": {"FF_AMP_DTYPE": "fp32"},
        },
        "trainers": [
            {
                "family": family,
                "device": "cuda:0",
                "amp": False,
                "graph": position != "K" or family != "attn_nn",
            }
            for family in ("nn", "attn_nn")
        ],
        "cohorts": {
            name: {"status": "available", "cohort_hash": "d" * 64, "models": {}}
            for name in ("elite_top24", "weekly_reference_top24")
        },
    }
    score = 2.0 if changed else 1.0
    return audit.Verified(
        record,
        pd.DataFrame(row),
        {f"{cohort}:attn_nn": {"mae": score, "rmse": score} for cohort in ("all", "elite_top24")},
        {family: {"weight": torch.tensor([1.0])} for family in ("nn", "attn_nn")},
    )


def complete_cells():
    return [cell(*identity) for identity in sorted(report.expected_cells())]


def fixture_record(position="WR", arm="baseline"):
    from tests.analysis.test_audit_development_report import fixture_record as old_fixture

    record, truth, data = old_fixture(position, missing=position == "K")
    record.update(
        schema="count-correctness-integration/v1",
        candidate_prs=[1608, 1613],
        variant=arm,
        probability_mean="corrected" if arm in {"mean", "combined"} else "legacy",
        count_likelihood_implementation=(
            "src.shared.training.ztnb2_log_prob"
            if arm in {"precision", "combined"}
            else "src.tuning.correctness_legacy_count.ztnb2_log_prob"
        ),
    )
    record.pop("promotion_eligible")
    gated = {
        "RB": ("receptions", "rushing_tds", "receiving_tds"),
        "WR": ("receptions", "receiving_tds"),
        "TE": ("receptions", "receiving_tds"),
    }.get(position, ())
    state = {"weight": torch.tensor([1.0])}
    state.update(
        {
            f"heads.{target}._ztnb_mean_version": torch.tensor(
                int(target == "receptions" and arm in {"mean", "combined"})
            )
            for target in gated
        }
    )
    buffer = io.BytesIO()
    torch.save(state, buffer)
    data["attn_state"] = buffer.getvalue()
    for trainer in record["trainers"]:
        trainer["checkpoint_selection"]["metric"] = "weighted_mae"
        if trainer["family"] == "attn_nn":
            trainer["checkpoint"] = "attn_state"
            if position in report.AFFECTED:
                with np.load(io.BytesIO(data["npz"])) as original:
                    values = {name: original[name] for name in original.files}
                n = len(values["truth__receptions"])
                values.update(
                    prediction__receptions_value_mu=np.ones(n),
                    prediction__receptions_value_log_alpha=np.zeros(n),
                )
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **values)
                data["attn_npz"] = buffer.getvalue()
                trainer["validation_raw"] = "attn_npz"
                trainer["count_numerics"] = {
                    "n_positive": n,
                    "active_numerical_defect": False,
                    "production_device": "cuda:0",
                    "reference_device": "cpu",
                    "observed_ranges": {"mu": [1.0, 1.0], "log_alpha": [0.0, 0.0]},
                    "errors": {
                        name: {"n_outside_tolerance": 0, "max_scaled_error": 0.0}
                        for name in ("log_probability", "d_mu", "d_log_alpha")
                    },
                }
    return record, truth, data


def mutate_attention_state(data, mutate):
    state = torch.load(io.BytesIO(data["attn_state"]), weights_only=True)
    mutate(state)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    data["attn_state"] = buffer.getvalue()


def test_full_plan_requires_all_108_cells_without_overlap():
    assert len(report.expected_cells()) == 108
    assert report.validate_plan(plan()) == plan()
    incomplete = plan()
    incomplete["runs"].pop()
    with pytest.raises(ValueError, match="108-cell"):
        report.validate_plan(incomplete)
    overlapping = plan()
    overlapping["runs"].append(deepcopy(overlapping["runs"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        report.validate_plan(overlapping)


def test_worsening_metrics_do_not_veto_valid_technical_evidence():
    result = report.summarize(complete_cells())
    assert result["verification_passed"]
    assert result["metric_improvement_veto"] is False
    affected = next(r for r in result["comparisons"] if r["position"] == "WR")
    assert affected["deltas"]["all:mae"]["mean"] == 1.0
    assert affected["deltas"]["all:mae"]["ci95_seed_mean"] == [1.0, 1.0]
    assert set(result["weekly_reference_top24"]) == {"2022", "2023"}


@pytest.mark.parametrize("field", ["prediction", "prepared_hashes", "truth_hash"])
def test_changed_control_or_input_identity_fails(field):
    cells = complete_cells()
    changed = next(
        c for c in cells if c.record["position"] == "WR" and c.record["variant"] == "combined"
    )
    if field == "prediction":
        changed.rows["pred_nn_x"] += 1e-8
    elif field == "prepared_hashes":
        changed.record[field] = {"X_train": "e" * 64}
    else:
        changed.record[field] = "e" * 64
    assert not report.summarize(cells)["verification_passed"]


def test_corrected_observed_defect_fails_before_loading_weights():
    record, _, _ = fixture_record("WR", "combined")
    next(t for t in record["trainers"] if t["family"] == "attn_nn")["count_numerics"][
        "active_numerical_defect"
    ] = True
    with pytest.raises(ValueError, match="observed numerical defect"):
        report.verify_record(record, None, lambda entry: pytest.fail("read unexpected artifact"))


def test_content_addressed_manifest_tampering_is_rejected(tmp_path):
    original = b'{"ok": true}'
    digest = hashlib.sha256(original).hexdigest()
    key = f"ab_runs/test/evidence/{digest}-manifest.json"
    path = tmp_path / "objects" / key
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ok": False}))
    archive = audit.Archive(tmp_path, "bucket", offline=True)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        archive.read(key, digest=digest)


@pytest.mark.parametrize("position,arm", [(p, a) for p in report.POSITIONS for a in report.arms(p)])
def test_schema_adapter_reuses_full_verification_without_mutating_evidence(position, arm):
    record, truth, data = fixture_record(position, arm)
    original = deepcopy(record)
    verified = report.verify_record(record, truth, data.__getitem__)
    assert verified.record == original == record
    assert verified.scores["all:nn"]["n"] == (1 if position == "K" else 2)


@pytest.mark.parametrize("position", report.AFFECTED)
@pytest.mark.parametrize("arm", report.ARMS)
def test_checkpoint_cannot_claim_a_different_reception_mean_arm(position, arm):
    record, truth, data = fixture_record(position, arm)
    wrong = 0 if arm in {"mean", "combined"} else 1
    mutate_attention_state(
        data,
        lambda state: state.update({"heads.receptions._ztnb_mean_version": torch.tensor(wrong)}),
    )
    with pytest.raises(ValueError, match="checkpoint.*expectation"):
        report.verify_record(record, truth, data.__getitem__)


@pytest.mark.parametrize("case", ["missing", "extra", "poisson", "nonscalar", "fractional"])
def test_exact_checkpoint_target_loss_map_is_required(case):
    record, truth, data = fixture_record("WR", "mean")

    def change(state):
        if case == "missing":
            del state["heads.receptions._ztnb_mean_version"]
        elif case == "extra":
            state["heads.fumbles_lost._ztnb_mean_version"] = torch.tensor(0)
        elif case == "poisson":
            state["heads.receiving_tds._ztnb_mean_version"] = torch.tensor(1)
        else:
            state["heads.receptions._ztnb_mean_version"] = torch.tensor(
                [1, 1] if case == "nonscalar" else 0.5
            )

    mutate_attention_state(data, change)
    with pytest.raises(ValueError, match="checkpoint.*expectation"):
        report.verify_record(record, truth, data.__getitem__)


@pytest.mark.parametrize("position", ["QB", "K", "DST"])
def test_control_checkpoint_cannot_add_a_corrected_count_head(position):
    record, truth, data = fixture_record(position, "combined")
    mutate_attention_state(
        data, lambda state: state.update({"heads.receptions._ztnb_mean_version": torch.tensor(1)})
    )
    with pytest.raises(ValueError, match="checkpoint.*expectation"):
        report.verify_record(record, truth, data.__getitem__)


@pytest.mark.parametrize("arm", report.ARMS)
def test_swapped_likelihood_implementation_is_rejected(arm):
    record, truth, data = fixture_record("WR", arm)
    record["count_likelihood_implementation"] = (
        "src.tuning.correctness_legacy_count.ztnb2_log_prob"
        if arm in {"precision", "combined"}
        else "src.shared.training.ztnb2_log_prob"
    )
    with pytest.raises(ValueError, match="likelihood implementation"):
        report.verify_record(record, truth, data.__getitem__)


@pytest.mark.parametrize("family", ["nn", "attn_nn"])
def test_changed_checkpoint_selection_policy_is_rejected(family):
    record, truth, data = fixture_record()
    next(t for t in record["trainers"] if t["family"] == family)["checkpoint_selection"][
        "metric"
    ] = "fantasy_rmse"
    with pytest.raises(ValueError, match="weighted_mae"):
        report.verify_record(record, truth, data.__getitem__)


def test_missing_cells_and_mixed_source_fail():
    cells = complete_cells()
    assert not report.summarize(cells[:-1])["verification_passed"]
    changed = next(c for c in cells if c.record["variant"] == "combined")
    changed.record["source_sha"] = "e" * 40
    assert not report.summarize(cells)["verification_passed"]
