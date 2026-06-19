"""Unit tests for src/tuning/launch_ablate.py.

Mirrors test_launch_ab.py: no AWS — boto3 clients are mocked and we assert the
*shape* of the submission (command, env) plus the variant selection and the
collect/report contract over per-cell S3 JSONs.
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest

from src.tuning import launch_ablate
from src.tuning.ablate_batch import cell_result_key

pytestmark = pytest.mark.unit

MOD = "src.tuning.ablate_attn_arch"


def _fake_module(record=None):
    def _print_summary(results, targets, variants_run):
        if record is not None:
            record.append((len(results), tuple(targets), tuple(variants_run)))
        return True

    return types.SimpleNamespace(
        ABLATION_NAME="attn_arch",
        BASELINE="baseline",
        VARIANTS={"baseline": (), "selfattn": (), "alibi": ()},
        print_summary=_print_summary,
    )


def test_selected_variants_keeps_baseline_first():
    module = _fake_module()
    assert launch_ablate._selected_variants(module, None) == ["baseline", "selfattn", "alibi"]
    assert launch_ablate._selected_variants(module, ["selfattn"]) == ["baseline", "selfattn"]
    # Baseline is not duplicated when explicitly named.
    assert launch_ablate._selected_variants(module, ["baseline", "alibi"]) == ["baseline", "alibi"]


def test_selected_variants_rejects_unknown():
    module = _fake_module()
    with pytest.raises(SystemExit, match="unknown variant"):
        launch_ablate._selected_variants(module, ["nope"])


def test_submit_ablate_job_command_and_env():
    """One position job rides --mode=tune and carries the FF_ABLATE_* env the
    container entry reads; --cuda-graph false forwards the eager force-off."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "job-123"}

    pos, job_id = launch_ablate.submit_ablate_job(
        "RB",
        mod_dotted=MOD,
        run_id="r1",
        s3_prefix="ablation_runs",
        job_definition="ff-ab-job:7",
        image_sha="deadbeef",
        seeds=[42, 7],
        only=["selfattn"],
        cuda_graph="false",
        batch_client=batch,
    )
    assert (pos, job_id) == ("RB", "job-123")
    kwargs = batch.submit_job.call_args.kwargs
    assert kwargs["jobDefinition"] == "ff-ab-job:7"
    assert kwargs["containerOverrides"]["command"] == ["--position", "RB", "--mode", "tune"]
    env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
    assert env["FF_TUNE_ABLATE_MOD"] == MOD
    assert env["FF_ABLATE_RUN_ID"] == "r1"
    assert env["FF_ABLATE_SEEDS"] == "42,7"
    assert env["FF_ABLATE_VARIANTS"] == "selfattn"
    assert env["FF_DEVICE"] == "cuda"
    assert env["FF_CUDA_GRAPH"] == "0"  # eager
    assert env["FF_TRAIN_GIT_SHA"] == "deadbeef"


def test_submit_ablate_job_cuda_graph_auto_omits_env():
    """--cuda-graph auto forwards nothing (container autodetects the production
    graphed path); --only absent means no FF_ABLATE_VARIANTS (run all)."""
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "j"}
    launch_ablate.submit_ablate_job(
        "QB",
        mod_dotted=MOD,
        run_id="r1",
        s3_prefix="ablation_runs",
        job_definition="ff-ab-job:7",
        image_sha="sha",
        seeds=[42],
        only=None,
        cuda_graph="auto",
        batch_client=batch,
    )
    env = {
        e["name"]: e["value"]
        for e in batch.submit_job.call_args.kwargs["containerOverrides"]["environment"]
    }
    assert "FF_CUDA_GRAPH" not in env
    assert "FF_ABLATE_VARIANTS" not in env


def test_collect_results_reconstructs_and_flags_missing():
    """A present cell round-trips into an AblationResult; a missing one becomes
    an error result so the report surfaces the gap."""
    present = {
        "position": "RB",
        "seed": 42,
        "variant": "selfattn",
        "metrics": {"attn_fp_mae": 4.1},
        "timings": {},
        "metadata": {},
        "error": None,
    }

    def _get_object(Bucket, Key):
        if Key == cell_result_key("ablation_runs", "r1", "RB-selfattn-42"):
            return {"Body": MagicMock(read=lambda: json.dumps(present).encode())}
        raise KeyError(Key)  # everything else missing

    s3 = MagicMock()
    s3.get_object.side_effect = _get_object

    by_pos = launch_ablate.collect_results(
        positions=["RB"],
        variants=["baseline", "selfattn"],
        seeds=[42],
        bucket="b",
        s3_prefix="ablation_runs",
        run_id="r1",
        s3_client=s3,
    )
    rows = {(r.variant, r.seed): r for r in by_pos["RB"]}
    assert rows[("selfattn", 42)].error is None
    assert rows[("selfattn", 42)].metrics == {"attn_fp_mae": 4.1}
    assert "no result" in rows[("baseline", 42)].error  # missing -> flagged


def test_report_calls_print_summary_per_position(monkeypatch):
    """_report routes each position's rows through the module's print_summary
    with the position targets and the variant list (3-arg attn_arch shape)."""
    record = []
    module = _fake_module(record)
    monkeypatch.setattr("src.shared.registry.get_config", lambda pos: {"targets": ["t1", "t2"]})

    from src.tuning.ablation_runner import AblationResult

    by_pos = {
        "RB": [
            AblationResult("RB", 42, "baseline", {}, {}, {}, None),
            AblationResult("RB", 42, "selfattn", {}, {}, {}, "boom"),
        ]
    }
    n_failed = launch_ablate._report(module, by_pos, ["baseline", "selfattn"])
    assert n_failed == 1  # one errored cell
    assert record == [(2, ("t1", "t2"), ("baseline", "selfattn"))]
