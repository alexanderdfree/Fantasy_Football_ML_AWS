"""CPU checks validate wiring; the actual capture proof remains CUDA-only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.analysis.verify_cuda_capture_rollback import verify_cuda_capture_rollback
from src.tuning import ab_verify_cuda_capture as spec
from src.tuning.ab_harness import build_cells, resolve_spec


@pytest.mark.unit
def test_capture_spec_import_and_grid_need_no_torch():
    source = """
import sys
class WithoutTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('the submitter has no torch')
sys.meta_path.insert(0, WithoutTorch())
from src.tuning.ab_harness import build_cells, resolve_spec
from src.analysis import verify_cuda_capture_rollback
cells = build_cells(resolve_spec('src.tuning.ab_verify_cuda_capture'))
assert [(c.position, c.variant, c.seed) for c in cells] == [('QB', 'verify', 42)]
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_capture_probe_refuses_cpu_emulation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires actual CUDA"):
        verify_cuda_capture_rollback()


@pytest.mark.unit
def test_capture_spec_keeps_data_callables_and_small_raw_stat_pipeline(monkeypatch):
    from src.shared.registry import get_config

    base = get_config("QB")
    targets = list(base["targets"])
    callables = {key: value for key, value in base.items() if callable(value)}
    cfg = spec._tiny_config(dict(base))
    assert cfg["targets"] == targets
    assert all(cfg[key] is value for key, value in callables.items())
    assert cfg["nn_epochs"] == 1 and cfg["nn_dropout"] == 0
    assert cfg["nn_backbone_layers"] == [8, 8]
    # The full pipeline attaches test_df only when both predictors are present.
    assert cfg["train_ridge"] is True and cfg["train_base_nn"] is True
    assert cfg["ridge_alpha_grids"] == {target: [1.0] for target in targets}
    assert cfg["ridge_refine_points"] == 0 and cfg["ridge_pca_components"] is None
    assert cfg["ridge_cv_folds"] == 2
    cells = build_cells(resolve_spec(spec))
    assert len(cells) == 1 and cells[0].seed == 42


@pytest.mark.unit
def test_metric_hook_attaches_numeric_proof_without_replacing_pipeline_results(monkeypatch):
    proof = {"passed_cases": 4.0, "prime_model_reset_max_abs": 0.0}
    monkeypatch.setattr(
        "src.analysis.verify_cuda_capture_rollback.verify_cuda_capture_rollback",
        lambda seed: proof,
    )
    result = {"test_df": [1, 2], "nn_metrics": {"total": {"mae": 2.5}}}
    assert spec.metric_fn(result, "QB") == {
        "cuda_capture": proof,
        "NN": {"mae": 2.5, "test_rows": 2.0},
    }


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA hardware")
def test_actual_cuda_capture_rollback_and_replay():
    proof = verify_cuda_capture_rollback()
    assert proof["passed_cases"] == 4.0
    assert proof["success_replay_steps"] == 2.0
