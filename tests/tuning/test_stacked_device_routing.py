from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from src.tuning import ab_ensemble_seeds as ensemble
from src.tuning import tune_nn

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("entrypoint", ["objective", "ensemble", "compare", "group"])
@pytest.mark.parametrize(
    "requested_device,cuda_available,captured_device,verify_boundary",
    [
        ("cpu", True, "cpu", False),
        ("mps", True, "mps", False),
        ("cuda", True, "cuda", False),
        ("auto", True, "cuda", False),
        ("auto", False, "cpu", False),
        ("mps", True, "mps", True),
    ],
)
def test_stacked_dispatch_uses_device_resolved_by_real_constructor(
    monkeypatch,
    tmp_path,
    entrypoint,
    requested_device,
    cuda_available,
    captured_device,
    verify_boundary,
):
    monkeypatch.setenv("FF_DEVICE", requested_device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    cuda_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: cuda_calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cuda_calls.append("empty"))
    monkeypatch.setattr(ensemble, "apply_ensemble_env", lambda epochs: None)
    monkeypatch.setattr(ensemble, "apply_eager_graph_env", lambda epochs: None)
    monkeypatch.setattr("src.shared.registry.get_config", lambda position: {})
    monkeypatch.setattr(
        "src.shared.platform_detect.detect_platform", lambda: SimpleNamespace(gpu_name="fixture")
    )
    monkeypatch.setattr("src.shared.utils.amp_dtype", lambda: torch.float32)
    monkeypatch.setattr("src.shared.utils.cuda_graph_full_enabled", lambda: False)
    monkeypatch.setattr(
        "src.tuning.resource_probe.ResourceProbe",
        lambda: SimpleNamespace(start=lambda: SimpleNamespace(stop=lambda: {})),
    )

    # MultiHeadTrainer.device is the result of the pipeline's CPU/CUDA/MPS
    # policy. No unavailable hardware is allocated in this dispatch test.
    def capture(position, seeds, **kwargs):
        return [
            {"trainer": SimpleNamespace(device=torch.device(captured_device))} for _ in seeds
        ], {}

    seen = []
    real_train = ensemble.train_stacked

    def train(captures, cfg, device, epochs, **kwargs):
        seen.append(device)
        if verify_boundary:
            return real_train(captures, cfg, device, epochs, **kwargs)
        if callback := kwargs.get("epoch_callback"):
            callback(0, 0.25)
        return {}, {}, None

    monkeypatch.setattr(ensemble, "capture_seeds", capture)
    monkeypatch.setattr(ensemble, "train_stacked", train)
    monkeypatch.setattr(ensemble, "predict_stacked", lambda *args: [])
    monkeypatch.setattr(ensemble, "run_eager_arm", lambda position, seeds, memo: [1.0] * len(seeds))

    def invoke():
        if entrypoint == "objective":
            monkeypatch.setattr(tune_nn, "_lease_cores", lambda *args, **kwargs: nullcontext())
            monkeypatch.setattr(
                tune_nn,
                "_sample_overrides",
                lambda *args: {"attn_max_seq_len": 8, "attn_history_stats": ["passing_yards"]},
            )
            trial = SimpleNamespace(number=0, report=lambda *args: None, should_prune=lambda: False)
            objective = tune_nn._make_stacked_objective("QB", {}, 42, 2, 1, scope="history")
            assert objective(trial) == 0.25
        elif entrypoint == "ensemble":
            result = ensemble.run_ensemble_ab("QB", 2, 1, parity_check=False)
            assert result["device"] == captured_device
        elif entrypoint == "compare":
            result = ensemble.run_compare("QB", 2, 1)
            assert result["device"] == captured_device
        else:
            import pandas as pd

            from src.qb.run_pipeline import CONFIG
            from src.tuning.ab_harness import Group, Variant, run_group_stacked

            prediction = {target: [0.0] for target in CONFIG["targets"]}
            monkeypatch.setattr(ensemble, "predict_stacked", lambda *args: [prediction, prediction])
            result = run_group_stacked(
                Group("QB", "baseline", (42, 43)),
                Variant("baseline"),
                lambda result, position: {"Ridge": {"mae": 0.0}},
                data_dir=str(tmp_path),
                stacked_epochs=1,
                run_fn=lambda *args, **kwargs: {"test_df": pd.DataFrame({"fantasy_points": [0.0]})},
            )
            assert len(result) == 2 and all(row["ok"] for row in result)

    if verify_boundary:
        with pytest.raises(RuntimeError, match="Use eager MPS"):
            invoke()
    else:
        invoke()
    assert seen == [torch.device(captured_device)]
    if captured_device != "cuda":
        assert not cuda_calls
