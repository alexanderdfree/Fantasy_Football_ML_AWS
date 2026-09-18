"""Actual-CUDA proof of full-step capture rollback and successful replay.

Runs only when explicitly called. The A/B spec in
``src.tuning.ab_verify_cuda_capture`` records these numeric checks alongside
one small QB pipeline cell. No CUDA APIs are replaced or emulated here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json


class _InjectedCaptureFailure(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _model_error(model, expected):
    import torch

    largest = 0.0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        largest = max(largest, float((value - expected[name]).abs().max().item()))
    return largest


def _optimizer_reset_error(optimizer):
    import torch

    _require(bool(optimizer.state), "The real priming update did not allocate Adam state")
    largest = 0.0
    for state in optimizer.state.values():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            value = state[name]
            delta = float(value.abs().max().item())
            _require(delta == 0.0, f"Adam {name} retained a warmup update: {delta}")
            largest = max(largest, delta)
            _require(torch.isfinite(value).all().item(), f"Adam {name} is non-finite")
    return largest


def _eager_update(model, optimizer, inputs, indices):
    import torch

    optimizer.zero_grad(set_to_none=False)
    loss = model(inputs[indices]).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    optimizer.step()
    return loss.detach()


def _training_error(model, optimizer, reference, reference_optimizer):
    import torch

    largest = 0.0
    expected = reference.state_dict()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-6)
        largest = max(largest, float((value - expected[name]).abs().max().item()))
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        for name in ("step", "exp_avg", "exp_avg_sq"):
            value = optimizer.state[parameter][name]
            expected_value = reference_optimizer.state[other][name]
            torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-6)
            largest = max(largest, float((value - expected_value).abs().max().item()))
    return largest


def _check_case(device, seed, phase, fail_after):
    import torch
    from torch import nn

    from src.shared.training import _GraphedFullStep

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.BatchNorm1d(2), nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.0), nn.Linear(4, 1)
    ).to(device=device, dtype=torch.float32)
    reference = copy.deepcopy(model)

    def make_optimizer(net):
        return torch.optim.AdamW(
            net.parameters(), lr=0.01, foreach=False, fused=True, capturable=True
        )

    optimizer = make_optimizer(model)
    reference_optimizer = make_optimizer(reference)
    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 7.0], [8.0, 5.0], [4.0, 3.0], [9.0, 1.0], [2.0, 6.0]],
        device=device,
        dtype=torch.float32,
    )
    snapshot = {name: value.detach().clone() for name, value in model.state_dict().items()}
    step = _GraphedFullStep(
        lambda indices: model(inputs[indices]).square().mean(),
        model,
        optimizer,
        4,
        device,
        contextlib.nullcontext,
    )
    body = step._run_body
    calls = 0

    def update_then_fail():
        nonlocal calls
        body()
        calls += 1
        if calls == fail_after:
            raise _InjectedCaptureFailure(f"injected {phase} failure after body {calls}")

    step._run_body = update_then_fail
    try:
        step.build()
    except _InjectedCaptureFailure:
        _require(fail_after is not None and calls == fail_after, "Unexpected failure phase")
    else:
        _require(fail_after is None, f"The {phase} failure injection did not execute")
    finally:
        step._run_body = body
    torch.cuda.synchronize(device)
    proof = {
        f"{phase}_model_reset_max_abs": _model_error(model, snapshot),
        f"{phase}_optimizer_reset_max_abs": _optimizer_reset_error(optimizer),
        f"{phase}_body_calls": float(calls),
    }
    indices = torch.tensor([2, 4, 1, 5], device=device)
    if fail_after is not None:
        _require(step._graph is None, f"{phase}: a failed graph remained installed")
        _require(not step._baked_lr_tensors, f"{phase}: baked LR tensors were retained")
        lr = optimizer.param_groups[0]["lr"]
        _require(isinstance(lr, float) and lr == 0.01, f"{phase}: original LR was not restored")
        proof[f"{phase}_lr_restored"] = 1.0
        _eager_update(model, optimizer, inputs, indices)
        _eager_update(reference, reference_optimizer, inputs, indices)
        proof[f"{phase}_fallback_max_abs"] = _training_error(
            model, optimizer, reference, reference_optimizer
        )
        return proof

    _require(calls == 5 and step._graph is not None, "Successful graph capture did not complete")
    _require(len(step._baked_lr_tensors) == 1, "Captured optimizer has no baked LR")
    baked_lr = step._baked_lr_tensors[0]
    _require(optimizer.param_groups[0]["lr"] is baked_lr, "Captured LR tensor is not bound")
    replay_error = 0.0
    for replay_number, lr in enumerate((0.01, 0.004), 1):
        if replay_number == 2:
            # Reproduce a scheduler rebinding LR to a fresh float between epochs.
            optimizer.param_groups[0]["lr"] = lr
            reference_optimizer.param_groups[0]["lr"] = lr
            step.refresh_lr_from_scheduler()
            _require(optimizer.param_groups[0]["lr"] is baked_lr, "LR refresh changed its address")
            torch.testing.assert_close(
                baked_lr, torch.tensor(lr, device=device, dtype=torch.float32), rtol=0, atol=0
            )
        step.replay(indices)
        expected_loss = _eager_update(reference, reference_optimizer, inputs, indices)
        torch.cuda.synchronize(device)
        _require(torch.isfinite(step.loss_value()).item(), "Replay produced a non-finite loss")
        torch.testing.assert_close(step.loss_value(), expected_loss, rtol=1e-5, atol=1e-6)
        replay_error = max(
            replay_error, _training_error(model, optimizer, reference, reference_optimizer)
        )
    delta = max(
        float((value.detach() - snapshot[name]).abs().max().item())
        for name, value in model.named_parameters()
    )
    _require(delta > 0.0, "Replay did not update any model parameter")
    _require(
        all(float(state["step"].item()) == 2.0 for state in optimizer.state.values()),
        "Replay did not execute exactly two Adam updates",
    )
    proof.update(
        success_replay_max_abs=replay_error,
        success_lr_refresh=1.0,
        success_replay_steps=2.0,
        success_parameter_delta=delta,
    )
    return proof


def verify_cuda_capture_rollback(seed=42):
    """Return flat numeric proof metrics, raising if any actual-CUDA check fails."""
    import torch

    _require(torch.cuda.is_available(), "This proof requires actual CUDA; CPU emulation is invalid")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    _require(major >= 8, "This proof targets the supported sm_80+ CUDA graph path")
    proof = {"cuda_available": 1.0, "sm": float(major * 10 + minor), "seed": float(seed)}
    with torch.random.fork_rng(devices=[device.index]):
        for phase, fail_after in (("prime", 1), ("warmup", 3), ("capture", 5), ("success", None)):
            proof.update(_check_case(device, seed, phase, fail_after))
    proof["passed_cases"] = 4.0
    print("[cuda-capture-proof] " + json.dumps(proof, sort_keys=True), flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    verify_cuda_capture_rollback(args.seed)


if __name__ == "__main__":
    main()
