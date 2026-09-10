"""Actual-CUDA acceptance for sample-weighted graph-prefix/tail validation.

Call ``verify_validation_reduction()`` from a GPU A/B spec's metric function,
or run this module directly. It exercises the real trainer, captured validation
and eager tail; no CUDA API is replaced and no CPU fallback can pass.
"""

from __future__ import annotations

import argparse
import json
import os


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _check_batch_size(device, batch_size):
    import torch
    from torch import nn

    from src.shared.training import MultiHeadTrainer, MultiTargetLoss, _GPUResidentBatcher

    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.ones(2, device=device))

        def forward(self, x):
            return {"a": x[:, 0] * self.gain[0], "b": x[:, 0] * 2 * self.gain[1]}

    model = ProbeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = MultiTargetLoss(
        target_names=["a", "b"],
        loss_weights={"a": 2.0, "b": 0.5},
        head_losses={"a": "mse", "b": "mse"},
    )
    train = _GPUResidentBatcher(
        (torch.zeros(4, 1, device=device),),
        {name: torch.zeros(4, device=device) for name in ("a", "b")},
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    val = _GPUResidentBatcher(
        (torch.tensor([[1.0], [1.0], [1.0], [10.0]], device=device),),
        {name: torch.zeros(4, device=device) for name in ("a", "b")},
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    callbacks = []
    trainer = MultiHeadTrainer(
        model,
        optimizer,
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        criterion,
        device,
        ["a", "b"],
        patience=10,
        log_every=100,
        epoch_callback=lambda epoch, value: callbacks.append((epoch, value)),
    )
    history = trainer.train(train, val, n_epochs=2)
    torch.cuda.synchronize(device)
    graph = trainer._graphed_val
    _require(trainer._graphed_step is not None, "Actual train capture did not engage")
    _require(
        graph is not None and graph._graph is not None, "Actual validation capture did not engage"
    )
    _require(graph.loss_sum.is_cuda, "Validation accumulators are not on CUDA")
    _require(graph._n_fixed == 4 // batch_size * batch_size, "Wrong captured prefix length")
    _require(graph._rem == 4 % batch_size, "Wrong eager tail length")
    expected = {
        "val_loss": 103.0,
        "val_loss_a": 25.75,
        "val_loss_b": 103.0,
        "val_mae_a": 3.25,
        "val_mae_b": 6.5,
    }
    for name, value in expected.items():
        _require(
            history[name] == [value, value],
            f"batch_size={batch_size}: {name}={history[name]}, expected {value}",
        )
    _require(callbacks == [(0, 103.0), (1, 103.0)], "Callback did not receive the sample mean")
    _require(trainer.scheduler.best == 103.0, "Plateau scheduler did not receive the sample mean")
    _require(abs(float(trainer.best_val_metric) - 3.9) < 1e-6, "Weighted-MAE selection changed")
    torch.testing.assert_close(model.gain, torch.ones_like(model.gain), rtol=0, atol=0)
    return {
        f"batch{batch_size}_loss": history["val_loss"][0],
        f"batch{batch_size}_loss_a": history["val_loss_a"][0],
        f"batch{batch_size}_loss_b": history["val_loss_b"][0],
        f"batch{batch_size}_prefix_rows": float(graph._n_fixed),
        f"batch{batch_size}_tail_rows": float(graph._rem),
        f"batch{batch_size}_verified_epochs": float(len(history["val_loss"])),
    }


def verify_validation_reduction(seed=42):
    """Return flat exact-result metrics, or fail if actual CUDA cannot verify them."""
    import torch

    _require(torch.cuda.is_available(), "Validation acceptance requires real CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    _require(major >= 8, "Validation graph acceptance requires the supported sm_80+ path")
    overrides = {
        "FF_DEVICE": "cuda",
        "FF_CUDA_GRAPH": "1",
        "FF_CUDA_GRAPH_FULL": "1",
        "FF_CUDA_GRAPH_OPT": "0",
        "FF_NN_FIXED_EPOCHS": "0",
        "FF_AMP_DTYPE": "fp32",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    proof = {"cuda_available": 1.0, "sm": float(major * 10 + minor), "seed": float(seed)}
    try:
        os.environ.update(overrides)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed)
            for batch_size in (2, 3, 4):
                proof.update(_check_batch_size(device, batch_size))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    proof["passed_cases"] = 3.0
    print("[validation-reduction-proof] " + json.dumps(proof, sort_keys=True), flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    verify_validation_reduction(args.seed)


if __name__ == "__main__":
    main()
