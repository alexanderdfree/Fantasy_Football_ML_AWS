"""Actual-CUDA acceptance for hurdle means, gradients and mixed precision."""

from __future__ import annotations

import math


def verify_hurdle_expectations(seed=42):
    """Compare GPU head outputs to float64 distribution means; refuse CPU."""
    import torch

    from src.shared.neural_net import GatedHead

    if not torch.cuda.is_available():
        raise RuntimeError("Hurdle acceptance requires actual CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError("Hurdle graph acceptance requires sm_80 or newer")
    proof = {"cuda_available": 1.0, "sm": float(major * 10 + minor)}
    cases = 0
    largest_error = 0.0
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed)
        for family in ("hurdle_negbin", "hurdle_poisson"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for rate, alpha in ((1.0, 1.0), (2e-6, 1e-6)):
                    head = GatedHead(2, gate_hidden=2, value_hidden=2, loss_family=family).to(
                        device=device, dtype=dtype
                    )
                    with torch.no_grad():
                        for parameter in head.parameters():
                            parameter.zero_()
                        head.value_mu[0].bias.fill_(math.log(math.expm1(rate - 1e-6)))
                        head.value_log_alpha.bias.fill_(math.log(alpha))
                    inputs = torch.zeros(4, 2, device=device, dtype=dtype)
                    prediction, gate, mu, log_alpha = head(inputs)
                    mu64 = mu.double()
                    if family == "hurdle_negbin":
                        alpha64 = log_alpha.double().exp().clamp(min=1e-6)
                        log_p0 = -torch.log1p(alpha64 * mu64) / alpha64
                    else:
                        log_p0 = -mu64
                    expected = gate.double().sigmoid() * mu64 / -torch.expm1(log_p0)
                    torch.testing.assert_close(prediction.double(), expected, rtol=1e-5, atol=1e-6)
                    conditional = head.conditional_mean(mu, log_alpha)
                    if not torch.isfinite(conditional).all() or not (conditional >= 1).all():
                        raise RuntimeError("A truncated positive mean must be finite and at least1")
                    largest_error = max(
                        largest_error, float((prediction.double() - expected).abs().max())
                    )
                    if rate == 1.0:
                        prediction.sum().backward()
                        parameters = [head.gate[-1].bias, head.value_mu[0].bias]
                        if family == "hurdle_negbin":
                            parameters.append(head.value_log_alpha.bias)
                        for parameter in parameters:
                            grad = parameter.grad
                            if (
                                grad is None
                                or not torch.isfinite(grad).all()
                                or not grad.abs().sum()
                            ):
                                raise RuntimeError("Hurdle expectation lost a parameter gradient")

                    # Exercise the same reporting operations under actual CUDA
                    # capture, including opt-in half/bfloat16 head arithmetic.
                    with torch.no_grad():
                        stream = torch.cuda.Stream(device=device)
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                head(inputs)
                        torch.cuda.current_stream(device).wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = head(inputs)[0]
                        graph.replay()
                        torch.cuda.synchronize(device)
                        torch.testing.assert_close(
                            captured.double(), expected, rtol=1e-5, atol=1e-6
                        )
                    cases += 1
    proof.update(passed_cases=float(cases), max_abs_error=largest_error)
    return proof
