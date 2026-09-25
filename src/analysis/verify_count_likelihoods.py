"""Actual CUDA count-likelihood checks against an independent Decimal law."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext


def _reference(count, mu, log_alpha=None):
    """Exact finite rising product and analytical derivatives at 90 digits."""
    with localcontext() as ctx:
        ctx.prec = 90
        y, m = Decimal(count), Decimal(str(mu))
        if log_alpha is None:
            p0 = (-m).exp()
            log_p = y * m.ln() - m - Decimal(math.factorial(count)).ln() - (1 - p0).ln()
            return float(log_p), float(y / m - 1 - p0 / (1 - p0)), 0.0
        raw_alpha = Decimal(str(log_alpha)).exp()
        alpha = max(raw_alpha, Decimal("1e-6"))
        r = (1 + alpha * m).ln()
        p0 = (-r / alpha).exp()
        log_p = (
            sum((1 + alpha * k).ln() for k in range(count))
            + y * m.ln()
            - Decimal(math.factorial(count)).ln()
            - (y + 1 / alpha) * r
            - (1 - p0).ln()
        )
        d_mu = y / m - (1 + alpha * y) / (1 + alpha * m)
        d_mu -= p0 / ((1 - p0) * (1 + alpha * m))
        d_alpha = (
            sum(Decimal(k) / (1 + alpha * k) for k in range(count))
            + r / alpha**2
            - (y + 1 / alpha) * m / (1 + alpha * m)
            + p0 / (1 - p0) * (r / alpha**2 - m / (alpha * (1 + alpha * m)))
        )
        return (
            float(log_p),
            float(d_mu),
            float(d_alpha * alpha) if raw_alpha > Decimal("1e-6") else 0.0,
        )


def _capture_count_gradients(log_probability, mu, log_alpha):
    """Own fresh capture leaves and keep warmup/capture on one explicit stream.

    Eager reference outputs can retain their leaves' AccumulateGrad nodes on
    the default stream even after autograd.grad frees saved intermediates.
    Reusing those leaves would make capture wait on the default stream. Return
    the fresh inputs as well so their storage stays live through both replays.
    """
    import torch

    stream = torch.cuda.Stream(device=mu.device)
    stream.wait_stream(torch.cuda.current_stream(mu.device))
    with torch.cuda.stream(stream):
        inputs = tuple(value.detach().clone().requires_grad_(True) for value in (mu, log_alpha))
        for _ in range(3):
            warm = log_probability(*inputs)
            warm_gradients = torch.autograd.grad(warm.sum(), inputs)
            del warm, warm_gradients
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = log_probability(*inputs)
            gradients = torch.autograd.grad(captured.sum(), inputs)
    torch.cuda.current_stream(mu.device).wait_stream(stream)
    return graph, stream, inputs, captured, gradients


def verify_count_likelihoods(seed=42):
    """Require real sm80+ CUDA; exercise eager, vmap and two graph replays."""
    import torch

    from src.shared.training import ztnb2_log_prob, ztp_log_prob

    if not torch.cuda.is_available():
        raise RuntimeError("Count likelihood acceptance requires actual CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError("Count likelihood graph acceptance requires sm_80 or newer")
    cases, max_scaled_error = 0, 0.0
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed)
        for family in ("nb", "poisson"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                counts = [1, 2, 30]
                mu = torch.tensor([2e-6, 0.2, 20.0], device=device, dtype=dtype, requires_grad=True)
                log_alpha = torch.tensor(
                    [-5.0, 0.0, 1.0], device=device, dtype=dtype, requires_grad=True
                )
                y = torch.tensor(counts, device=device, dtype=dtype)
                expected = torch.tensor(
                    [
                        _reference(k, m, a if family == "nb" else None)
                        for k, m, a in zip(
                            counts,
                            mu.detach().cpu().tolist(),
                            log_alpha.detach().cpu().tolist(),
                            strict=True,
                        )
                    ],
                    device=device,
                    dtype=torch.float64,
                )

                def log_probability(m, a, y=y, family=family):
                    value = ztnb2_log_prob(y, m, a) if family == "nb" else ztp_log_prob(y, m)
                    return value + a * 0  # Poisson has a correctly zero dispersion derivative.

                def check(value, gradients, expected=expected, dtype=dtype):
                    nonlocal max_scaled_error
                    actual = (value, *gradients)
                    for index, (got, want) in enumerate(zip(actual, expected.T, strict=True)):
                        if not torch.isfinite(got).all():
                            raise RuntimeError("Count likelihood or gradient became nonfinite")
                        # Gradients are cast back into leaf dtype by autograd.
                        tolerance = 1e-5 if index == 0 or dtype == torch.float32 else 0.01
                        torch.testing.assert_close(got.double(), want, rtol=tolerance, atol=2e-6)
                        max_scaled_error = max(
                            max_scaled_error,
                            float(((got.double() - want).abs() / (1 + want.abs())).max()),
                        )
                    if (value > 0).any():
                        raise RuntimeError("A count probability exceeded one")

                value = log_probability(mu, log_alpha)
                gradients = torch.autograd.grad(value.sum(), (mu, log_alpha))
                value = value.detach()
                check(value, gradients)
                vmapped = torch.vmap(
                    torch.func.grad(lambda m, a: log_probability(m, a).sum(), argnums=(0, 1))
                )(mu.detach().repeat(2, 1), log_alpha.detach().repeat(2, 1))
                for member in range(2):
                    check(value, tuple(gradient[member] for gradient in vmapped))
                graph, stream, _capture_inputs, captured, captured_gradients = (
                    _capture_count_gradients(log_probability, mu, log_alpha)
                )
                for _ in range(2):
                    with torch.cuda.stream(stream):
                        graph.replay()
                    torch.cuda.synchronize(device)
                    check(captured.detach(), captured_gradients)
                cases += 1
    return {
        "cuda_available": 1.0,
        "sm": float(major * 10 + minor),
        "passed_cases": float(cases),
        "graph_replays": float(cases * 2),
        "max_scaled_error": max_scaled_error,
    }
