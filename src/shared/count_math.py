"""Tensor-only numerical primitives shared by count heads and their likelihoods."""

import math

import torch


def _count_loss_inputs(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Keep AMP count arithmetic in FP32, while preserving FP64 callers."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if not dtype.is_floating_point or dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return tuple(tensor.to(dtype=dtype) for tensor in tensors)


def _log_exprel(x: torch.Tensor) -> torch.Tensor:
    """Stable log((exp(x)-1)/x), including its value and derivative at zero."""
    small = x.abs() < 0.1
    s = torch.where(small, x, torch.zeros_like(x))
    s2 = s.square()
    series = s / 2 + s2 * (
        1 / 24 + s2 * (-1 / 2880 + s2 * (1 / 181440 + s2 * (-1 / 9676800 + s2 / 479001600)))
    )
    # Sanitize both branches: torch.where still differentiates their arithmetic.
    q = torch.where(small, torch.full_like(x, 0.1), x)
    magnitude = q.abs()
    direct = q.clamp_min(0) + torch.log(-torch.expm1(-magnitude)) - magnitude.log()
    return torch.where(small, series, direct)


def _log1p_div_minus_one(x: torch.Tensor) -> torch.Tensor:
    """Stable log1p(x)/x - 1, including its derivative near zero."""
    small = x.abs() < 0.001
    s = torch.where(small, x, torch.zeros_like(x))
    series = s * (-1 / 2 + s * (1 / 3 + s * (-1 / 4 + s * (1 / 5 + s * (-1 / 6 + s / 7)))))
    q = torch.where(small, torch.full_like(x, 0.001), x)
    return torch.where(small, series, torch.log1p(q) / q - 1)


def _nb2_zero_mass_terms(
    mu: torch.Tensor, log_alpha: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Effective NB parameters, r=log1p(alpha*mu), and z=-log(P0)=r/alpha."""
    mu = mu.clamp_min(1e-10)
    log_alpha = log_alpha.clamp_min(math.log(1e-6))
    log_alpha_mu = log_alpha + mu.log()
    r = torch.logaddexp(torch.zeros_like(log_alpha_mu), log_alpha_mu)
    # Near zero, the log-alpha derivative of z must not subtract almost
    # equal chain-rule terms. Away from zero, avoid exp(log_alpha) overflow.
    small_product = log_alpha_mu < math.log(0.001)
    product = torch.exp(
        torch.where(small_product, log_alpha_mu, torch.full_like(log_alpha_mu, math.log(0.001)))
    )
    z = torch.where(
        small_product,
        mu * (1 + _log1p_div_minus_one(product)),
        torch.exp(r.log() - log_alpha),
    )
    return mu, log_alpha, r, z
