"""Experiment-only likelihood candidate, copied verbatim from PR #1613.

Source: 00873f2b9dbddfc338b53604555ccc9ae9bac97f. No production caller imports this module.
"""

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


def _stirling_correction(x: torch.Tensor) -> torch.Tensor:
    """Six Bernoulli terms; callers shift gamma arguments to at least nine."""
    inv = x.reciprocal()
    sq = inv.square()
    return inv * (
        1 / 12
        + sq * (-1 / 360 + sq * (1 / 1260 + sq * (-1 / 1680 + sq * (1 / 1188 - sq * 691 / 360360))))
    )


def _nb2_log_positive_ratio(
    y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """Log(P_NB(y)/P_NB(1)) without subtracting nearly equal lgamma values.

    Write the gamma ratio as Gamma(b+n)/Gamma(b), b=1+1/alpha,
    n=y-1. Eight recurrence shifts put both arguments >= 9, where the
    six-term Stirling remainder is below 4e-15. Expand their difference
    with log1p before evaluation. The shift count is numerical precision,
    not a bound on y; this has fixed tensor shapes for capture and vmap.
    """
    beta = torch.exp(-log_alpha)
    base = 1 + beta
    shifted = base + 8
    n = y - 1
    q = n / shifted
    remainder = n * _log1p_div_minus_one(q) + (n - 0.5) * torch.log1p(q)
    remainder = remainder + _stirling_correction(shifted + n) - _stirling_correction(shifted)
    offsets = torch.arange(8, device=mu.device, dtype=mu.dtype)
    recurrence = torch.log1p(n.unsqueeze(-1) / (base.unsqueeze(-1) + offsets)).sum(-1)

    log_mu = mu.log()
    log_alpha_mu = log_alpha + log_mu
    log_p = -torch.logaddexp(torch.zeros_like(log_alpha_mu), -log_alpha_mu)
    # Combine log(9+1/alpha) + log(alpha*mu/(1+alpha*mu)) analytically
    # near the Poisson limit, so log(alpha) cannot cancel its own gradient.
    poisson_limit = (
        log_mu - r + torch.logaddexp(torch.zeros_like(log_alpha), log_alpha + math.log(9))
    )
    combined = torch.where(beta >= 1, poisson_limit, shifted.log() + log_p)
    return n * combined + remainder - recurrence - torch.lgamma(y + 1)


def _nb2_log_prob(
    y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor, *, truncated: bool
) -> torch.Tensor:
    mu, log_alpha, r, z = _nb2_zero_mass_terms(mu, log_alpha)
    safe_y = torch.where(y > 0, y, torch.ones_like(y))
    # P1 | Y>0 = exprel(-r) / exprel(z). This avoids both
    # subtraction of P0 from one and cancellation of 1/mu gradients.
    log_p1 = _log_exprel(-r) - _log_exprel(z) if truncated else mu.log() - r - z
    positive = log_p1 + _nb2_log_positive_ratio(safe_y, mu, log_alpha, r)
    log_p0 = torch.full_like(positive, float("-inf")) if truncated else -z
    return torch.where(y > 0, positive, log_p0)


def negbin2_log_prob(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Log-pmf of NB-2: mean ``mu``, ``var = mu + alpha*mu^2``; supports y=0."""
    y, mu, alpha = _count_loss_inputs(y, mu, alpha)
    return _nb2_log_prob(y, mu, alpha.clamp_min(1e-6).log(), truncated=False)


def ztnb2_log_prob(y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """Zero-truncated NB-2 log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu, alpha) = log P_NB(k) - log(1 - P_NB(0))``.
    """
    y, mu, log_alpha = _count_loss_inputs(y, mu, log_alpha)
    return _nb2_log_prob(y, mu, log_alpha, truncated=True)


def ztp_log_prob(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Zero-truncated Poisson log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu) = k*log(mu) - mu - lgamma(k+1) - log(1 - exp(-mu))``.
    Mirrors ztnb2 but without the dispersion parameter — appropriate when
    empirical var/mean ≈ 1 (e.g. RB rushing_tds ≈ 1.16, fumbles_lost ≈ 1.01).
    """
    y, mu = _count_loss_inputs(y, mu)
    mu = mu.clamp_min(1e-10)
    safe_y = torch.where(y > 0, y, torch.ones_like(y))
    # P1 | Y>0 = 1/exprel(mu); higher counts differ by mu**(y-1)/y!.
    log_p = -_log_exprel(mu) + (safe_y - 1) * mu.log() - torch.lgamma(safe_y + 1)
    return torch.where(y > 0, log_p, torch.full_like(log_p, float("-inf")))
