"""Exact count functions from frozen main a69a813d; diagnostic baseline only."""

import torch


def negbin2_log_prob(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Log-pmf of the NB-2 parameterization: mean ``mu``, ``var = mu + alpha*mu^2``.

    Equivalent to ``NegBin(r=1/alpha, p=r/(r+mu))``. Supports ``y=0``.
    """
    alpha = torch.clamp(alpha, min=1e-6)
    mu = torch.clamp(mu, min=1e-10)
    r = 1.0 / alpha
    log_coeff = torch.lgamma(y + r) - torch.lgamma(y + 1.0) - torch.lgamma(r)
    log_r_ratio = torch.log(r) - torch.log(r + mu)
    log_mu_ratio = torch.log(mu) - torch.log(r + mu)
    return log_coeff + r * log_r_ratio + y * log_mu_ratio


def ztnb2_log_prob(y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """Zero-truncated NB-2 log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu, alpha) = log P_NB(k) - log(1 - P_NB(0))``.
    """
    alpha = torch.exp(log_alpha)
    log_p = negbin2_log_prob(y, mu, alpha)
    log_p_zero = negbin2_log_prob(torch.zeros_like(y), mu, alpha)
    # log(1 - p_zero) via log1p for numerical stability when p_zero is small.
    log_survival = torch.log1p(-torch.exp(log_p_zero).clamp(max=1.0 - 1e-7))
    return log_p - log_survival


def ztp_log_prob(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Zero-truncated Poisson log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu) = k*log(mu) - mu - lgamma(k+1) - log(1 - exp(-mu))``.
    Mirrors ztnb2 but without the dispersion parameter — appropriate when
    empirical var/mean ≈ 1 (e.g. RB rushing_tds ≈ 1.16, fumbles_lost ≈ 1.01).
    """
    mu = torch.clamp(mu, min=1e-10)
    log_p = y * torch.log(mu) - mu - torch.lgamma(y + 1.0)
    log_survival = torch.log1p(-torch.exp(-mu).clamp(max=1.0 - 1e-7))
    return log_p - log_survival
