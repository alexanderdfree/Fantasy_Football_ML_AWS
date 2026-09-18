"""One measured count-likelihood repair candidate; not a production default."""

from __future__ import annotations

import torch


def stable_ztnb2_log_prob(y, mu, log_alpha):
    """Keep NB gamma differences and their gradients out of FP32 cancellation.

    Model parameters, predictions and optimizer state remain FP32. Only the
    probability arithmetic uses FP64, returning the original FP32 loss dtype.
    log1p/expm1 preserve the exact zero-truncated law without a survival clamp.
    All operations remain in Torch and are CUDA-graph capturable.
    """
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    count = y.double()
    mean = mu.double().clamp(min=1e-10)
    alpha = log_alpha.double().exp().clamp(min=1e-6)
    inverse = alpha.reciprocal()
    log_scale = torch.log1p(alpha * mean)
    log_zero = -inverse * log_scale
    coefficient = torch.lgamma(count + inverse) - torch.lgamma(inverse) - torch.lgamma(count + 1)
    log_mass = coefficient + log_zero + count * (mean.log() - inverse.log() - log_scale)
    return (log_mass - torch.log(-torch.expm1(log_zero))).to(dtype)


def select_numerical_candidate(enabled):
    """Change only this isolated A/B worker's shared likelihood primitive."""
    from src.shared import training

    original = getattr(training.ztnb2_log_prob, "_repair_original", training.ztnb2_log_prob)
    stable_ztnb2_log_prob._repair_original = original
    training.ztnb2_log_prob = stable_ztnb2_log_prob if enabled else original
