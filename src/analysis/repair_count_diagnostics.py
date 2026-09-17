"""Evaluate observed count likelihood values and gradients, without fitting."""

from __future__ import annotations

import numpy as np
import torch


def stable_ztnb_reference(y, mu, log_alpha):
    """Independent FP64 finite rising product for observed integer counts."""
    alpha = log_alpha.exp().clamp(min=1e-6)
    if (y < 1).any() or (y != y.round()).any() or y.max() > 128:
        raise ValueError("Reference requires positive integer counts no larger than 128")
    coefficient = torch.zeros_like(mu)
    for k in range(int(y.max().item())):
        coefficient = coefficient + torch.where(y > k, torch.log1p(alpha * k), 0.0)
    log_zero = -torch.log1p(alpha * mu) / alpha
    return (
        coefficient
        + y * mu.log()
        - torch.lgamma(y + 1)
        - y * torch.log1p(alpha * mu)
        + log_zero
        - torch.log(-torch.expm1(log_zero))
    )


def observed_likelihood_check(actuals, mu, log_alpha):
    """Test every observed positive; synthetic extremes cannot authorize a fix."""
    from src.shared.training import ztnb2_log_prob

    mask = np.asarray(actuals) > 0
    if not mask.any():
        raise ValueError("No observed positive counts to assess")
    outputs = []
    for dtype, function in (
        (torch.float32, ztnb2_log_prob),
        (torch.float64, stable_ztnb_reference),
    ):
        y = torch.tensor(np.asarray(actuals)[mask], dtype=dtype)
        m = torch.tensor(np.asarray(mu)[mask], dtype=dtype, requires_grad=True)
        a = torch.tensor(np.asarray(log_alpha)[mask], dtype=dtype, requires_grad=True)
        value = function(y, m, a)
        gradients = torch.autograd.grad(value.sum(), (m, a))
        outputs.append([v.detach().double().numpy() for v in (value, *gradients)])
    report = {"n_positive": int(mask.sum()), "active_numerical_defect": False, "errors": {}}
    for name, actual, expected in zip(
        ("log_probability", "d_mu", "d_log_alpha"), *outputs, strict=True
    ):
        scaled = np.abs(actual - expected) / (1 + np.abs(expected))
        failed = ~np.isfinite(actual) | (scaled > 1e-4)
        report["errors"][name] = {
            "max_scaled_error": float(np.nan_to_num(scaled, nan=np.inf).max()),
            "n_outside_tolerance": int(failed.sum()),
        }
        report["active_numerical_defect"] |= bool(failed.any())
    report["observed_ranges"] = {
        name: [float(np.min(values)), float(np.max(values))]
        for name, values in (
            ("mu", np.asarray(mu)[mask]),
            ("log_alpha", np.asarray(log_alpha)[mask]),
        )
    }
    return report


def error_summary(predictions, truth):
    targets = sorted(truth)
    errors = np.column_stack([np.asarray(predictions[t]) - np.asarray(truth[t]) for t in targets])
    return {
        "targets": targets,
        "error_covariance": np.cov(errors, rowvar=False).tolist(),
        "raw_metrics": {
            t: {
                "mae": float(np.abs(errors[:, i]).mean()),
                "rmse": float(np.sqrt(np.square(errors[:, i]).mean())),
            }
            for i, t in enumerate(targets)
        },
    }
