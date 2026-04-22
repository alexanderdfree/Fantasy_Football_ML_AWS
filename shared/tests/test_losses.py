"""Unit tests for the per-head loss primitives in shared.training."""

import numpy as np
import pytest
import torch
from scipy.stats import nbinom

from shared.training import (
    MultiTargetLoss,
    hurdle_negbin_value_loss,
    negbin2_log_prob,
    ztnb2_log_prob,
)


@pytest.mark.unit
class TestNegBin2LogProb:
    """Cross-check against scipy.stats.nbinom using the NB-2 parameter map:
    r = 1/alpha (total_count), p = r/(r+mu) (success probability)."""

    @pytest.mark.parametrize("mu,alpha", [(1.0, 0.1), (3.0, 1.0), (10.0, 5.0)])
    @pytest.mark.parametrize("k", [0, 1, 3, 10])
    def test_matches_scipy_nbinom(self, mu, alpha, k):
        r = 1.0 / alpha
        p = r / (r + mu)
        expected = nbinom.logpmf(k, n=r, p=p)

        got = negbin2_log_prob(
            torch.tensor(float(k)),
            torch.tensor(float(mu)),
            torch.tensor(float(alpha)),
        ).item()
        assert np.isclose(got, expected, atol=1e-5), (
            f"mu={mu}, alpha={alpha}, k={k}: got {got}, expected {expected}"
        )

    def test_mean_and_variance_match(self):
        """Monte Carlo sanity: samples from NegBin(r=1/alpha, p=r/(r+mu))
        should have mean ~ mu and variance ~ mu + alpha*mu^2."""
        torch.manual_seed(0)
        mu, alpha = 3.0, 1.0
        r = 1.0 / alpha
        p = r / (r + mu)
        samples = torch.tensor(np.random.default_rng(0).negative_binomial(r, p, size=50000))
        assert abs(samples.float().mean().item() - mu) < 0.1
        expected_var = mu + alpha * mu**2
        assert abs(samples.float().var().item() - expected_var) < 0.5


@pytest.mark.unit
class TestZTNB2LogProb:
    """Zero-truncated NB-2 = NB-2 conditioned on y>=1. Verify log P = log P_NB - log(1 - P_NB(0))."""

    @pytest.mark.parametrize("mu,alpha,k", [(1.0, 0.5, 1), (3.0, 1.0, 2), (10.0, 2.0, 5)])
    def test_matches_manual_formula(self, mu, alpha, k):
        r = 1.0 / alpha
        p = r / (r + mu)
        log_p_k = nbinom.logpmf(k, n=r, p=p)
        log_p_zero = nbinom.logpmf(0, n=r, p=p)
        expected = log_p_k - np.log(1.0 - np.exp(log_p_zero))

        got = ztnb2_log_prob(
            torch.tensor(float(k)),
            torch.tensor(float(mu)),
            torch.tensor(float(np.log(alpha))),
        ).item()
        assert np.isclose(got, expected, atol=1e-5), (
            f"mu={mu}, alpha={alpha}, k={k}: got {got}, expected {expected}"
        )

    def test_normalizes_to_one(self):
        """Sum over all k>=1 of P_ZTNB should equal 1."""
        mu, alpha = 2.0, 1.0
        log_alpha = torch.tensor(float(np.log(alpha)))
        total = 0.0
        for k in range(1, 200):  # enough to cover the tail
            total += float(
                torch.exp(ztnb2_log_prob(torch.tensor(float(k)), torch.tensor(mu), log_alpha))
            )
        assert abs(total - 1.0) < 1e-4


@pytest.mark.unit
class TestHurdleNegBinValueLoss:
    def test_skips_zero_only_batch(self):
        """All-zero batch should produce zero value loss (ZTNB undefined at y=0)."""
        preds = {
            "receptions_value_mu": torch.tensor([1.0, 2.0, 3.0]),
            "receptions_value_log_alpha": torch.tensor([0.0, 0.0, 0.0]),
        }
        targets = {"receptions": torch.tensor([0.0, 0.0, 0.0])}
        loss = hurdle_negbin_value_loss(preds, targets, "receptions")
        assert loss.item() == 0.0

    def test_positive_loss_when_positives_exist(self):
        """Mixed batch should produce non-zero loss scaled by fraction positive."""
        preds = {
            "receptions_value_mu": torch.tensor([2.0, 2.0, 2.0, 2.0]),
            "receptions_value_log_alpha": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        }
        targets = {"receptions": torch.tensor([0.0, 3.0, 0.0, 1.0])}
        loss = hurdle_negbin_value_loss(preds, targets, "receptions")
        assert loss.item() > 0

    def test_scales_by_fraction_positive(self):
        """Doubling the batch with extra zeros halves the scaling factor."""
        preds_short = {
            "y_value_mu": torch.tensor([2.0, 2.0]),
            "y_value_log_alpha": torch.tensor([0.0, 0.0]),
        }
        targets_short = {"y": torch.tensor([1.0, 3.0])}  # frac_pos = 1.0
        loss_short = hurdle_negbin_value_loss(preds_short, targets_short, "y").item()

        preds_long = {
            "y_value_mu": torch.tensor([2.0, 2.0, 2.0, 2.0]),
            "y_value_log_alpha": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        }
        targets_long = {"y": torch.tensor([1.0, 3.0, 0.0, 0.0])}  # frac_pos = 0.5
        loss_long = hurdle_negbin_value_loss(preds_long, targets_long, "y").item()

        # ZTNB mean is identical in both (same y=1,3 samples); only frac_pos differs.
        assert abs(loss_long - 0.5 * loss_short) < 1e-5


@pytest.mark.unit
class TestMultiTargetLossDispatch:
    def test_rejects_unsupported_loss(self):
        with pytest.raises(ValueError, match="Unsupported head_losses"):
            MultiTargetLoss(
                target_names=["a"],
                loss_weights={"a": 1.0},
                head_losses={"a": "not_a_real_loss"},
            )

    def test_poisson_targets_alias_maps_to_head_losses(self):
        """poisson_targets=['a'] should be equivalent to head_losses={'a': 'poisson_nll'}."""
        loss_a = MultiTargetLoss(
            target_names=["a", "b"],
            loss_weights={"a": 1.0, "b": 1.0},
            poisson_targets=["a"],
        )
        loss_b = MultiTargetLoss(
            target_names=["a", "b"],
            loss_weights={"a": 1.0, "b": 1.0},
            head_losses={"a": "poisson_nll", "b": "huber"},
        )
        assert loss_a.head_losses == loss_b.head_losses

    def test_hurdle_negbin_uses_per_sample_dispersion(self):
        """Different log_alpha values should produce different losses."""
        torch.manual_seed(0)
        preds_a = {
            "y": torch.tensor([1.0, 2.0]),
            "y_gate_logit": torch.tensor([1.0, 1.0]),
            "y_value_mu": torch.tensor([2.0, 2.0]),
            "y_value_log_alpha": torch.tensor([0.0, 0.0]),
        }
        preds_b = {k: v.clone() for k, v in preds_a.items()}
        preds_b["y_value_log_alpha"] = torch.tensor([1.0, 1.0])
        targets = {"y": torch.tensor([1.0, 3.0])}

        loss_fn = MultiTargetLoss(
            target_names=["y"],
            loss_weights={"y": 1.0},
            head_losses={"y": "hurdle_negbin"},
            gated_targets=["y"],
        )
        la, _ = loss_fn(preds_a, targets)
        lb, _ = loss_fn(preds_b, targets)
        assert la.item() != pytest.approx(lb.item())
