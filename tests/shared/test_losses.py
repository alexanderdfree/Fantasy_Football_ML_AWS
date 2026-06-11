"""Unit tests for the per-head loss primitives in src.shared.training."""

import numpy as np
import pytest
import torch
from scipy.stats import nbinom, poisson

from src.shared.training import (
    MultiTargetLoss,
    hurdle_negbin_value_loss,
    hurdle_poisson_value_loss,
    negbin2_log_prob,
    ztnb2_log_prob,
    ztp_log_prob,
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
class TestZTPLogProb:
    """Zero-truncated Poisson = Poisson conditioned on y>=1. Verify
    log P = log P_Pois - log(1 - P_Pois(0))."""

    @pytest.mark.parametrize("mu,k", [(0.5, 1), (1.0, 2), (3.0, 5), (10.0, 12)])
    def test_matches_scipy_poisson(self, mu, k):
        log_p_k = poisson.logpmf(k, mu=mu)
        log_p_zero = poisson.logpmf(0, mu=mu)
        expected = log_p_k - np.log(1.0 - np.exp(log_p_zero))

        got = ztp_log_prob(torch.tensor(float(k)), torch.tensor(float(mu))).item()
        assert np.isclose(got, expected, atol=1e-5), (
            f"mu={mu}, k={k}: got {got}, expected {expected}"
        )

    def test_normalizes_to_one(self):
        """Sum over all k>=1 of P_ZTP should equal 1."""
        mu = torch.tensor(2.0)
        total = 0.0
        for k in range(1, 200):
            total += float(torch.exp(ztp_log_prob(torch.tensor(float(k)), mu)))
        assert abs(total - 1.0) < 1e-4


@pytest.mark.unit
class TestHurdlePoissonValueLoss:
    def test_skips_zero_only_batch(self):
        """All-zero batch should produce zero value loss (ZTP undefined at y=0)."""
        preds = {"y_value_mu": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"y": torch.tensor([0.0, 0.0, 0.0])}
        loss = hurdle_poisson_value_loss(preds, targets, "y")
        assert loss.item() == 0.0

    def test_positive_loss_when_positives_exist(self):
        """Mixed batch should produce non-zero loss scaled by fraction positive."""
        preds = {"y_value_mu": torch.tensor([2.0, 2.0, 2.0, 2.0])}
        targets = {"y": torch.tensor([0.0, 3.0, 0.0, 1.0])}
        loss = hurdle_poisson_value_loss(preds, targets, "y")
        assert loss.item() > 0

    def test_scales_by_fraction_positive(self):
        """Adding zero-rows to a batch with the same positives halves frac_pos
        and so halves the value loss."""
        preds_short = {"y_value_mu": torch.tensor([2.0, 2.0])}
        targets_short = {"y": torch.tensor([1.0, 3.0])}  # frac_pos = 1.0
        loss_short = hurdle_poisson_value_loss(preds_short, targets_short, "y").item()

        preds_long = {"y_value_mu": torch.tensor([2.0, 2.0, 2.0, 2.0])}
        targets_long = {"y": torch.tensor([1.0, 3.0, 0.0, 0.0])}  # frac_pos = 0.5
        loss_long = hurdle_poisson_value_loss(preds_long, targets_long, "y").item()

        assert abs(loss_long - 0.5 * loss_short) < 1e-5

    def test_ignores_log_alpha(self):
        """log_alpha emission from GatedHead must be ignored (no NegBin term)."""
        preds_a = {
            "y_value_mu": torch.tensor([2.0, 2.0]),
            "y_value_log_alpha": torch.tensor([0.0, 0.0]),
        }
        preds_b = {
            "y_value_mu": torch.tensor([2.0, 2.0]),
            "y_value_log_alpha": torch.tensor([5.0, -3.0]),  # wildly different
        }
        targets = {"y": torch.tensor([1.0, 2.0])}
        la = hurdle_poisson_value_loss(preds_a, targets, "y").item()
        lb = hurdle_poisson_value_loss(preds_b, targets, "y").item()
        assert la == pytest.approx(lb)


@pytest.mark.unit
class TestMultiTargetLossDispatch:
    def test_rejects_unsupported_loss(self):
        with pytest.raises(ValueError, match="Unsupported head_losses"):
            MultiTargetLoss(
                target_names=["a"],
                loss_weights={"a": 1.0},
                head_losses={"a": "not_a_real_loss"},
            )

    def test_mse_head_dispatches_as_squared_error(self):
        """head_losses={'a':'mse'} routes through nn.MSELoss and equals mean(e^2)."""
        loss_fn = MultiTargetLoss(
            target_names=["a"],
            loss_weights={"a": 1.0},
            head_losses={"a": "mse"},
        )
        preds = {"a": torch.tensor([1.0, 2.0, 3.0])}
        targets = {"a": torch.tensor([1.5, 0.0, 3.0])}
        combined, comps = loss_fn(preds, targets)
        expected = torch.mean((preds["a"] - targets["a"]) ** 2)
        assert comps["loss_a"] == pytest.approx(expected.item())
        assert combined.item() == pytest.approx(expected.item())

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

    def test_hurdle_poisson_dispatches_and_zeros_on_all_zero_batch(self):
        """End-to-end: head_losses={'y':'hurdle_poisson'} routes through the
        new path; an all-zero target batch yields zero value loss but still
        emits the BCE gate contribution (component key present)."""
        preds = {
            "y": torch.tensor([0.1, 0.1, 0.1]),
            "y_gate_logit": torch.tensor([-2.0, -2.0, -2.0]),
            "y_value_mu": torch.tensor([1.0, 2.0, 3.0]),
            "y_value_log_alpha": torch.tensor([0.0, 0.0, 0.0]),
        }
        targets = {"y": torch.tensor([0.0, 0.0, 0.0])}
        loss_fn = MultiTargetLoss(
            target_names=["y"],
            loss_weights={"y": 1.0},
            head_losses={"y": "hurdle_poisson"},
            gated_targets=["y"],
        )
        combined, comps = loss_fn(preds, targets)
        # Value loss is 0 (no positives); gate BCE is non-zero (targets are
        # all 0, gate predicts low prob, so BCE is small but positive).
        assert comps["loss_y"] == 0.0
        assert "loss_gate_y" in comps and comps["loss_gate_y"] > 0
        assert combined.item() == pytest.approx(comps["loss_gate_y"])

    def test_hurdle_poisson_requires_gated_target(self):
        """Misconfiguration: hurdle_poisson without gate membership raises."""
        with pytest.raises(ValueError, match="hurdle_poisson"):
            MultiTargetLoss(
                target_names=["y"],
                loss_weights={"y": 1.0},
                head_losses={"y": "hurdle_poisson"},
                gated_targets=[],
            )


@pytest.mark.unit
class TestTrainBranchLossPath:
    """The train branch calls ``_compute_loss_components`` directly (not
    ``forward``) to skip the per-component ``.item()`` GPU syncs whose float
    dict it discards. These tests pin the two paths to the same combined
    loss and a working backward, so that call-site swap stays inert."""

    def _loss_fn(self):
        return MultiTargetLoss(
            target_names=["a", "b"],
            loss_weights={"a": 2.0, "b": 0.5},
            head_losses={"a": "huber", "b": "poisson_nll"},
        )

    def _batch(self):
        preds = {
            "a": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            "b": torch.tensor([0.5, 1.5, 2.5], requires_grad=True),
        }
        targets = {
            "a": torch.tensor([1.5, 0.0, 3.0]),
            "b": torch.tensor([1.0, 1.0, 2.0]),
        }
        return preds, targets

    def test_combined_identical_to_forward(self):
        loss_fn = self._loss_fn()
        preds, targets = self._batch()
        combined_direct, comps_t = loss_fn._compute_loss_components(preds, targets)
        combined_forward, comps_f = loss_fn(preds, targets)
        assert combined_direct.item() == combined_forward.item()
        # Contract split: direct path keeps tensors on-device; forward
        # returns the historical float dict.
        assert all(torch.is_tensor(v) for v in comps_t.values())
        assert all(isinstance(v, float) for v in comps_f.values())
        assert set(comps_t) == set(comps_f)

    def test_backward_flows_through_direct_path(self):
        loss_fn = self._loss_fn()
        preds, targets = self._batch()
        combined, _ = loss_fn._compute_loss_components(preds, targets)
        combined.backward()
        assert preds["a"].grad is not None
        assert preds["b"].grad is not None
