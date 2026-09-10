"""Count likelihoods retain probability mass and gradients near the Poisson limit."""

import math
from decimal import Decimal, localcontext

import pytest
import torch

from src.shared.neural_net import GatedHead
from src.shared.training import (
    hurdle_negbin_value_loss,
    hurdle_negbin_value_loss_capturable,
    hurdle_poisson_value_loss,
    hurdle_poisson_value_loss_capturable,
    negbin2_log_prob,
    ztnb2_log_prob,
    ztp_log_prob,
)

pytestmark = pytest.mark.unit


def _reference(count, mu, log_alpha=None, *, truncated=True):
    """Independent 90-digit pmf/derivatives, using the finite NB rising product."""
    with localcontext() as ctx:
        ctx.prec = 90
        y, m = Decimal(count), Decimal(str(mu))
        log_factorial = Decimal(math.factorial(count)).ln()
        if log_alpha is None:
            p0 = (-m).exp()
            log_p = y * m.ln() - m - log_factorial
            d_mu = y / m - 1
            if truncated:
                log_p -= (1 - p0).ln()
                d_mu -= p0 / (1 - p0)
            return float(log_p), float(d_mu)

        raw_alpha = Decimal(str(log_alpha)).exp()
        a = max(raw_alpha, Decimal("1e-6"))
        r = (1 + a * m).ln()
        p0 = (-r / a).exp()
        log_p = (
            sum((1 + a * k).ln() for k in range(count))
            + y * m.ln()
            - log_factorial
            - (y + 1 / a) * r
        )
        d_mu = y / m - (1 + a * y) / (1 + a * m)
        d_alpha = (
            sum(Decimal(k) / (1 + a * k) for k in range(count))
            + r / a**2
            - (y + 1 / a) * m / (1 + a * m)
        )
        if truncated:
            log_p -= (1 - p0).ln()
            d_mu -= p0 / ((1 - p0) * (1 + a * m))
            d_alpha += p0 / (1 - p0) * (r / a**2 - m / (a * (1 + a * m)))
        d_log_alpha = d_alpha * a if raw_alpha > Decimal("1e-6") else Decimal(0)
        return float(log_p), float(d_mu), float(d_log_alpha)


_NB_CASES = [
    (1, 2e-10, 2e-6),
    (1, 2e-6, math.exp(-5)),
    (1, 1e-5, 0.1),
    (2, 2e-6, 2e-6),
    (3, 0.01, 2e-6),
    (1, 0.1, 0.1),
    (1, 0.001 * 0.99999, 1.0),
    (1, 0.001 * 1.00001, 1.0),
    (1, math.expm1(0.1) * 0.99999, 1.0),
    (1, math.expm1(0.1) * 1.00001, 1.0),
    (1, 1.0, 1.0),
    (3, 1.0, 0.99999),
    (3, 1.0, 1.00001),
    (5, 3.0, 1.0),
    (30, 20.0, 3.0),
    (100, 3.0, 2e-6),
    (1000, 100.0, 1e6),
    (1, 2e-6, math.exp(90)),
    (4, 2e-6, math.exp(90)),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("truncated", [False, True])
def test_nb_probability_and_parameter_gradients_match_decimal(dtype, truncated):
    cases = _NB_CASES if truncated else [(0, 2e-6, 2e-6), (0, 3.0, 0.1), *_NB_CASES]
    if not truncated:
        # The ordinary NB API takes alpha itself, not log(alpha): infinity is
        # not a valid finite parameter. The log-domain hurdle API tests all cases.
        cases = [row for row in cases if row[2] < torch.finfo(dtype).max]
    y = torch.tensor([row[0] for row in cases], dtype=dtype)
    mu = torch.tensor([row[1] for row in cases], dtype=dtype, requires_grad=True)
    log_alpha = torch.tensor([math.log(row[2]) for row in cases], dtype=dtype, requires_grad=True)
    got = (
        ztnb2_log_prob(y, mu, log_alpha) if truncated else negbin2_log_prob(y, mu, log_alpha.exp())
    )
    gradients = torch.autograd.grad(got.sum(), (mu, log_alpha))
    expected = torch.tensor(
        [
            _reference(int(k), float(m), float(a), truncated=truncated)
            for k, m, a in zip(y.detach(), mu.detach(), log_alpha.detach(), strict=True)
        ],
        dtype=dtype,
    )
    rtol, atol = (1.2e-4, 5e-6) if dtype == torch.float32 else (3e-12, 2e-12)
    for actual, reference in zip((got, *gradients), expected.T, strict=True):
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    assert torch.all(got <= 0)


@pytest.mark.parametrize("family", ["nb", "poisson"])
@pytest.mark.parametrize("mu_value", [2e-10, 2e-6, 1e-5])
def test_small_rate_p1_and_gradient_are_accurate_without_probability_clamping(family, mu_value):
    mu = torch.tensor(mu_value, requires_grad=True)
    log_alpha = torch.tensor(-5.0, requires_grad=True)
    y = torch.ones_like(mu)
    log_p = ztnb2_log_prob(y, mu, log_alpha) if family == "nb" else ztp_log_prob(y, mu)
    parameters = (mu, log_alpha) if family == "nb" else (mu,)
    gradients = torch.autograd.grad(log_p, parameters)
    expected = _reference(1, float(mu.detach()), -5.0 if family == "nb" else None)
    assert 0 < log_p.exp().item() <= 1
    for actual, reference in zip((log_p, *gradients), expected, strict=True):
        assert actual.item() == pytest.approx(reference, rel=5e-6, abs=1e-15)
    # An observation of one approaches certainty as the conditional rate goes to zero.
    assert gradients[0].item() < 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_poisson_probability_and_gradient_match_decimal(dtype):
    counts = [1, 2, 3, 10, 1000]
    mu = torch.tensor([2e-10, 2e-6, 0.1, 30.0, 1000.0], dtype=dtype, requires_grad=True)
    y = torch.tensor(counts, dtype=dtype)
    log_p = ztp_log_prob(y, mu)
    (grad,) = torch.autograd.grad(log_p.sum(), (mu,))
    expected = torch.tensor(
        [_reference(k, float(m)) for k, m in zip(counts, mu.detach(), strict=True)], dtype=dtype
    )
    rtol, atol = (5e-5, 2e-6) if dtype == torch.float32 else (2e-12, 2e-12)
    torch.testing.assert_close(log_p, expected[:, 0], rtol=rtol, atol=atol)
    torch.testing.assert_close(grad, expected[:, 1], rtol=rtol, atol=atol)


@pytest.mark.parametrize("family", ["nb", "poisson"])
@pytest.mark.parametrize("mu_value,alpha", [(2e-6, 2e-6), (1.0, 1.0), (20.0, 3.0)])
def test_conditional_distribution_normalizes_and_has_no_zero_mass(family, mu_value, alpha):
    y = torch.arange(4097, dtype=torch.float64)
    mu = torch.full_like(y, mu_value)
    log_p = (
        ztnb2_log_prob(y, mu, torch.full_like(y, math.log(alpha)))
        if family == "nb"
        else ztp_log_prob(y, mu)
    )
    assert log_p[0].exp().item() == 0
    assert log_p.exp().sum().item() == pytest.approx(1.0, rel=1e-10, abs=1e-12)


def test_real_head_low_rate_likelihood_has_correct_probability_and_gradient_direction():
    head = GatedHead(2, gate_hidden=4, value_hidden=4, loss_family="hurdle_negbin")
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.value_mu[0].bias.fill_(math.log(math.expm1(1e-6)))
        head.value_log_alpha.bias.fill_(-5.0)
    _, _, mu, log_alpha = head(torch.zeros(1, 2))
    loss = -ztnb2_log_prob(torch.ones_like(mu), mu, log_alpha).sum()
    loss.backward()
    expected = _reference(1, mu.item(), log_alpha.item())
    assert loss.item() == pytest.approx(-expected[0], rel=5e-6)
    # Correct NLL encourages a lower rate for a near-certain conditional one.
    assert head.value_mu[0].bias.grad.item() > 0
    assert head.value_log_alpha.bias.grad.item() > 0


def _mean_reference(mu, log_alpha):
    with localcontext() as ctx:
        ctx.prec = 90
        m, a = Decimal(str(mu)), Decimal(str(log_alpha)).exp()
        r = (1 + a * m).ln()
        z = r / a
        p0 = (-z).exp()
        survival = 1 - p0
        mean = m / survival
        d_mu = 1 / survival - m * p0 / (survival**2 * (1 + a * m))
        d_log_alpha = m * p0 / survival**2 * (z - m / (1 + a * m))
        return tuple(map(float, (mean, d_mu, d_log_alpha)))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_large_finite_dispersion_keeps_representable_mean_and_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires actual MPS")
    head = GatedHead(1, loss_family="hurdle_negbin").to(device)
    mu = torch.tensor(2e-6, device=device, requires_grad=True)
    log_alpha = torch.tensor(90.0, device=device, requires_grad=True)
    actual = head.conditional_mean(mu, log_alpha)
    expected = _mean_reference(float(mu.detach().cpu()), 90.0)
    gradients = torch.autograd.grad(actual, (mu, log_alpha))
    for value, reference in zip((actual, *gradients), expected, strict=True):
        assert value.item() == pytest.approx(reference, rel=2e-5)


def test_actual_large_dispersion_head_preserves_raw_parameters_and_finite_gradients():
    head = GatedHead(2, gate_hidden=4, value_hidden=4, loss_family="hurdle_negbin")
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.value_mu[0].bias.fill_(math.log(math.expm1(1e-6)))
        head.value_log_alpha.bias.fill_(90.0)
    prediction, gate_logit, mu, log_alpha = head(torch.zeros(1, 2))
    expected, _, _ = _mean_reference(mu.item(), 90.0)
    assert log_alpha.item() == 90.0
    assert prediction.item() == pytest.approx(expected * gate_logit.sigmoid().item(), rel=2e-5)
    prediction.sum().backward()
    for parameter in (head.gate[-1].bias, head.value_mu[0].bias, head.value_log_alpha.bias):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_low_precision_inputs_compute_loss_in_float32_and_keep_gradients(dtype, family):
    mu = torch.tensor([2e-6, 1.0], dtype=dtype, requires_grad=True)
    log_alpha = torch.tensor([-5.0, 0.0], dtype=dtype, requires_grad=True)
    y = torch.tensor([1.0, 2.0], dtype=dtype)
    loss = ztnb2_log_prob(y, mu, log_alpha) if family == "nb" else ztp_log_prob(y, mu)
    assert loss.dtype == torch.float32
    parameters = (mu, log_alpha) if family == "nb" else (mu,)
    gradients = torch.autograd.grad(loss.sum(), parameters)
    expected = torch.tensor(
        [
            _reference(int(k), float(m), float(a) if family == "nb" else None)
            for k, m, a in zip(y.detach(), mu.detach(), log_alpha.detach(), strict=True)
        ]
    )
    torch.testing.assert_close(loss, expected[:, 0], rtol=5e-6, atol=1e-12)
    for actual, reference in zip(gradients, expected[:, 1:].T, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), reference, rtol=0.01, atol=1e-7)


@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_capturable_mask_and_vmap_gradients_match_independent_sample_oracle(family):
    y = torch.tensor([0.0, 1.0, 2.0])
    mu = torch.tensor([1e-6, 2e-6, 0.2], requires_grad=True)
    log_alpha = torch.tensor([-5.0, -5.0, 0.0], requires_grad=True)
    eager, capturable = (
        (hurdle_negbin_value_loss, hurdle_negbin_value_loss_capturable)
        if family == "nb"
        else (hurdle_poisson_value_loss, hurdle_poisson_value_loss_capturable)
    )

    def value_loss(m, a):
        preds = {"count_value_mu": m, "count_value_log_alpha": a}
        return capturable(preds, {"count": y}, "count")

    actual = value_loss(mu, log_alpha)
    torch.testing.assert_close(
        actual,
        eager({"count_value_mu": mu, "count_value_log_alpha": log_alpha}, {"count": y}, "count"),
    )
    expected = [
        _reference(int(k), float(m), float(a) if family == "nb" else None)
        for k, m, a in zip(y[1:], mu.detach()[1:], log_alpha.detach()[1:], strict=True)
    ]
    assert actual.item() == pytest.approx(-sum(row[0] for row in expected) / 3, rel=2e-6)
    vmapped = torch.vmap(torch.func.grad(value_loss, argnums=(0, 1)))(
        mu.detach().repeat(2, 1), log_alpha.detach().repeat(2, 1)
    )
    for parameter_index, gradients in enumerate(vmapped):
        expected_grad = (
            [0.0, *[-row[parameter_index + 1] / 3 for row in expected]]
            if (family == "nb" or parameter_index == 0)
            else [0.0, 0.0, 0.0]
        )
        torch.testing.assert_close(
            gradients, torch.tensor([expected_grad, expected_grad]), rtol=5e-6, atol=1e-12
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires actual MPS")
@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_native_mps_probability_and_gradients_match_decimal(family):
    mu = torch.tensor([2e-6, 0.2], device="mps", requires_grad=True)
    log_alpha = torch.tensor([-5.0, 0.0], device="mps", requires_grad=True)
    y = torch.tensor([1.0, 2.0], device="mps")
    log_p = ztnb2_log_prob(y, mu, log_alpha) if family == "nb" else ztp_log_prob(y, mu)
    parameters = (mu, log_alpha) if family == "nb" else (mu,)
    gradients = torch.autograd.grad(log_p.sum(), parameters)
    expected = torch.tensor(
        [
            _reference(int(k), float(m), float(a) if family == "nb" else None)
            for k, m, a in zip(y.cpu(), mu.detach().cpu(), log_alpha.detach().cpu(), strict=True)
        ]
    )
    for actual, reference in zip((log_p, *gradients), expected.T, strict=True):
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA capture")
@pytest.mark.parametrize("family", ["nb", "poisson"])
def test_native_cuda_capture_replays_count_loss_and_gradients(family):
    y = torch.tensor([0.0, 1.0, 2.0], device="cuda")
    mu = torch.tensor([1e-6, 2e-6, 0.2], device="cuda", requires_grad=True)
    log_alpha = torch.tensor([-5.0, -5.0, 0.0], device="cuda", requires_grad=True)

    def forward():
        preds = {"count_value_mu": mu, "count_value_log_alpha": log_alpha}
        value_loss = (
            hurdle_negbin_value_loss_capturable
            if family == "nb"
            else hurdle_poisson_value_loss_capturable
        )
        return value_loss(preds, {"count": y}, "count")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            mu.grad = log_alpha.grad = None
            forward().backward()
    torch.cuda.current_stream().wait_stream(stream)
    mu.grad = log_alpha.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = forward()
        loss.backward()
    mu.grad.zero_()
    if family == "nb":
        log_alpha.grad.zero_()
    graph.replay()
    expected = [
        _reference(1, float(mu[1]), -5.0 if family == "nb" else None),
        _reference(2, float(mu[2]), 0.0 if family == "nb" else None),
    ]
    assert loss.item() == pytest.approx(-sum(row[0] for row in expected) / 3, rel=1e-5)
    gradients = (mu.grad, log_alpha.grad) if family == "nb" else (mu.grad,)
    for index, gradient in enumerate(gradients, start=1):
        torch.testing.assert_close(
            gradient.cpu(),
            torch.tensor([0.0, *[-row[index] / 3 for row in expected]]),
            rtol=1e-5,
            atol=1e-10,
        )
