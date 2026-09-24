"""The CUDA count-likelihood probe must not label CPU execution as GPU evidence."""

import math

import pytest


@pytest.mark.unit
def test_count_probe_refuses_cpu(monkeypatch):
    import torch

    from src.analysis.verify_count_likelihoods import verify_count_likelihoods

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="actual CUDA"):
        verify_count_likelihoods()


@pytest.mark.unit
def test_count_oracle_has_closed_form_geometric_and_poisson_controls():
    from src.analysis.verify_count_likelihoods import _reference

    assert _reference(1, 1.0, 0.0) == pytest.approx((-math.log(2), -0.5, 2 * math.log(2) - 1.5))
    assert _reference(1, 1.0) == pytest.approx((-math.log(math.expm1(1)), -1 / math.expm1(1), 0.0))


@pytest.mark.unit
def test_capture_owns_fresh_leaves_and_releases_warmup_graphs(monkeypatch):
    """CPU tensors check lifetime/stream plumbing, not actual CUDA execution."""
    import weakref
    from contextlib import contextmanager

    import torch

    from src.analysis.verify_count_likelihoods import _capture_count_gradients

    waits = []

    class Stream:
        def wait_stream(self, other):
            waits.append((self, other))

    default, capture = Stream(), Stream()
    active = [default]
    graph = object()
    calls, outputs = [], []

    @contextmanager
    def use_stream(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    @contextmanager
    def capture_graph(requested, *, stream):
        assert requested is graph and stream is capture and active[0] is capture
        assert len(outputs) == 3
        assert all(output() is None for output in outputs)
        yield

    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: capture)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: active[0])
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", capture_graph)

    mu = torch.tensor([0.2, 1.0], requires_grad=True)
    alpha = torch.tensor([-5.0, 0.0], requires_grad=True)
    eager = mu.square() + alpha.square()
    torch.autograd.grad(eager.sum(), (mu, alpha))
    assert eager.grad_fn is not None  # Keep the earlier graph alive deliberately.

    def probability(m, a):
        assert active[0] is capture
        calls.append((m, a))
        output = m.square() + a.square()
        outputs.append(weakref.ref(output))
        return output

    got_graph, got_stream, inputs, value, gradients = _capture_count_gradients(
        probability, mu, alpha
    )
    assert got_graph is graph and got_stream is capture
    assert waits == [(capture, default), (default, capture)]
    assert active[0] is default
    assert len(calls) == 4
    for original, owned in zip((mu, alpha), inputs, strict=True):
        assert owned.is_leaf and owned.requires_grad and owned is not original
        assert owned.data_ptr() != original.data_ptr()
        torch.testing.assert_close(owned, original)
    assert all(m is inputs[0] and a is inputs[1] for m, a in calls)
    torch.testing.assert_close(value, eager)
    for actual, parameter in zip(gradients, inputs, strict=True):
        torch.testing.assert_close(actual, 2 * parameter)
