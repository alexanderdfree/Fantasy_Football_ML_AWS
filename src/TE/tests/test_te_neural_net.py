"""Tests for src.shared.neural_net.MultiHeadNet (using TE targets)."""

import numpy as np
import pytest
import torch

from src.shared.neural_net import MultiHeadNet
from src.TE.te_config import TE_TARGETS

pytestmark = pytest.mark.unit


class TestMultiHeadNet:
    @pytest.fixture
    def model(self):
        return MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.1,
        )

    def test_output_keys(self, model):
        x = torch.randn(4, 10)
        out = model(x)
        assert set(out.keys()) == set(TE_TARGETS)

    def test_output_shapes(self, model):
        batch_size = 8
        x = torch.randn(batch_size, 10)
        out = model(x)
        for key in out:
            assert out[key].shape == (batch_size,)

    def test_custom_backbone(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=TE_TARGETS,
            backbone_layers=[64, 32, 16],
        )
        x = torch.randn(2, 5)
        out = model(x)
        for t in TE_TARGETS:
            assert out[t].shape == (2,)

    def test_single_sample_eval_mode(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[16, 8],
        )
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10)
            out = model(x)
        for t in TE_TARGETS:
            assert out[t].shape == (1,)

    def test_predict_numpy(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[16, 8],
        )
        X = np.random.randn(5, 10).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)

        assert set(preds.keys()) == set(TE_TARGETS)
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (5,)

    def test_predict_numpy_single_sample(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=TE_TARGETS,
            backbone_layers=[8, 4],
        )
        X = np.random.randn(1, 5).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)
        for t in TE_TARGETS:
            assert preds[t].shape == (1,)

    def test_gradients_flow(self, model):
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = sum(out[t].sum() for t in TE_TARGETS)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 10)

    def test_gradient_near_zero(self):
        # Seed BEFORE constructing the model so layer init (which consumes
        # torch's RNG state) is deterministic across test-suite orderings.
        torch.manual_seed(0)
        model = MultiHeadNet(
            input_dim=5,
            target_names=TE_TARGETS,
            backbone_layers=[16],
            head_hidden=4,
            dropout=0.0,
        )
        model.train()
        x = torch.randn(4, 5) * 0.01
        x.requires_grad_(True)
        out = model(x)
        loss = sum(out[t].sum() for t in TE_TARGETS)
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert (x.grad != 0).any()

    def test_single_backbone_layer(self):
        """Single-layer backbone should work correctly."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[96],
            head_hidden=32,
            dropout=0.2,
        )
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in TE_TARGETS:
            assert out[key].shape == (4,)
            assert (out[key] >= 0).all()

    def test_dropout_effect(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[32, 16],
            dropout=0.5,
        )
        x = torch.randn(8, 10)

        model.train()
        torch.manual_seed(0)
        out_train = model(x)

        model.eval()
        with torch.no_grad():
            out_eval = model(x)

        train_sum = sum(out_train[t].detach() for t in TE_TARGETS)
        eval_sum = sum(out_eval[t] for t in TE_TARGETS)
        assert not torch.allclose(train_sum, eval_sum)

    def test_outputs_non_negative_eval(self, model):
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in TE_TARGETS:
            assert (out[key] >= 0).all(), f"Negative value in {key} (eval)"

    def test_outputs_non_negative_train(self, model):
        model.train()
        x = torch.randn(4, 10)
        out = model(x)
        for key in TE_TARGETS:
            assert (out[key] >= 0).all(), f"Negative value in {key} (train)"

    def test_no_nan_output(self, model):
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in out:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_large_input_values(self, model):
        model.eval()
        x = torch.randn(4, 10) * 1000
        with torch.no_grad():
            out = model(x)
        for key in out:
            assert not torch.isnan(out[key]).any()

    def test_head_hidden_overrides(self):
        """TE config uses head_hidden_overrides on receiving_tds."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=TE_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            head_hidden_overrides={"receiving_tds": 32},
        )
        x = torch.randn(4, 10)
        out = model(x)
        for t in TE_TARGETS:
            assert out[t].shape == (4,)
        td_head = model.heads["receiving_tds"]
        assert td_head[0].out_features == 32
