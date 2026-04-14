"""Tests for RB.rb_neural_net — RBMultiHeadNet."""

import numpy as np
import torch
import pytest

from RB.rb_neural_net import RBMultiHeadNet


class TestRBMultiHeadNet:
    @pytest.fixture
    def model(self):
        return RBMultiHeadNet(input_dim=10, backbone_layers=[32, 16], head_hidden=8, dropout=0.1)

    def test_output_keys(self, model):
        x = torch.randn(4, 10)
        out = model(x)
        assert set(out.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}

    def test_output_shapes(self, model):
        batch_size = 8
        x = torch.randn(batch_size, 10)
        out = model(x)
        for key in out:
            assert out[key].shape == (batch_size,)

    def test_total_equals_sum(self, model):
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        expected = out["rushing_floor"] + out["receiving_floor"] + out["td_points"]
        torch.testing.assert_close(out["total"], expected)

    def test_default_backbone_layers(self):
        model = RBMultiHeadNet(input_dim=20)
        x = torch.randn(4, 20)
        out = model(x)
        assert out["total"].shape == (4,)

    def test_custom_backbone(self):
        model = RBMultiHeadNet(input_dim=5, backbone_layers=[64, 32, 16])
        x = torch.randn(2, 5)
        out = model(x)
        assert out["total"].shape == (2,)

    def test_single_sample_eval_mode(self):
        """Batch norm can fail with batch_size=1 in train mode; eval mode should work."""
        model = RBMultiHeadNet(input_dim=10, backbone_layers=[16, 8])
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10)
            out = model(x)
        assert out["total"].shape == (1,)

    def test_predict_numpy(self):
        model = RBMultiHeadNet(input_dim=10, backbone_layers=[16, 8])
        X = np.random.randn(5, 10).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)

        assert set(preds.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (5,)

    def test_predict_numpy_single_sample(self):
        model = RBMultiHeadNet(input_dim=5, backbone_layers=[8, 4])
        X = np.random.randn(1, 5).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)
        assert preds["total"].shape == (1,)

    def test_gradients_flow(self, model):
        """Verify backward pass works (gradients reach input)."""
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out["total"].sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 10)

    def test_dropout_effect(self):
        """Train mode (dropout active) vs eval mode should give different outputs."""
        model = RBMultiHeadNet(input_dim=10, backbone_layers=[32, 16], dropout=0.5)
        x = torch.randn(8, 10)

        model.train()
        torch.manual_seed(0)
        out_train = model(x)

        model.eval()
        with torch.no_grad():
            out_eval = model(x)

        # With 50% dropout, outputs should differ between train and eval
        # (statistically near-certain with 8 samples)
        assert not torch.allclose(out_train["total"].detach(), out_eval["total"])

    def test_no_nan_output(self, model):
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in out:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_large_input_values(self, model):
        """Model should handle large feature values without NaN."""
        model.eval()
        x = torch.randn(4, 10) * 1000
        with torch.no_grad():
            out = model(x)
        for key in out:
            assert not torch.isnan(out[key]).any()
