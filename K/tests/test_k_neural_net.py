"""Tests for shared.neural_net.MultiHeadNet (using Kicker targets).

Kickers use 4 non-negative raw-value targets: fg_yard_points, pat_points,
fg_misses, xp_misses. Signs are applied in the fantasy-total aggregation,
not on the heads themselves, so every head is clamped to >= 0.
"""

import numpy as np
import pytest
import torch

from shared.neural_net import MultiHeadNet

K_TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]
K_TARGETS_SET = set(K_TARGETS)


@pytest.mark.unit
class TestMultiHeadNet:
    @pytest.fixture
    def model(self):
        return MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.1,
        )

    def test_output_keys(self, model):
        """Kicker model exposes the 4 K heads plus the unsigned total."""
        x = torch.randn(4, 10)
        out = model(x)
        assert set(out.keys()) == K_TARGETS_SET | {"total"}

    def test_output_shapes(self, model):
        batch_size = 8
        x = torch.randn(batch_size, 10)
        out = model(x)
        for key in out:
            assert out[key].shape == (batch_size,)

    def test_total_equals_sum(self, model):
        """Total = unsigned sum of the 4 K heads (training label target)."""
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        expected = sum(out[t] for t in K_TARGETS)
        torch.testing.assert_close(out["total"], expected)

    def test_custom_backbone(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=K_TARGETS,
            backbone_layers=[64, 32, 16],
        )
        x = torch.randn(2, 5)
        out = model(x)
        assert out["total"].shape == (2,)

    def test_single_sample_eval_mode(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[16, 8],
        )
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10)
            out = model(x)
        assert out["total"].shape == (1,)

    def test_predict_numpy(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[16, 8],
        )
        X = np.random.randn(5, 10).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)

        assert set(preds.keys()) == K_TARGETS_SET | {"total"}
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (5,)

    def test_predict_numpy_single_sample(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=K_TARGETS,
            backbone_layers=[8, 4],
        )
        X = np.random.randn(1, 5).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)
        assert preds["total"].shape == (1,)

    def test_gradients_flow(self, model):
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out["total"].sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 10)

    def test_gradient_near_zero(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=K_TARGETS,
            backbone_layers=[16],
            head_hidden=4,
            dropout=0.0,
        )
        model.train()
        torch.manual_seed(0)
        x = torch.randn(4, 5) * 0.01
        x.requires_grad_(True)
        out = model(x)
        loss = out["total"].sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert (x.grad != 0).any()

    def test_total_equals_sum_train_mode(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.0,
        )
        model.train()
        x = torch.randn(4, 10)
        out = model(x)
        expected = sum(out[t] for t in K_TARGETS)
        torch.testing.assert_close(out["total"], expected)

    def test_k_config_backbone(self):
        """K config uses [64, 32] backbone with head_hidden=16."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[64, 32],
            head_hidden=16,
            dropout=0.25,
        )
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        assert out["total"].shape == (4,)
        for key in K_TARGETS:
            assert (out[key] >= 0).all()

    def test_dropout_effect(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
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

        assert not torch.allclose(out_train["total"].detach(), out_eval["total"])

    def test_outputs_non_negative_eval(self, model):
        """Kicker scoring is always non-negative (FG and PAT points)."""
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in K_TARGETS:
            assert (out[key] >= 0).all(), f"Negative value in {key} (eval)"

    def test_outputs_non_negative_train(self, model):
        model.train()
        x = torch.randn(4, 10)
        out = model(x)
        for key in K_TARGETS:
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
        model = MultiHeadNet(
            input_dim=10,
            target_names=K_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            head_hidden_overrides={"fg_yard_points": 24},
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out["total"].shape == (4,)
        fg_head = model.heads["fg_yard_points"]
        assert fg_head[0].out_features == 24
