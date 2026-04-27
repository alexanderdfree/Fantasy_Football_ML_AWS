"""Tests for shared.neural_net.MultiHeadNet (using a DST target subset).

Uses 3 of the 10 DST raw-stat targets (def_sacks, def_tds, points_allowed)
as representative test data — exercises multi-head NN shapes, dropout,
non-negative clamping, and head-hidden overrides. All 10 real DST targets
are non-negative, so the "selectively clamped" test uses a subset to cover
the non_negative_targets kwarg path.
"""

import numpy as np
import pytest
import torch

from src.shared.neural_net import MultiHeadNet

DST_TARGETS = ["def_sacks", "def_tds", "points_allowed"]


@pytest.mark.unit
class TestMultiHeadNet:
    @pytest.fixture
    def model(self):
        return MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.1,
        )

    def test_output_keys(self, model):
        x = torch.randn(4, 10)
        out = model(x)
        assert set(out.keys()) == {"def_sacks", "def_tds", "points_allowed"}

    def test_output_shapes(self, model):
        batch_size = 8
        x = torch.randn(batch_size, 10)
        out = model(x)
        for key in out:
            assert out[key].shape == (batch_size,)

    def test_custom_backbone(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=DST_TARGETS,
            backbone_layers=[64, 32, 16],
        )
        x = torch.randn(2, 5)
        out = model(x)
        for t in DST_TARGETS:
            assert out[t].shape == (2,)

    def test_single_sample_eval_mode(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[16, 8],
        )
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10)
            out = model(x)
        for t in DST_TARGETS:
            assert out[t].shape == (1,)

    def test_predict_numpy(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[16, 8],
        )
        X = np.random.randn(5, 10).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)

        assert set(preds.keys()) == {"def_sacks", "def_tds", "points_allowed"}
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (5,)

    def test_predict_numpy_single_sample(self):
        model = MultiHeadNet(
            input_dim=5,
            target_names=DST_TARGETS,
            backbone_layers=[8, 4],
        )
        X = np.random.randn(1, 5).astype(np.float32)
        device = torch.device("cpu")
        preds = model.predict_numpy(X, device)
        for t in DST_TARGETS:
            assert preds[t].shape == (1,)

    def test_gradients_flow(self, model):
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = sum(out[t].sum() for t in DST_TARGETS)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 10)

    def test_gradient_near_zero(self):
        # Seed BEFORE constructing the model so layer init (which consumes
        # torch's RNG state) is deterministic across test-suite orderings.
        torch.manual_seed(0)
        model = MultiHeadNet(
            input_dim=5,
            target_names=DST_TARGETS,
            backbone_layers=[16],
            head_hidden=4,
            dropout=0.0,
        )
        model.train()
        x = torch.randn(4, 5) * 0.01
        x.requires_grad_(True)
        out = model(x)
        loss = sum(out[t].sum() for t in DST_TARGETS)
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert (x.grad != 0).any()

    def test_dst_config_backbone(self):
        """DST config uses [128, 64] backbone with per-target head overrides."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[128, 64],
            head_hidden=32,
            dropout=0.30,
            head_hidden_overrides={"def_tds": 16, "points_allowed": 48},
        )
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for t in DST_TARGETS:
            assert out[t].shape == (4,)
        assert model.heads["def_tds"][0].out_features == 16
        assert model.heads["points_allowed"][0].out_features == 48

    def test_dropout_effect(self):
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
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

        train_sum = sum(out_train[t].detach() for t in DST_TARGETS)
        eval_sum = sum(out_eval[t] for t in DST_TARGETS)
        assert not torch.allclose(train_sum, eval_sum)

    def test_selective_non_negative_targets(self):
        """Only selected targets should be clamped to >= 0."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.0,
            non_negative_targets={"def_sacks", "def_tds"},  # not points_allowed
        )
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(16, 10)
        with torch.no_grad():
            out = model(x)
        assert (out["def_sacks"] >= 0).all()
        assert (out["def_tds"] >= 0).all()
        # points_allowed is excluded from the clamp set — may be negative

    def test_default_clamp_all(self):
        """Default (no non_negative_targets arg) — all targets clamped to >=0."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.1,
        )
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out = model(x)
        for key in DST_TARGETS:
            assert (out[key] >= 0).all(), f"Negative value in {key} (default clamp)"

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
        """DST uses per-target head overrides — sparse heads smaller, PA wider."""
        model = MultiHeadNet(
            input_dim=10,
            target_names=DST_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=16,
            head_hidden_overrides={"def_tds": 8, "points_allowed": 48},
        )
        x = torch.randn(4, 10)
        out = model(x)
        for t in DST_TARGETS:
            assert out[t].shape == (4,)
        assert model.heads["def_tds"][0].out_features == 8
        assert model.heads["points_allowed"][0].out_features == 48
