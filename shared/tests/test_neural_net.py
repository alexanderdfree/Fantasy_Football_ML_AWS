"""Tests for shared.neural_net — GatedTDHead, AttentionPool, MultiHeadNetWithHistory."""

import numpy as np
import pytest
import torch

from shared.neural_net import AttentionPool, GatedTDHead, MultiHeadNetWithHistory

TARGETS = ["rushing_floor", "receiving_floor", "td_points"]


# ---------------------------------------------------------------------------
# GatedTDHead
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatedTDHead:
    def test_output_shapes(self):
        head = GatedTDHead(in_dim=16)
        x = torch.randn(4, 16)
        td_pred, gate_logit = head(x)
        assert td_pred.shape == (4,)
        assert gate_logit.shape == (4,)

    def test_td_pred_non_negative(self):
        head = GatedTDHead(in_dim=16)
        x = torch.randn(8, 16)
        td_pred, _ = head(x)
        assert (td_pred >= 0).all()

    def test_gate_logit_finite(self):
        head = GatedTDHead(in_dim=16)
        x = torch.randn(4, 16)
        _, gate_logit = head(x)
        assert torch.isfinite(gate_logit).all()

    def test_gradient_flow(self):
        head = GatedTDHead(in_dim=16)
        x = torch.randn(4, 16, requires_grad=True)
        td_pred, _ = head(x)
        td_pred.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_hidden_size_config(self):
        head = GatedTDHead(in_dim=16, gate_hidden=4, value_hidden=8)
        assert head.gate[0].out_features == 4
        assert head.value[0].out_features == 8

    def test_single_sample(self):
        head = GatedTDHead(in_dim=16)
        head.eval()
        with torch.no_grad():
            td_pred, gate_logit = head(torch.randn(1, 16))
        assert td_pred.shape == (1,)
        assert gate_logit.shape == (1,)


# ---------------------------------------------------------------------------
# AttentionPool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAttentionPool:
    def test_output_shape(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        keys = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = pool(keys, mask)
        assert out.shape == (4, 2 * 16)

    def test_mask_ignores_padding(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        pool.eval()
        keys = torch.randn(2, 8, 16)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, :3] = True  # 3 real games
        mask[1, :8] = True  # 8 real games
        with torch.no_grad():
            out = pool(keys, mask)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

    def test_all_padding_produces_zeros(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        pool.eval()
        keys = torch.randn(2, 5, 16)
        mask = torch.zeros(2, 5, dtype=torch.bool)  # all padding
        with torch.no_grad():
            out = pool(keys, mask)
        assert (out == 0).all()

    def test_gradient_flow(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        keys = torch.randn(4, 8, 16, requires_grad=True)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = pool(keys, mask)
        out.sum().backward()
        assert keys.grad is not None

    def test_project_kv(self):
        torch.manual_seed(42)
        pool_no = AttentionPool(d_model=16, n_heads=2, project_kv=False)
        pool_yes = AttentionPool(d_model=16, n_heads=2, project_kv=True)
        # Both should produce valid output of same shape
        keys = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out_no = pool_no(keys, mask)
        out_yes = pool_yes(keys, mask)
        assert out_no.shape == out_yes.shape

    def test_n_heads_changes_output_dim(self):
        pool1 = AttentionPool(d_model=16, n_heads=1)
        pool4 = AttentionPool(d_model=16, n_heads=4)
        keys = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out1 = pool1(keys, mask)
        out4 = pool4(keys, mask)
        assert out1.shape == (2, 16)
        assert out4.shape == (2, 64)

    def test_single_sequence_length(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        pool.eval()
        keys = torch.randn(2, 1, 16)
        mask = torch.ones(2, 1, dtype=torch.bool)
        with torch.no_grad():
            out = pool(keys, mask)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

    def test_no_mask(self):
        pool = AttentionPool(d_model=16, n_heads=2)
        pool.eval()
        keys = torch.randn(4, 8, 16)
        with torch.no_grad():
            out = pool(keys)  # no mask
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MultiHeadNetWithHistory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiHeadNetWithHistory:
    @pytest.fixture
    def model(self):
        return MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
        )

    @pytest.fixture
    def inputs(self):
        batch = 4
        seq_len = 6
        x_static = torch.randn(batch, 5)
        x_history = torch.randn(batch, seq_len, 3)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        return x_static, x_history, mask

    def test_output_keys(self, model, inputs):
        out = model(*inputs)
        assert set(out.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}

    def test_output_shapes(self, model, inputs):
        out = model(*inputs)
        for key in out:
            assert out[key].shape == (4,)

    def test_total_equals_sum(self, model, inputs):
        model.eval()
        with torch.no_grad():
            out = model(*inputs)
        expected = out["rushing_floor"] + out["receiving_floor"] + out["td_points"]
        torch.testing.assert_close(out["total"], expected)

    def test_non_negative_outputs(self, model, inputs):
        model.eval()
        with torch.no_grad():
            out = model(*inputs)
        for key in TARGETS:
            assert (out[key] >= 0).all()

    def test_gated_td_head(self):
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
            gated_td=True,
        )
        x_static = torch.randn(4, 5)
        x_history = torch.randn(4, 6, 3)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        assert "td_points_gate_logit" in out
        assert (out["td_points"] >= 0).all()

    def test_positional_encoding(self, inputs):
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.0,
            use_positional_encoding=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(*inputs)
        assert out["total"].shape == (4,)
        assert torch.isfinite(out["total"]).all()

    def test_gated_fusion(self, inputs):
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.0,
            use_gated_fusion=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(*inputs)
        assert out["total"].shape == (4,)

    def test_predict_numpy(self, model):
        X_s = np.random.randn(4, 5).astype(np.float32)
        X_h = np.random.randn(4, 6, 3).astype(np.float32)
        mask = np.ones((4, 6), dtype=bool)
        device = torch.device("cpu")
        preds = model.predict_numpy(X_s, X_h, mask, device)
        assert set(preds.keys()) == {"rushing_floor", "receiving_floor", "td_points", "total"}
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (4,)

    def test_gradient_flow_both_branches(self, model):
        x_static = torch.randn(4, 5, requires_grad=True)
        x_history = torch.randn(4, 6, 3, requires_grad=True)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        out["total"].sum().backward()
        assert x_static.grad is not None
        assert x_history.grad is not None

    def test_all_padding_mask(self, model):
        """All-padding should produce valid outputs (AttentionPool nan_to_num handles this)."""
        model.eval()
        x_static = torch.randn(2, 5)
        x_history = torch.randn(2, 6, 3)
        mask = torch.zeros(2, 6, dtype=torch.bool)  # all padding
        with torch.no_grad():
            out = model(x_static, x_history, mask)
        for key in out:
            assert torch.isfinite(out[key]).all(), f"NaN/Inf in {key}"
