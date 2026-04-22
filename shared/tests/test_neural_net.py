"""Tests for shared.neural_net — GatedTDHead, AttentionPool, and the
MultiHeadNetWithHistory / MultiHeadNetWithNestedHistory variants."""

import numpy as np
import pytest
import torch

from shared.neural_net import (
    AttentionPool,
    GatedTDHead,
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
    apply_non_negative,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]


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
        assert set(out.keys()) == {"rushing_yards", "receiving_yards", "rushing_tds"}

    def test_output_shapes(self, model, inputs):
        out = model(*inputs)
        for key in out:
            assert out[key].shape == (4,)

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
            gated_td_targets=["rushing_tds"],
        )
        x_static = torch.randn(4, 5)
        x_history = torch.randn(4, 6, 3)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        assert "rushing_tds_gate_logit" in out
        assert (out["rushing_tds"] >= 0).all()

    def test_gated_td_targets_list_accepts_multiple(self):
        """New multi-gate API: multiple gated heads coexist via gated_td_targets list."""
        targets = ["rushing_tds", "receiving_tds", "rushing_yards"]
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=targets,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
            gated_td=True,
            gated_td_targets=["rushing_tds", "receiving_tds"],
        )
        x_static = torch.randn(4, 5)
        x_history = torch.randn(4, 6, 3)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        assert "rushing_tds_gate_logit" in out
        assert "receiving_tds_gate_logit" in out
        # Non-gated target should have no gate logit emitted.
        assert "rushing_yards_gate_logit" not in out

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
        for t in TARGETS:
            assert out[t].shape == (4,)
            assert torch.isfinite(out[t]).all()

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
        for t in TARGETS:
            assert out[t].shape == (4,)

    def test_predict_numpy(self, model):
        X_s = np.random.randn(4, 5).astype(np.float32)
        X_h = np.random.randn(4, 6, 3).astype(np.float32)
        mask = np.ones((4, 6), dtype=bool)
        device = torch.device("cpu")
        preds = model.predict_numpy(X_s, X_h, mask, device)
        assert set(preds.keys()) == {"rushing_yards", "receiving_yards", "rushing_tds"}
        for key in preds:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (4,)

    def test_gradient_flow_both_branches(self, model):
        x_static = torch.randn(4, 5, requires_grad=True)
        x_history = torch.randn(4, 6, 3, requires_grad=True)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        sum(out[t].sum() for t in TARGETS).backward()
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


# ---------------------------------------------------------------------------
# MultiHeadNetWithNestedHistory
# ---------------------------------------------------------------------------

K_TARGETS = ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"]


@pytest.mark.unit
class TestMultiHeadNetWithNestedHistory:
    """Nested attention: inner pool over kicks + outer attention over games."""

    @pytest.fixture
    def model(self):
        return MultiHeadNetWithNestedHistory(
            static_dim=7,
            kick_dim=9,
            target_names=K_TARGETS,
            backbone_layers=[16, 8],
            d_kick=8,
            d_model=16,
            n_attn_heads=2,
            head_hidden=8,
            dropout=0.0,
            use_positional_encoding=True,
            max_games=5,
        )

    @pytest.fixture
    def batch(self):
        B, G, K = 4, 5, 3
        x_static = torch.randn(B, 7)
        x_kicks = torch.randn(B, G, K, 9)
        outer_mask = torch.ones(B, G, dtype=torch.bool)
        inner_mask = torch.ones(B, G, K, dtype=torch.bool)
        return x_static, x_kicks, outer_mask, inner_mask

    def test_output_keys(self, model, batch):
        preds = model(*batch)
        assert set(preds.keys()) == set(K_TARGETS)

    def test_output_shapes(self, model, batch):
        preds = model(*batch)
        for key in K_TARGETS:
            assert preds[key].shape == (4,), f"{key}: {preds[key].shape}"

    def test_non_negative_outputs(self, model, batch):
        """Default: all K heads are non-negative (clamp min=0)."""
        model.eval()
        with torch.no_grad():
            preds = model(*batch)
        for t in K_TARGETS:
            assert (preds[t] >= 0).all(), f"{t} has negative outputs"

    def test_all_outer_padding(self, model):
        """Row with no prior games — all-False outer_mask must not produce NaN."""
        model.eval()
        B, G, K = 2, 5, 3
        x_static = torch.randn(B, 7)
        x_kicks = torch.zeros(B, G, K, 9)
        outer_mask = torch.zeros(B, G, dtype=torch.bool)
        inner_mask = torch.zeros(B, G, K, dtype=torch.bool)
        with torch.no_grad():
            preds = model(x_static, x_kicks, outer_mask, inner_mask)
        for key in preds:
            assert torch.isfinite(preds[key]).all(), f"NaN/Inf in {key}"

    def test_all_inner_padding_for_real_game(self, model):
        """A real game with 0 kicks (all-False inner row) must not produce NaN."""
        model.eval()
        B, G, K = 2, 5, 3
        x_static = torch.randn(B, 7)
        x_kicks = torch.zeros(B, G, K, 9)
        outer_mask = torch.zeros(B, G, dtype=torch.bool)
        outer_mask[0, 0] = True  # One real game
        inner_mask = torch.zeros(B, G, K, dtype=torch.bool)
        # Note: inner_mask all False for that real game — no kicks recorded
        with torch.no_grad():
            preds = model(x_static, x_kicks, outer_mask, inner_mask)
        for key in preds:
            assert torch.isfinite(preds[key]).all(), f"NaN/Inf in {key}"

    def test_gradient_flow_both_branches(self, model):
        """Gradients must flow to both static and kicks tensors."""
        B, G, K = 4, 5, 3
        x_static = torch.randn(B, 7, requires_grad=True)
        x_kicks = torch.randn(B, G, K, 9, requires_grad=True)
        outer_mask = torch.ones(B, G, dtype=torch.bool)
        inner_mask = torch.ones(B, G, K, dtype=torch.bool)
        preds = model(x_static, x_kicks, outer_mask, inner_mask)
        sum(preds[t].sum() for t in K_TARGETS).backward()
        assert x_static.grad is not None
        assert x_kicks.grad is not None
        assert (x_static.grad != 0).any()
        assert (x_kicks.grad != 0).any()

    def test_positional_encoding_creates_embedding(self):
        m = MultiHeadNetWithNestedHistory(
            static_dim=4,
            kick_dim=6,
            target_names=K_TARGETS,
            backbone_layers=[8],
            d_kick=4,
            d_model=8,
            max_games=7,
            use_positional_encoding=True,
        )
        assert hasattr(m, "pos_embedding")
        assert m.pos_embedding.num_embeddings == 7

    def test_predict_numpy(self, model):
        model.eval()
        B, G, K = 4, 5, 3
        X_static = np.random.randn(B, 7).astype(np.float32)
        X_kicks = np.random.randn(B, G, K, 9).astype(np.float32)
        outer = np.ones((B, G), dtype=bool)
        inner = np.ones((B, G, K), dtype=bool)
        preds = model.predict_numpy(X_static, X_kicks, outer, inner, torch.device("cpu"))
        for key in K_TARGETS:
            assert isinstance(preds[key], np.ndarray)
            assert preds[key].shape == (B,)


# ---------------------------------------------------------------------------
# Non-negative head floor regression (TODO.md archive:
# "Softplus floor inflated low-scoring predictions")
#
# softplus(0) ~ 0.693, so a softplus-based non-neg clamp produces a floor of
# ~0.693 per head — compounding to ~2 pts (3 heads) or ~2.8 pts (K's 4 heads).
# The non-neg clamp must be torch.clamp(min=0.0) so a head whose pre-activation
# is <= 0 emits *exactly* 0, not ~0.693.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyNonNegative:
    """Direct regression test for the shared clamp helper — a softplus
    replacement here would make all three variants wrong at once, so this
    test guards the single point of change."""

    def test_clamps_negative_to_exact_zero(self):
        val = torch.tensor([-3.0, -0.1, 0.0, 0.5, 2.0])
        out = apply_non_negative(val, "yards", {"yards"})
        assert torch.equal(out, torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))

    def test_passes_through_when_name_not_in_set(self):
        val = torch.tensor([-3.0, 0.0, 2.0])
        out = apply_non_negative(val, "bonus", {"yards"})
        assert torch.equal(out, val)

    def test_zero_stays_exact_zero_not_softplus_floor(self):
        val = torch.zeros(4)
        out = apply_non_negative(val, "t", {"t"})
        assert torch.all(out == 0.0), f"expected exact zeros, got {out.tolist()}"


def _zero_head_biases(model):
    """Force each per-target head's final Linear bias to a large negative
    value so ReLU-style clamping yields exact zeros regardless of input."""
    for head in model.heads.values():
        # Walk nested Sequentials; final Linear is the output layer.
        linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
        last = linears[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(-5.0)  # Pushed well below 0 for robust clamp test.


@pytest.mark.unit
class TestNonNegativeFloor:
    """Confirm non-negative heads emit exact zeros (clamp, not softplus)."""

    def test_multi_head_net_exact_zero(self):
        model = MultiHeadNet(
            input_dim=6,
            target_names=TARGETS,
            backbone_layers=[8, 4],
            head_hidden=4,
            dropout=0.0,
        )
        _zero_head_biases(model)
        model.eval()
        with torch.no_grad():
            preds = model(torch.randn(5, 6))
        for t in TARGETS:
            assert preds[t].shape == (5,)
            assert torch.all(preds[t] == 0.0), f"{t}: expected exact zeros, got {preds[t].tolist()}"

    def test_multi_head_net_with_history_exact_zero(self):
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[8, 4],
            d_model=8,
            n_attn_heads=1,
            head_hidden=4,
            dropout=0.0,
        )
        _zero_head_biases(model)
        model.eval()
        B, T = 3, 5
        # gated_td defaults to False — all heads are plain Sequentials and
        # subject to the non-neg clamp we're testing.
        with torch.no_grad():
            preds = model(
                torch.randn(B, 5),
                torch.randn(B, T, 3),
                torch.ones(B, T, dtype=torch.bool),
            )
        for t in TARGETS:
            assert torch.all(preds[t] == 0.0), f"{t}: expected exact zeros, got {preds[t].tolist()}"

    def test_multi_head_net_with_nested_history_exact_zero(self):
        model = MultiHeadNetWithNestedHistory(
            static_dim=7,
            kick_dim=9,
            target_names=K_TARGETS,
            backbone_layers=[16, 8],
            d_kick=8,
            d_model=16,
            n_attn_heads=2,
            head_hidden=8,
            dropout=0.0,
        )
        _zero_head_biases(model)
        model.eval()
        B, G, K = 4, 5, 3
        with torch.no_grad():
            preds = model(
                torch.randn(B, 7),
                torch.randn(B, G, K, 9),
                torch.ones(B, G, dtype=torch.bool),
                torch.ones(B, G, K, dtype=torch.bool),
            )
        for t in K_TARGETS:
            assert torch.all(preds[t] == 0.0), f"{t}: expected exact zeros, got {preds[t].tolist()}"
