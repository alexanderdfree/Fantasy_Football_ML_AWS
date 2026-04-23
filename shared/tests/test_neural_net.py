"""Tests for shared.neural_net — GatedHead, AttentionPool, and the
MultiHeadNetWithHistory / MultiHeadNetWithNestedHistory variants."""

import numpy as np
import pytest
import torch

from shared.neural_net import (
    AttentionPool,
    GatedHead,
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
    SwiGLU,
    _apply_history_dropout,
    _build_game_encoder,
    apply_non_negative,
    build_multihead_net,
    build_multihead_net_with_history,
    build_multihead_net_with_nested_history,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]


# ---------------------------------------------------------------------------
# GatedHead
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGatedHead:
    def test_output_shapes(self):
        head = GatedHead(in_dim=16)
        x = torch.randn(4, 16)
        pred, gate_logit, mu, log_alpha = head(x)
        assert pred.shape == (4,)
        assert gate_logit.shape == (4,)
        assert mu.shape == (4,)
        assert log_alpha.shape == (4,)

    def test_pred_non_negative(self):
        head = GatedHead(in_dim=16)
        x = torch.randn(8, 16)
        pred, _, mu, _ = head(x)
        assert (pred >= 0).all()
        assert (mu > 0).all()  # strictly positive thanks to softplus + 1e-6 floor

    def test_gate_logit_and_log_alpha_finite(self):
        head = GatedHead(in_dim=16)
        x = torch.randn(4, 16)
        _, gate_logit, _, log_alpha = head(x)
        assert torch.isfinite(gate_logit).all()
        assert torch.isfinite(log_alpha).all()

    def test_gradient_flow(self):
        head = GatedHead(in_dim=16)
        x = torch.randn(4, 16, requires_grad=True)
        pred, _, _, _ = head(x)
        pred.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_log_alpha_gradient_flows(self):
        """log_alpha has its own linear layer; need grad path from its output to input."""
        head = GatedHead(in_dim=16)
        x = torch.randn(4, 16, requires_grad=True)
        _, _, _, log_alpha = head(x)
        log_alpha.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_hidden_size_config(self):
        head = GatedHead(in_dim=16, gate_hidden=4, value_hidden=8)
        assert head.gate[0].out_features == 4
        assert head.value_trunk[0].out_features == 8
        assert head.value_mu[0].in_features == 8
        assert head.value_log_alpha.in_features == 8

    def test_single_sample(self):
        head = GatedHead(in_dim=16)
        head.eval()
        with torch.no_grad():
            pred, gate_logit, mu, log_alpha = head(torch.randn(1, 16))
        assert pred.shape == (1,)
        assert gate_logit.shape == (1,)
        assert mu.shape == (1,)
        assert log_alpha.shape == (1,)


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

    def test_learn_temperature_default_off(self):
        pool = AttentionPool(d_model=16, n_heads=2, n_targets=3)
        assert pool.learn_temperature is False
        assert not hasattr(pool, "log_temperature")

    def test_learn_temperature_param_shape_and_init(self):
        pool = AttentionPool(d_model=16, n_heads=2, n_targets=3, learn_temperature=True)
        assert pool.learn_temperature is True
        # Per-target scalar (one temperature replicated across heads).
        assert pool.log_temperature.shape == (3,)
        # Init at 0 → T=exp(0)=1, equivalent to no temperature at step zero.
        assert torch.allclose(pool.log_temperature, torch.zeros(3))

    def test_learn_temperature_matches_baseline_at_init(self):
        """At init log_T=0 so outputs must match a non-temperature pool exactly."""
        torch.manual_seed(0)
        base = AttentionPool(d_model=16, n_heads=2, n_targets=3)
        torch.manual_seed(0)
        with_temp = AttentionPool(d_model=16, n_heads=2, n_targets=3, learn_temperature=True)
        keys = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        base.eval()
        with_temp.eval()
        with torch.no_grad():
            out_base = base(keys, mask)
            out_temp = with_temp(keys, mask)
        assert torch.allclose(out_base, out_temp, atol=1e-6)

    def test_learn_temperature_sharpens_distribution(self):
        """Negative log_temperature ⇒ T<1 ⇒ sharper (lower-entropy) attention."""
        torch.manual_seed(0)
        pool = AttentionPool(d_model=8, n_heads=1, n_targets=1, learn_temperature=True)
        with torch.no_grad():
            pool.log_temperature.fill_(-2.0)  # T = exp(-2) ≈ 0.135 → very sharp
        keys = torch.randn(2, 10, 8)

        # Reproduce the internal attn scores to measure entropy directly.
        q = pool.queries.reshape(-1, 8)
        attn = torch.einsum("qd,bsd->bqs", q, keys) * pool.scale
        baseline_weights = torch.softmax(attn, dim=-1)
        sharpened_weights = torch.softmax(
            attn * torch.exp(-pool.log_temperature).view(1, -1, 1), dim=-1
        )
        baseline_entropy = -(baseline_weights * baseline_weights.clamp_min(1e-12).log()).sum(-1)
        sharpened_entropy = -(sharpened_weights * sharpened_weights.clamp_min(1e-12).log()).sum(-1)
        assert (sharpened_entropy < baseline_entropy).all()

    def test_learn_temperature_gradient_flows(self):
        pool = AttentionPool(d_model=16, n_heads=2, n_targets=3, learn_temperature=True)
        keys = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = pool(keys, mask)
        out.sum().backward()
        assert pool.log_temperature.grad is not None
        assert pool.log_temperature.grad.shape == (3,)

    def test_compute_entropy_default_off(self):
        pool = AttentionPool(d_model=8, n_heads=2, n_targets=3)
        keys = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        pool(keys, mask)
        # Disabled → no cached entropy (zero-cost on the hot path).
        assert pool.compute_entropy is False
        assert not hasattr(pool, "last_attn_entropy")

    def test_compute_entropy_caches_scalar(self):
        pool = AttentionPool(d_model=8, n_heads=2, n_targets=3, compute_entropy=True)
        keys = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        pool(keys, mask)
        H = pool.last_attn_entropy
        assert H.dim() == 0  # scalar
        # Entropy of a discrete distribution over 5 positions ∈ [0, log(5)].
        assert 0.0 <= H.item() <= float(np.log(5)) + 1e-6

    def test_compute_entropy_near_max_for_uniform_queries(self):
        """If queries project to zero, attention becomes uniform and
        entropy ≈ log(seq_len). Exercises the upper bound."""
        pool = AttentionPool(d_model=8, n_heads=1, n_targets=1, compute_entropy=True)
        with torch.no_grad():
            pool.queries.zero_()  # zero queries -> constant scores -> uniform softmax
        keys = torch.randn(3, 7, 8)
        mask = torch.ones(3, 7, dtype=torch.bool)
        pool(keys, mask)
        torch.testing.assert_close(
            pool.last_attn_entropy, torch.tensor(float(np.log(7))), atol=1e-4, rtol=0
        )

    def test_compute_entropy_gradient_flows(self):
        """Entropy must be differentiable w.r.t. keys so the regulariser trains."""
        pool = AttentionPool(d_model=8, n_heads=2, n_targets=1, compute_entropy=True)
        keys = torch.randn(2, 5, 8, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.bool)
        pool(keys, mask)
        pool.last_attn_entropy.backward()
        assert keys.grad is not None and (keys.grad != 0).any()


# ---------------------------------------------------------------------------
# SwiGLU + _build_game_encoder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSwiGLU:
    def test_output_shape_and_finite(self):
        block = SwiGLU(4, 8)
        x = torch.randn(3, 4)
        out = block(x)
        assert out.shape == (3, 8)
        assert torch.isfinite(out).all()

    def test_is_nonlinear(self):
        """Sanity: SwiGLU shouldn't degenerate to a pure linear map."""
        block = SwiGLU(4, 8)
        x = torch.randn(2, 4)
        y_2x = block(2.0 * x)
        y_x = block(x)
        # For a linear map we'd have y(2x) == 2*y(x); SwiGLU's gate breaks that.
        assert not torch.allclose(y_2x, 2.0 * y_x, atol=1e-4)

    def test_gradient_flows_through_both_projections(self):
        block = SwiGLU(4, 8)
        x = torch.randn(3, 4, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and (x.grad != 0).any()
        # Both projections must receive a gradient.
        assert block.gate_proj.weight.grad is not None
        assert block.value_proj.weight.grad is not None
        assert (block.gate_proj.weight.grad != 0).any()
        assert (block.value_proj.weight.grad != 0).any()


@pytest.mark.unit
class TestBuildGameEncoder:
    def test_default_is_linear_relu(self):
        enc = _build_game_encoder(in_dim=4, d_model=8, encoder_hidden_dim=0, use_swiglu=False)
        # Baseline 1-layer: Linear -> ReLU. Must not silently change structure.
        assert len(enc) == 2
        assert isinstance(enc[0], torch.nn.Linear)
        assert isinstance(enc[1], torch.nn.ReLU)
        assert enc[0].in_features == 4 and enc[0].out_features == 8

    def test_default_two_layer_shape(self):
        enc = _build_game_encoder(in_dim=4, d_model=8, encoder_hidden_dim=16, use_swiglu=False)
        # Baseline 2-layer: Linear -> ReLU -> LN -> Linear -> ReLU.
        assert len(enc) == 5
        assert isinstance(enc[0], torch.nn.Linear) and isinstance(enc[1], torch.nn.ReLU)
        assert isinstance(enc[2], torch.nn.LayerNorm)
        assert isinstance(enc[3], torch.nn.Linear) and isinstance(enc[4], torch.nn.ReLU)

    def test_swiglu_one_layer(self):
        enc = _build_game_encoder(in_dim=4, d_model=8, encoder_hidden_dim=0, use_swiglu=True)
        assert len(enc) == 1
        assert isinstance(enc[0], SwiGLU)
        x = torch.randn(3, 4)
        out = enc(x)
        assert out.shape == (3, 8)

    def test_swiglu_two_layer(self):
        enc = _build_game_encoder(in_dim=4, d_model=8, encoder_hidden_dim=16, use_swiglu=True)
        # SwiGLU -> LN -> SwiGLU.
        assert len(enc) == 3
        assert isinstance(enc[0], SwiGLU)
        assert isinstance(enc[1], torch.nn.LayerNorm)
        assert isinstance(enc[2], SwiGLU)
        x = torch.randn(3, 4)
        out = enc(x)
        assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# _apply_history_dropout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyHistoryDropout:
    # Every test that consumes the torch RNG (rate>0, training=True) is wrapped
    # in ``torch.random.fork_rng`` so the global stream is unchanged for
    # downstream tests — otherwise later model-init draws shift and a
    # pre-existing flaky gradient-flow test would land on a bad seed.
    def test_noop_when_rate_zero(self):
        mask = torch.tensor([[True, True, False], [True, True, True]])
        out = _apply_history_dropout(mask, rate=0.0, training=True)
        assert torch.equal(out, mask)

    def test_noop_when_eval(self):
        mask = torch.ones(4, 8, dtype=torch.bool)
        # rate>0 but training=False → no rand_like call, RNG untouched.
        out = _apply_history_dropout(mask, rate=0.9, training=False)
        assert torch.equal(out, mask)

    def test_padding_never_restored(self):
        mask = torch.tensor([[True, False, False], [False, True, True]])
        with torch.random.fork_rng():
            torch.manual_seed(0)
            for _ in range(20):
                out = _apply_history_dropout(mask, rate=0.5, training=True)
                # Positions that were False in input must remain False.
                assert not (out & ~mask).any()

    def test_all_dropped_row_restored(self):
        # Single row with only one real game — with rate=1.0 every real slot
        # would drop, so the fallback must restore the original mask.
        mask = torch.tensor([[True, False, False, False]])
        with torch.random.fork_rng():
            out = _apply_history_dropout(mask, rate=1.0, training=True)
        assert torch.equal(out, mask)

    def test_empty_row_stays_empty(self):
        # A row that was all-padding stays all-padding even with dropout.
        mask = torch.tensor([[False, False, False], [True, True, True]])
        with torch.random.fork_rng():
            torch.manual_seed(0)
            out = _apply_history_dropout(mask, rate=0.5, training=True)
        assert not out[0].any()
        assert out[1].any()  # row 1 must still carry signal

    def test_drops_something_at_high_rate(self):
        mask = torch.ones(64, 16, dtype=torch.bool)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            out = _apply_history_dropout(mask, rate=0.5, training=True)
        # At 50% rate over 64*16=1024 positions, losing zero is astronomically
        # unlikely. Must strictly shrink the mask.
        assert (out != mask).sum() > 0
        # Every row still keeps at least one real game.
        assert out.any(dim=-1).all()


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

    def test_gated_head(self):
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
            gated=True,
            gated_targets=["rushing_tds"],
        )
        x_static = torch.randn(4, 5)
        x_history = torch.randn(4, 6, 3)
        mask = torch.ones(4, 6, dtype=torch.bool)
        out = model(x_static, x_history, mask)
        assert "rushing_tds_gate_logit" in out
        # GatedHead emits value_mu + value_log_alpha for hurdle_negbin loss access.
        assert "rushing_tds_value_mu" in out
        assert "rushing_tds_value_log_alpha" in out
        assert (out["rushing_tds"] >= 0).all()
        assert (out["rushing_tds_value_mu"] > 0).all()  # strictly positive rate
        assert torch.isfinite(out["rushing_tds_value_log_alpha"]).all()

    def test_gated_targets_list_accepts_multiple(self):
        """Multi-gate API: multiple gated heads coexist via gated_targets list."""
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
            gated=True,
            gated_targets=["rushing_tds", "receiving_tds"],
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
        # gated defaults to False — all heads are plain Sequentials and
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


# ---------------------------------------------------------------------------
# Factory helpers
#
# The factories are the single place that maps a training cfg dict to the
# MultiHeadNet* constructor kwargs. Tests here pin the contract so a future
# kwarg that a caller forgets to thread through cannot silently regress.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildMultiHeadNet:
    def _base_cfg(self):
        return {
            "nn_backbone_layers": [8, 4],
            "nn_head_hidden": 4,
            "nn_dropout": 0.0,
        }

    def test_builds_correct_type(self):
        model = build_multihead_net(self._base_cfg(), input_dim=6, targets=TARGETS)
        assert isinstance(model, MultiHeadNet)

    def test_forwards_non_negative_targets_from_cfg(self):
        cfg = self._base_cfg() | {"nn_non_negative_targets": {"rushing_yards"}}
        model = build_multihead_net(cfg, input_dim=6, targets=TARGETS)
        assert model.non_negative_targets == {"rushing_yards"}

    def test_missing_non_negative_targets_defaults_to_all_clamped(self):
        model = build_multihead_net(self._base_cfg(), input_dim=6, targets=TARGETS)
        assert model.non_negative_targets == set(TARGETS)

    def test_forwards_head_hidden_overrides_from_cfg(self):
        cfg = self._base_cfg() | {"nn_head_hidden_overrides": {"rushing_tds": 8}}
        model = build_multihead_net(cfg, input_dim=6, targets=TARGETS)
        # Override goes through: rushing_tds head's first Linear has out_features=8
        tds_head = model.heads["rushing_tds"]
        assert tds_head[0].out_features == 8
        # Others still use the default
        assert model.heads["rushing_yards"][0].out_features == 4


@pytest.mark.unit
class TestBuildMultiHeadNetWithHistory:
    def _base_cfg(self):
        return {
            "nn_backbone_layers": [8, 4],
            "nn_head_hidden": 4,
            "nn_dropout": 0.0,
        }

    def test_builds_correct_type(self):
        model = build_multihead_net_with_history(
            self._base_cfg(), static_dim=5, game_dim=3, targets=TARGETS
        )
        assert isinstance(model, MultiHeadNetWithHistory)

    def test_forwards_attn_kwargs_from_cfg(self):
        cfg = self._base_cfg() | {
            "attn_d_model": 16,
            "attn_n_heads": 4,
            "attn_gated_fusion": True,
            "attn_positional_encoding": True,
            "attn_max_seq_len": 12,
        }
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert model.d_model == 16
        assert model.use_gated_fusion is True
        assert model.use_positional_encoding is True
        assert model.pos_embedding.num_embeddings == 12

    def test_forwards_non_negative_targets_from_cfg(self):
        cfg = self._base_cfg() | {"nn_non_negative_targets": set()}
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert model.non_negative_targets == set()

    def test_learn_attn_temperature_defaults_off(self):
        model = build_multihead_net_with_history(
            self._base_cfg(), static_dim=5, game_dim=3, targets=TARGETS
        )
        assert model.attn_pool.learn_temperature is False

    def test_learn_attn_temperature_opt_in(self):
        cfg = self._base_cfg() | {"attn_learn_temperature": True}
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert model.attn_pool.learn_temperature is True
        assert model.attn_pool.log_temperature.shape == (len(TARGETS),)

    def test_history_dropout_defaults_zero(self):
        model = build_multihead_net_with_history(
            self._base_cfg(), static_dim=5, game_dim=3, targets=TARGETS
        )
        assert model.history_dropout == 0.0

    def test_history_dropout_from_cfg(self):
        cfg = self._base_cfg() | {"attn_history_dropout": 0.25}
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert model.history_dropout == 0.25

    def test_history_dropout_is_noop_in_eval(self):
        """Eval path must ignore history_dropout entirely (determinism)."""
        cfg = self._base_cfg() | {"attn_history_dropout": 0.9}
        with torch.random.fork_rng():
            model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
            model.eval()
            x_static = torch.randn(4, 5)
            x_history = torch.randn(4, 6, 3)
            mask = torch.ones(4, 6, dtype=torch.bool)
            with torch.no_grad():
                out1 = model(x_static, x_history, mask)
                out2 = model(x_static, x_history, mask)
        for k in TARGETS:
            torch.testing.assert_close(out1[k], out2[k])

    def test_swiglu_encoder_defaults_off(self):
        model = build_multihead_net_with_history(
            self._base_cfg(), static_dim=5, game_dim=3, targets=TARGETS
        )
        assert model.use_swiglu_encoder is False
        assert not any(isinstance(m, SwiGLU) for m in model.game_encoder.modules())

    def test_swiglu_encoder_opt_in_one_layer(self):
        cfg = self._base_cfg() | {"attn_use_swiglu_encoder": True}
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert model.use_swiglu_encoder is True
        assert any(isinstance(m, SwiGLU) for m in model.game_encoder.modules())

    def test_swiglu_encoder_opt_in_two_layer(self):
        cfg = self._base_cfg() | {
            "attn_use_swiglu_encoder": True,
            "attn_encoder_hidden_dim": 16,
        }
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        swiglus = [m for m in model.game_encoder.modules() if isinstance(m, SwiGLU)]
        assert len(swiglus) == 2  # SwiGLU -> LN -> SwiGLU
        lns = [m for m in model.game_encoder.modules() if isinstance(m, torch.nn.LayerNorm)]
        assert len(lns) == 1

    def test_swiglu_encoder_forward_produces_predictions(self):
        cfg = self._base_cfg() | {"attn_use_swiglu_encoder": True}
        with torch.random.fork_rng():
            model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
            model.eval()
            with torch.no_grad():
                out = model(
                    torch.randn(4, 5),
                    torch.randn(4, 6, 3),
                    torch.ones(4, 6, dtype=torch.bool),
                )
        for t in TARGETS:
            assert out[t].shape == (4,)
            assert torch.isfinite(out[t]).all()

    def test_attn_entropy_defaults_off(self):
        model = build_multihead_net_with_history(
            self._base_cfg(), static_dim=5, game_dim=3, targets=TARGETS
        )
        assert model.attn_entropy_coeff == 0.0
        assert model.attn_pool.compute_entropy is False
        with torch.random.fork_rng():
            _ = model(
                torch.randn(2, 5),
                torch.randn(2, 4, 3),
                torch.ones(2, 4, dtype=torch.bool),
            )
        assert model.attention_entropy_loss() is None

    def test_attn_entropy_opt_in_produces_scaled_loss(self):
        """Positive coefficient yields loss = coeff * mean_entropy."""
        coeff = 0.01
        cfg = self._base_cfg() | {"attn_entropy_coeff": coeff}
        with torch.random.fork_rng():
            model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
            _ = model(
                torch.randn(2, 5),
                torch.randn(2, 4, 3),
                torch.ones(2, 4, dtype=torch.bool),
            )
        loss = model.attention_entropy_loss()
        assert loss is not None
        torch.testing.assert_close(loss, coeff * model.attn_pool.last_attn_entropy)

    def test_attn_entropy_sign_controls_direction(self):
        """Negative coeff gives a negative regulariser for non-zero entropy."""
        cfg = self._base_cfg() | {"attn_entropy_coeff": -0.1}
        with torch.random.fork_rng():
            model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
            _ = model(
                torch.randn(2, 5),
                torch.randn(2, 4, 3),
                torch.ones(2, 4, dtype=torch.bool),
            )
        loss = model.attention_entropy_loss()
        assert loss is not None
        # Non-degenerate attention over >1 real position has H > 0 → -0.1*H < 0.
        assert loss.item() < 0.0

    def test_attn_entropy_returns_none_before_forward(self):
        """coeff>0 but no forward yet → short-circuit to None (trainer-safe)."""
        cfg = self._base_cfg() | {"attn_entropy_coeff": 0.1}
        model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=TARGETS)
        assert not hasattr(model.attn_pool, "last_attn_entropy")
        assert model.attention_entropy_loss() is None


@pytest.mark.unit
class TestBuildMultiHeadNetWithNestedHistory:
    def _base_cfg(self):
        return {
            "nn_backbone_layers": [8, 4],
            "nn_head_hidden": 4,
            "nn_dropout": 0.0,
        }

    def test_builds_correct_type(self):
        model = build_multihead_net_with_nested_history(
            self._base_cfg(), static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert isinstance(model, MultiHeadNetWithNestedHistory)

    def test_forwards_nested_kwargs_from_cfg(self):
        cfg = self._base_cfg() | {
            "attn_kick_dim": 8,
            "attn_d_model": 16,
            "attn_positional_encoding": True,
        }
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=10, targets=K_TARGETS
        )
        assert model.d_kick == 8
        assert model.d_model == 16
        assert model.pos_embedding.num_embeddings == 10

    def test_forwards_non_negative_targets_from_cfg(self):
        cfg = self._base_cfg() | {"nn_non_negative_targets": {"fg_yard_points"}}
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.non_negative_targets == {"fg_yard_points"}

    def test_learn_attn_temperature_defaults_off(self):
        model = build_multihead_net_with_nested_history(
            self._base_cfg(), static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.attn_pool.learn_temperature is False

    def test_learn_attn_temperature_opt_in(self):
        cfg = self._base_cfg() | {"attn_learn_temperature": True}
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.attn_pool.learn_temperature is True
        assert model.attn_pool.log_temperature.shape == (len(K_TARGETS),)

    def test_history_dropout_defaults_zero(self):
        model = build_multihead_net_with_nested_history(
            self._base_cfg(), static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.history_dropout == 0.0

    def test_history_dropout_from_cfg(self):
        cfg = self._base_cfg() | {"attn_history_dropout": 0.15}
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.history_dropout == 0.15

    def test_swiglu_encoder_defaults_off(self):
        model = build_multihead_net_with_nested_history(
            self._base_cfg(), static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.use_swiglu_encoder is False
        # Game encoder (outer) must be vanilla; kick encoder always stays
        # Linear+ReLU regardless of the flag.
        assert not any(isinstance(m, SwiGLU) for m in model.game_encoder.modules())
        assert not any(isinstance(m, SwiGLU) for m in model.kick_encoder.modules())

    def test_swiglu_encoder_opt_in_leaves_kick_encoder_vanilla(self):
        """The SwiGLU flag targets the outer game encoder only — kick
        encoder stays Linear+ReLU so we don't silently broaden its scope."""
        cfg = self._base_cfg() | {"attn_use_swiglu_encoder": True}
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.use_swiglu_encoder is True
        assert any(isinstance(m, SwiGLU) for m in model.game_encoder.modules())
        assert not any(isinstance(m, SwiGLU) for m in model.kick_encoder.modules())

    def test_swiglu_encoder_forward_still_produces_predictions(self):
        cfg = self._base_cfg() | {
            "attn_use_swiglu_encoder": True,
            "attn_encoder_hidden_dim": 16,
        }
        with torch.random.fork_rng():
            model = build_multihead_net_with_nested_history(
                cfg, static_dim=7, kick_dim=9, max_games=5, targets=K_TARGETS
            )
            model.eval()
            with torch.no_grad():
                out = model(
                    torch.randn(4, 7),
                    torch.randn(4, 5, 3, 9),
                    torch.ones(4, 5, dtype=torch.bool),
                    torch.ones(4, 5, 3, dtype=torch.bool),
                )
        for t in K_TARGETS:
            assert out[t].shape == (4,)
            assert torch.isfinite(out[t]).all()

    def test_attn_entropy_defaults_off(self):
        model = build_multihead_net_with_nested_history(
            self._base_cfg(), static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.attn_entropy_coeff == 0.0
        assert model.attn_pool.compute_entropy is False
        # Inner kick pool must never turn on entropy computation.
        assert model.inner_pool.compute_entropy is False
        # coeff=0 short-circuit returns None even before any forward.
        assert model.attention_entropy_loss() is None

    def test_attn_entropy_returns_none_before_forward(self):
        """With coeff>0 but no forward yet, ``last_attn_entropy`` is absent
        and the loss method must short-circuit to None (trainer-safe)."""
        cfg = self._base_cfg() | {"attn_entropy_coeff": 0.1}
        model = build_multihead_net_with_nested_history(
            cfg, static_dim=7, kick_dim=9, max_games=17, targets=K_TARGETS
        )
        assert model.attn_entropy_coeff == 0.1
        assert not hasattr(model.attn_pool, "last_attn_entropy")
        assert model.attention_entropy_loss() is None

    def test_attn_entropy_opt_in_outer_only(self):
        """Entropy regularisation must target the outer pool only; the inner
        kick pool stays untouched so the regulariser doesn't silently leak to
        a different granularity."""
        cfg = self._base_cfg() | {"attn_entropy_coeff": 0.05}
        with torch.random.fork_rng():
            model = build_multihead_net_with_nested_history(
                cfg, static_dim=7, kick_dim=9, max_games=5, targets=K_TARGETS
            )
            _ = model(
                torch.randn(2, 7),
                torch.randn(2, 5, 3, 9),
                torch.ones(2, 5, dtype=torch.bool),
                torch.ones(2, 5, 3, dtype=torch.bool),
            )
        assert model.attn_pool.compute_entropy is True
        assert model.inner_pool.compute_entropy is False
        assert hasattr(model.attn_pool, "last_attn_entropy")
        assert not hasattr(model.inner_pool, "last_attn_entropy")
        loss = model.attention_entropy_loss()
        assert loss is not None
        torch.testing.assert_close(loss, 0.05 * model.attn_pool.last_attn_entropy)
