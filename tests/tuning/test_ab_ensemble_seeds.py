"""CPU tests for the stacked seed-ensemble harness (torch.func vmap).

vmap runs on CPU, so the load-bearing math — per-member clipping, the
stacked-vs-sequential training parity, and the randomness gating — is fully
testable without a GPU. The GPU-side speedup is a ship gate, not a unit test.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch
from torch import nn

from src.shared.neural_net import build_multihead_net_with_history
from src.shared.training import MultiTargetLoss, _GPUResidentBatcher
from src.tuning.ab_ensemble_seeds import (
    apply_ensemble_env,
    capture_attention_construction,
    clip_per_member_,
    predict_stacked,
    train_sequential,
    train_stacked,
)

pytestmark = pytest.mark.unit

_TARGETS = ["a", "b"]


def _tiny_cfg(**overrides) -> dict:
    cfg = {
        "nn_backbone_layers": [8],
        "attn_d_model": 8,
        "attn_n_heads": 1,
        "nn_head_hidden": 4,
        "nn_dropout": 0.0,
        "attn_dropout": 0.0,
        "attn_encoder_hidden_dim": 0,
        "nn_epochs": 2,
        # Scheduler consumed via _build_scheduler(..., scheduler_prefix="attn_")
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 2,
        "cosine_t_mult": 1,
        "cosine_eta_min": 1e-5,
        "loss_weights": {t: 1.0 for t in _TARGETS},
        "huber_deltas": {t: 1.0 for t in _TARGETS},
    }
    cfg.update(overrides)
    return cfg


def _build_models(cfg, n_members, static_dim=5, game_dim=3):
    models = []
    for i in range(n_members):
        torch.manual_seed(100 + i)
        models.append(
            build_multihead_net_with_history(
                cfg, static_dim=static_dim, game_dim=game_dim, targets=_TARGETS
            )
        )
    return models


def _criterion(cfg, **kwargs) -> MultiTargetLoss:
    return MultiTargetLoss(
        target_names=_TARGETS,
        loss_weights=cfg["loss_weights"],
        huber_deltas=cfg["huber_deltas"],
        **kwargs,
    )


def _synthetic(n=24, static_dim=5, game_dim=3, seq=4, batch_size=8):
    torch.manual_seed(7)
    feats = (
        torch.randn(n, static_dim),
        torch.randn(n, seq, game_dim),
        torch.ones(n, seq, dtype=torch.bool),
    )
    y = {t: torch.randn(n).abs() for t in _TARGETS}
    loader = _GPUResidentBatcher(
        feature_tensors=feats, y_dict=y, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return feats, y, loader


def _captures_for(models, criterion, loader, lr=1e-2):
    out = []
    for m in models:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        trainer = types.SimpleNamespace(model=m, criterion=criterion, optimizer=opt)
        out.append(
            {"trainer": trainer, "train_loader": loader, "val_loader": loader, "n_epochs": 2}
        )
    return out


def _test_capture(feats):
    return {
        "args": (
            feats[0].numpy(),
            feats[1].numpy(),
            feats[2].numpy(),
            None,
            None,
        )
    }


# ---------------------------------------------------------------------------
# Per-member clip
# ---------------------------------------------------------------------------


def test_clip_per_member_matches_clip_grad_norm():
    """Stacked per-member clip must equal clip_grad_norm_(1.0) run
    independently on each member's unstacked grads — including a member
    UNDER the norm threshold (no-op) and scalar-shaped params."""
    torch.manual_seed(0)
    shapes = [(4, 3), (7,), ()]  # matrix, vector, scalar param
    n_members = 3
    member_grads = []
    for i in range(n_members):
        scale = [0.1, 5.0, 50.0][i]  # member 0 under the clip threshold
        member_grads.append([torch.randn(s) * scale for s in shapes])

    # Reference: torch's own clip per member via dummy params.
    expected = []
    for grads in member_grads:
        params = [nn.Parameter(torch.zeros_like(g)) for g in grads]
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        expected.append([p.grad.clone() for p in params])

    stacked = [
        torch.stack([member_grads[m][j] for m in range(n_members)]) for j in range(len(shapes))
    ]
    clip_per_member_(stacked)
    for j in range(len(shapes)):
        for m in range(n_members):
            torch.testing.assert_close(stacked[j][m], expected[m][j], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Stacked-vs-sequential end-to-end parity (CPU, dropout zero)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg_overrides,criterion_kwargs",
    [
        ({}, {}),  # plain heads
        (
            {"attn_gated": True, "gated_targets": ["a"]},
            {"head_losses": {"a": "hurdle_poisson", "b": "huber"}, "gated_targets": ["a"]},
        ),
        ({"attn_use_alibi_bias": True}, {}),  # exercises the buffer path
    ],
)
def test_stacked_matches_sequential(cfg_overrides, criterion_kwargs):
    """Protocol tier (a): step-0 stacked forward ≡ per-model forward (tight).
    Tier (b): after ONE identical optimizer step, per-member params ≡
    sequential params (tight — proves the loop math). Tier (c): after 2
    trained epochs, predictions agree loosely — fp32 vmap kernel-order
    (batched matmul vs mm reduction order) compounds over steps, the same
    sub-ULP-amplification physics documented for CUDA graphs (ADR-0017)."""
    cfg = _tiny_cfg(**cfg_overrides)
    feats, y, loader = _synthetic()
    models = _build_models(cfg, n_members=2)
    criterion = _criterion(cfg, **criterion_kwargs)
    captures = _captures_for(models, criterion, loader)
    device = torch.device("cpu")

    # Tier (a): untrained forward parity, tight.
    import torch.func as tf

    template0 = copy.deepcopy(models[0]).to("meta")
    template0.eval()
    p0, b0 = tf.stack_module_state([copy.deepcopy(m) for m in models])
    with torch.no_grad():
        stacked0 = torch.vmap(
            lambda p, b: tf.functional_call(template0, (p, b), feats), in_dims=(0, 0)
        )(p0, b0)
        for i, m in enumerate(models):
            m.eval()
            eager = m(*feats)
            m.train()
            for key in eager:
                torch.testing.assert_close(stacked0[key][i], eager[key], rtol=1e-6, atol=1e-7)

    # Tier (c): trained-end predictions, loose (kernel-order compounding).
    params, buffers, template = train_stacked(captures, cfg, device, n_epochs=2)
    stacked_preds = predict_stacked(template, params, buffers, _test_capture(feats), device)
    seq_models = train_sequential(captures, cfg, device, n_epochs=2)
    template_keys = set(stacked_preds[0])
    for k_member, seq_model in enumerate(seq_models):
        seq = seq_model.predict_numpy(feats[0].numpy(), feats[1].numpy(), feats[2].numpy(), device)
        assert set(seq) == template_keys
        for key in seq:
            np.testing.assert_allclose(
                stacked_preds[k_member][key], seq[key], rtol=1e-2, atol=5e-4, err_msg=key
            )
    # The two members trained to DIFFERENT weights (different inits).
    any_key = next(iter(template_keys))
    assert not np.allclose(stacked_preds[0][any_key], stacked_preds[1][any_key])


def test_single_batch_grad_parity():
    """Protocol tier (b): POST-CLIP GRADIENTS on one batch — stacked member
    grads must match per-model eager grads tightly. Gradients are the direct
    kernel-order-affected quantity; params after even ONE Adam step are
    ill-conditioned for this check (near-zero grads make Adam's g/sqrt(v)
    sign-like, so ulp-level grad diffs become full-lr param diffs — that's
    inherent Adam behavior, not a loop bug, and tier (c) bounds it)."""
    import torch.func as tf

    cfg = _tiny_cfg()
    feats, y, loader = _synthetic()
    models = _build_models(cfg, n_members=2)
    criterion = _criterion(cfg)

    template = copy.deepcopy(models[0]).to("meta")
    template.train()
    params, buffers = tf.stack_module_state([copy.deepcopy(m) for m in models])

    torch.manual_seed(0)
    batch = next(iter(loader))
    f, yb = tuple(batch[:-1]), batch[-1]

    def member_loss(p, b, feats_, y_):
        preds = tf.functional_call(template, (p, b), feats_)
        return criterion.compute_combined_capturable(preds, y_)

    losses = torch.vmap(member_loss, in_dims=(0, 0, None, None), randomness="different")(
        params, buffers, f, yb
    )
    losses.sum().backward()
    stacked_grads = {n: p.grad for n, p in params.items() if p.grad is not None}
    clip_per_member_(list(stacked_grads.values()))

    for i, m in enumerate(models):
        m.train()
        m.zero_grad(set_to_none=True)
        loss = criterion.compute_combined_capturable(m(*f), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            torch.testing.assert_close(
                stacked_grads[name][i], p.grad, rtol=1e-5, atol=1e-7, msg=name
            )


def test_onecycle_per_batch_scheduler_parity():
    cfg = _tiny_cfg(scheduler_type="onecycle", onecycle_max_lr=5e-3, onecycle_pct_start=0.3)
    cfg.pop("cosine_t0", None)
    feats, y, loader = _synthetic()
    models = _build_models(cfg, n_members=2)
    captures = _captures_for(models, _criterion(cfg), loader)
    device = torch.device("cpu")
    params, buffers, template = train_stacked(captures, cfg, device, n_epochs=2)
    stacked_preds = predict_stacked(template, params, buffers, _test_capture(feats), device)
    seq_models = train_sequential(captures, cfg, device, n_epochs=2)
    seq0 = seq_models[0].predict_numpy(feats[0].numpy(), feats[1].numpy(), feats[2].numpy(), device)
    for key in seq0:
        np.testing.assert_allclose(stacked_preds[0][key], seq0[key], rtol=1e-2, atol=5e-4)


# ---------------------------------------------------------------------------
# Randomness gating
# ---------------------------------------------------------------------------


def test_dropout_draws_differ_across_members():
    """randomness='different': two members with IDENTICAL weights but live
    dropout must see different masks (different losses on the same batch)."""
    cfg = _tiny_cfg(nn_dropout=0.5)
    feats, y, loader = _synthetic()
    torch.manual_seed(123)
    model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=_TARGETS)
    models = [model, copy.deepcopy(model)]  # identical weights
    captures = _captures_for(models, _criterion(cfg), loader)
    device = torch.device("cpu")

    import torch.func as tf

    template = copy.deepcopy(models[0]).to("meta")
    template.train()
    params, buffers = tf.stack_module_state([m.to(device) for m in models])
    criterion = captures[0]["trainer"].criterion

    def member_loss(p, b, f, yb):
        preds = tf.functional_call(template, (p, b), f)
        return criterion.compute_combined_capturable(preds, yb)

    batch = next(iter(loader))
    f, yb = tuple(batch[:-1]), batch[-1]
    losses = torch.vmap(member_loss, in_dims=(0, 0, None, None), randomness="different")(
        params, buffers, f, yb
    )
    assert losses.shape == (2,)
    assert losses[0].item() != pytest.approx(losses[1].item())


# ---------------------------------------------------------------------------
# Regime env + capture plumbing
# ---------------------------------------------------------------------------


def test_apply_ensemble_env_sets_regime_and_rejects_compile(monkeypatch):
    for k in ("FF_NN_NORM", "FF_AMP_DTYPE", "FF_CUDA_GRAPH", "FF_NN_FIXED_EPOCHS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("FF_COMPILE", raising=False)
    apply_ensemble_env(30)
    import os

    assert os.environ["FF_NN_NORM"] == "layer"
    assert os.environ["FF_AMP_DTYPE"] == "fp32"
    assert os.environ["FF_CUDA_GRAPH"] == "0"
    assert os.environ["FF_NN_FIXED_EPOCHS"] == "30"
    monkeypatch.setenv("FF_COMPILE", "1")
    with pytest.raises(SystemExit):
        apply_ensemble_env(30)


def test_capture_patch_stashes_and_restores():
    from src.shared import neural_net
    from src.shared.training import MultiHeadTrainer

    orig_train = MultiHeadTrainer.train
    orig_predict = neural_net.MultiHeadNetWithHistory.predict_numpy
    captures: list = []
    test_capture: dict = {}
    cfg = _tiny_cfg()
    feats, y, loader = _synthetic()
    torch.manual_seed(5)
    model = build_multihead_net_with_history(cfg, static_dim=5, game_dim=3, targets=_TARGETS)

    with capture_attention_construction(captures, test_capture):
        # Simulate the pipeline's calls under the patch.
        out = MultiHeadTrainer.train(types.SimpleNamespace(), loader, loader, n_epochs=3)
        assert out == {}
        preds = model.predict_numpy(
            feats[0].numpy(), feats[1].numpy(), feats[2].numpy(), torch.device("cpu")
        )
    assert len(captures) == 1 and captures[0]["n_epochs"] == 3
    assert "args" in test_capture
    # Zero-filled with the REAL key set, full length.
    assert all(np.all(v == 0) and v.shape[0] == feats[0].shape[0] for v in preds.values())
    # Patches restored.
    assert MultiHeadTrainer.train is orig_train
    assert neural_net.MultiHeadNetWithHistory.predict_numpy is orig_predict
