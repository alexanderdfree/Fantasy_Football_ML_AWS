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
    """Protocol tier (b), real path: POST-CLIP GRADIENTS on one batch —
    stacked member grads must match per-model eager grads tightly — and then
    post-Adam-step params must match on the WELL-CONDITIONED mask. At step 1
    Adam's update is lr*g/(|g|+eps): sign-like near g=0, where ulp-level
    kernel-order grad noise flips signs into full-lr param diffs (inherent
    Adam behavior, not a loop bug). Away from zero (|g| > 1e-4) the update
    is insensitive to that noise, so the masked comparison restores the
    tight post-step check; the g≈0 region is covered exactly by
    test_adam_step_parity_with_injected_grads (noise-free grads)."""
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

    # Masked post-Adam-step param parity (the restored real-path check).
    hp = {"lr": 1e-2, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8}
    torch.optim.AdamW(list(params.values()), foreach=True, **hp).step()
    checked = 0
    for i, m in enumerate(models):
        torch.optim.AdamW(m.parameters(), foreach=True, **hp).step()
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            mask = p.grad.abs() > 1e-4
            if not mask.any():
                continue
            checked += int(mask.sum())
            torch.testing.assert_close(
                params[name][i][mask], p.detach()[mask], rtol=0.0, atol=1e-7, msg=name
            )
    assert checked > 0  # the mask must not silently empty the check


def test_adam_step_parity_with_injected_grads():
    """The well-conditioned reformulation of post-Adam-step parity: with
    IDENTICAL injected gradients (no kernel-order noise) one stacked
    foreach-AdamW over [N, *shape] params must equal N independent AdamWs
    BITWISE across multiple steps and lr changes — AdamW is elementwise, so
    stacking is purely a layout change. This pins per-member optimizer-state
    independence (exp_avg/exp_avg_sq never couple across members), hyperparam
    wiring, and bias-correction step counting; combined with the tight grad
    parity above it covers what the naive post-step param comparison (removed
    for Adam's sign-like step-1 ill-conditioning near g=0) could not."""
    import torch.func as tf

    cfg = _tiny_cfg()
    models = _build_models(cfg, n_members=3)
    seq_models = [copy.deepcopy(m) for m in models]
    seq_named = [dict(m.named_parameters()) for m in seq_models]

    params, _ = tf.stack_module_state([copy.deepcopy(m) for m in models])
    hp = {"lr": 3e-3, "weight_decay": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8}
    opt_stacked = torch.optim.AdamW(list(params.values()), foreach=True, **hp)
    opts_seq = [torch.optim.AdamW(m.parameters(), foreach=True, **hp) for m in seq_models]

    gen = torch.Generator().manual_seed(11)
    for lr in (3e-3, 1e-3, 5e-4):  # lr changes mimic a scheduler across steps
        for opt in (opt_stacked, *opts_seq):
            for group in opt.param_groups:
                group["lr"] = lr
        for name, stacked_p in params.items():
            g = torch.randn(stacked_p.shape, generator=gen)
            stacked_p.grad = g
            for i, named in enumerate(seq_named):
                named[name].grad = g[i].clone()
        opt_stacked.step()
        for opt in opts_seq:
            opt.step()

    for i, named in enumerate(seq_named):
        for name, p in named.items():
            torch.testing.assert_close(
                params[name][i], p, rtol=0.0, atol=0.0, msg=f"{name}[member {i}]"
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
    # apply_ensemble_env writes os.environ directly (it's the CLI regime
    # setter), so route every key it touches through monkeypatch.setenv FIRST:
    # that registers an undo to the pre-test state, which teardown applies
    # regardless of what the function wrote. The plain-delenv version leaked
    # FF_NN_FIXED_EPOCHS=30 into the xdist worker and made a later trainer
    # test run 30 epochs instead of 3.
    for k in (
        "FF_NN_NORM",
        "FF_AMP_DTYPE",
        "FF_CUDA_GRAPH",
        "FF_CUDA_GRAPH_FULL",
        "FF_NN_FIXED_EPOCHS",
        "FF_COMPILE",
    ):
        monkeypatch.setenv(k, "managed-for-restore")
        monkeypatch.delenv(k)
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


# ---------------------------------------------------------------------------
# Parity gate (decision-level: FP fork vs seed noise)
# ---------------------------------------------------------------------------


def test_parity_report_gates_on_fp_fork_not_raw_key_ratio():
    """The gate is decision-level: member forks must be small vs the
    seed-to-seed FP spread. A near-constant clamped key (RB fumbles_lost
    pinned at 0) with a small absolute blip must NOT fail the gate — it
    exploded the old spread-scaled scalar to 1.86e5 on the first Batch run —
    but it must surface as the top raw-key diagnostic. A genuinely forked
    member (FP-scale shift) must fail."""
    from src.tuning.ab_ensemble_seeds import _parity_report

    rng = np.random.default_rng(0)
    n = 400

    def member():
        return {
            "rushing_yards": rng.normal(60, 20, n),
            "receiving_yards": rng.normal(20, 10, n),
            "rushing_tds": rng.gamma(1.0, 0.4, n),
            "receiving_tds": rng.gamma(1.0, 0.2, n),
            "receptions": rng.gamma(2.0, 1.0, n),
            "fumbles_lost": np.zeros(n),
        }

    seq = [member(), member()]  # two "seeds": O(seed-noise) apart by construction
    stacked = []
    for p in seq:
        q = {k: v + rng.normal(0.0, 0.01, n) for k, v in p.items()}  # tiny fork
        q["fumbles_lost"] = p["fumbles_lost"] + 0.02  # constant blip, seq std == 0
        stacked.append(q)

    rep = _parity_report("RB", seq, stacked)
    assert rep["ok"], rep
    assert rep["fp_fork_over_seed_noise"] < 0.05
    # The degenerate-spread key dominates diagnostics without failing the gate.
    assert rep["raw_key_diagnostics_top"][0]["key"] == "fumbles_lost"
    assert rep["raw_key_diagnostics_top"][0]["ratio_over_spread"] > 1000

    # A genuine FP-scale fork on one member fails.
    bad = [stacked[0], {k: v + 10.0 for k, v in stacked[1].items()}]
    assert not _parity_report("RB", seq, bad)["ok"]

    with pytest.raises(SystemExit):
        _parity_report("RB", seq[:1], stacked[:1])


# ---------------------------------------------------------------------------
# Batch route (FF_TUNE_ENSEMBLE_AB env dispatch)
# ---------------------------------------------------------------------------


def test_tune_nn_env_dispatch_routes_to_batch_entry(monkeypatch):
    """FF_TUNE_ENSEMBLE_AB=1 must route tune_nn.main() into
    ab_ensemble_seeds.run_batch_entry(position) right after argparse —
    before backend/n_jobs resolution and the data bootstrap (this is the
    fixed-ENTRYPOINT Batch route; see run_batch_entry's docstring)."""
    import sys

    from src.tuning import tune_nn

    calls: list[str] = []
    monkeypatch.setattr("src.tuning.ab_ensemble_seeds.run_batch_entry", calls.append)

    def _boom(*a, **k):
        raise AssertionError("must dispatch before any tune-path work")

    monkeypatch.setattr(tune_nn, "_ensure_data_from_s3", _boom)
    monkeypatch.setattr(tune_nn, "_resolve_n_jobs", _boom)
    monkeypatch.setenv("FF_TUNE_ENSEMBLE_AB", "1")
    monkeypatch.setattr(sys, "argv", ["tune_nn", "rb", "--checkpoint-s3", "--seed", "42"])
    tune_nn.main()
    assert calls == ["RB"]


def test_resolve_default_stacked_seeds_gpu_gated(monkeypatch):
    import src.tuning.ab_ensemble_seeds as ab
    from src.tuning.ab_ensemble_seeds import (
        DEFAULT_STACKED_SEEDS,
        resolve_default_stacked_seeds,
        stacked_default_seed_list,
    )

    # Explicit wins regardless of device.
    assert resolve_default_stacked_seeds(0) == 0  # force eager
    assert resolve_default_stacked_seeds(8) == 8  # explicit width
    assert resolve_default_stacked_seeds("16") == 16  # env arrives as str
    with pytest.raises(SystemExit):
        resolve_default_stacked_seeds(1)  # N=1 rejected

    # Auto (None/empty): 24 on CUDA, 0 (eager) off-CUDA.
    monkeypatch.setattr(ab, "cuda_enabled", lambda: True, raising=False)
    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: True)
    assert resolve_default_stacked_seeds(None) == DEFAULT_STACKED_SEEDS == 24
    assert resolve_default_stacked_seeds("") == 24
    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: False)
    assert resolve_default_stacked_seeds(None) == 0  # CPU/MPS -> eager (no regress)

    assert stacked_default_seed_list() == list(range(42, 66))
    assert stacked_default_seed_list(3) == [42, 43, 44]


def test_apply_eager_graph_env_sets_fp16_fullgraph_and_holds_norm(monkeypatch):
    import os

    from src.tuning.ab_ensemble_seeds import apply_eager_graph_env

    # Register EVERY key apply_eager_graph_env writes for monkeypatch restore
    # (incl. FF_NN_FIXED_EPOCHS) so the regime can't leak into a later trainer
    # test — the function writes os.environ directly, and a plain delenv of an
    # absent key records no undo (the #1138 leak shape).
    for k in (
        "FF_NN_NORM",
        "FF_AMP_DTYPE",
        "FF_CUDA_GRAPH",
        "FF_CUDA_GRAPH_FULL",
        "FF_NN_FIXED_EPOCHS",
        "FF_COMPILE",
    ):
        monkeypatch.setenv(k, "managed-for-restore")
        monkeypatch.delenv(k)
    monkeypatch.setenv("FF_COMPILE", "1")  # must be cleared (incompatible)
    apply_eager_graph_env(30)
    assert os.environ["FF_NN_NORM"] == "layer"  # held constant with stacked arm
    assert os.environ["FF_AMP_DTYPE"] == "auto"  # -> FP32+TF32 (AMP off) on CUDA
    assert os.environ["FF_CUDA_GRAPH"] == "1"
    assert os.environ["FF_CUDA_GRAPH_FULL"] == "1"
    assert os.environ["FF_NN_FIXED_EPOCHS"] == "30"
    assert "FF_COMPILE" not in os.environ


def test_tune_nn_env_dispatch_routes_to_compare(monkeypatch):
    import sys

    from src.tuning import tune_nn

    calls: list[str] = []
    monkeypatch.setattr("src.tuning.ab_ensemble_seeds.run_compare_batch_entry", calls.append)
    monkeypatch.setattr(
        tune_nn, "_resolve_n_jobs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("early"))
    )
    monkeypatch.delenv("FF_TUNE_ENSEMBLE_AB", raising=False)
    monkeypatch.setenv("FF_TUNE_ENSEMBLE_COMPARE", "1")
    monkeypatch.setattr(sys, "argv", ["tune_nn", "rb", "--checkpoint-s3"])
    tune_nn.main()
    assert calls == ["RB"]


def test_run_compare_batch_entry_env_knobs(monkeypatch):
    import src.tuning.ab_ensemble_seeds as ab

    seen: dict = {}

    def fake_compare(position, n_seeds, fixed_epochs):
        seen.update(position=position, n_seeds=n_seeds, fixed_epochs=fixed_epochs)
        return {"position": position, "stacked_speedup_per_seed": 4.0}

    monkeypatch.setattr(ab, "run_compare", fake_compare)
    monkeypatch.setattr("src.tuning.tune_nn._ensure_data_from_s3", lambda: None)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("FF_COMPARE_SEEDS", "4")
    monkeypatch.setenv("FF_COMPARE_FIXED_EPOCHS", "30")
    ab.run_compare_batch_entry("RB")
    assert seen == {"position": "RB", "n_seeds": 4, "fixed_epochs": 30}

    with pytest.raises(SystemExit):
        ab.run_compare_batch_entry("K")  # not a flat-history ensemble position


def test_run_batch_entry_env_knobs_and_parity_gate(monkeypatch):
    import os

    import src.tuning.ab_ensemble_seeds as ab

    seen: dict = {}

    def fake_run(position, n_seeds, fixed_epochs, parity_check):
        seen.update(
            position=position, n_seeds=n_seeds, fixed_epochs=fixed_epochs, parity=parity_check
        )
        return {"position": position, "parity": {"ok": True, "worst_diff_over_spread": 0.001}}

    monkeypatch.setattr(ab, "run_ensemble_ab", fake_run)
    monkeypatch.setattr("src.tuning.tune_nn._ensure_data_from_s3", lambda: None)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("FF_ENSEMBLE_SEEDS", "4")
    monkeypatch.setenv("FF_ENSEMBLE_FIXED_EPOCHS", "7")
    monkeypatch.setenv("FF_ENSEMBLE_PARITY", "1")
    # run_batch_entry writes FF_FORCE_DROPOUT_ZERO directly; manage the key
    # via setenv-then-delenv so teardown force-restores it (no worker leak).
    monkeypatch.setenv("FF_FORCE_DROPOUT_ZERO", "managed-for-restore")
    monkeypatch.delenv("FF_FORCE_DROPOUT_ZERO")
    ab.run_batch_entry("RB")
    assert seen == {"position": "RB", "n_seeds": 4, "fixed_epochs": 7, "parity": True}
    assert os.environ["FF_FORCE_DROPOUT_ZERO"] == "1"

    # FF_ENSEMBLE_PARITY=0: no sequential arm, no dropout-zero forcing.
    monkeypatch.setenv("FF_ENSEMBLE_PARITY", "0")
    monkeypatch.delenv("FF_FORCE_DROPOUT_ZERO", raising=False)
    seen.clear()
    ab.run_batch_entry("RB")
    assert seen["parity"] is False
    assert "FF_FORCE_DROPOUT_ZERO" not in os.environ

    # Parity drift must fail the Batch job (non-zero exit), report already out.
    monkeypatch.setenv("FF_ENSEMBLE_PARITY", "1")
    monkeypatch.setattr(
        ab,
        "run_ensemble_ab",
        lambda *a, **k: {"parity": {"ok": False, "worst_diff_over_spread": 9.9}},
    )
    with pytest.raises(SystemExit):
        ab.run_batch_entry("RB")
