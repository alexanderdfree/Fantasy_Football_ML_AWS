"""Unit tests for the stacked attn-arch A/B spec (src/tuning/ab_attn_arch.py).

No training in the resolve/mutator tests (that's the GPU-fleet run). Coverage: spec resolution
(the ``entropy`` arm is DROPPED for the stacked port; attention-only ⇒
``expect_ridge_identical=True``), the flag mutators (touch only ``KNOWN_FLAG_KEYS`` — guards the
typo'd-no-op footgun), and a CPU vmap smoke for the ``selfattn`` flag (``nn.MultiheadAttention``
under ``torch.func`` — the one path the existing ensemble tests don't already cover).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tuning import ab_harness as H

pytestmark = pytest.mark.unit

_FLAGS = {"temp", "seqdrop", "swiglu", "alibi", "alibi_only", "selfattn", "condq"}


def test_attn_arch_spec_resolves_without_entropy():
    """RB-default; baseline + 7 flags with the ``entropy`` arm DROPPED. Every flag is a
    cfg-mutator-only NN change (no frame injector) declaring ``expect_ridge_identical=True`` (the
    flags feed the NN only — Ridge must stay byte-identical); ``baseline`` is identity."""
    spec = H.resolve_spec("src.tuning.ab_attn_arch")
    assert spec.dotted == "src.tuning.ab_attn_arch"
    assert spec.positions == ["RB"]
    assert spec.baseline == "baseline"
    assert (
        "entropy" not in spec.variants
    )  # vmap side-channel reject — dropped from the stacked port
    assert set(spec.variants) == {"baseline", *_FLAGS}
    assert spec.variants["baseline"].is_baseline_shape
    for name in _FLAGS:
        v = spec.variants[name]
        assert v.cfg_mutator is not None
        assert v.frame_injector is None  # NN-config only — no data injection
        assert v.expect_ridge_identical is True


def test_attn_arch_mutators_touch_only_known_flag_keys():
    """A typo'd cfg key silently falls through to its OFF default (a no-op variant) — the failure
    mode the legacy ``KNOWN_FLAG_KEYS`` set guards. Every mutator must change exactly known keys."""
    from src.tuning.ablate_attn_arch import KNOWN_FLAG_KEYS

    spec = H.resolve_spec("src.tuning.ab_attn_arch")
    base = {"sentinel": 0}
    for name in _FLAGS:
        out = spec.variants[name].cfg_mutator(dict(base))
        touched = set(out) - set(base)
        assert touched, f"{name} mutator is a no-op"
        assert touched <= KNOWN_FLAG_KEYS, (
            f"{name} touches unknown keys {touched - KNOWN_FLAG_KEYS}"
        )


def test_selfattn_flag_vmaps_under_ensemble_regime():
    """De-risk the one flag whose vmap-safety the existing ensemble tests don't exercise:
    ``attn_self_layers`` wires in a ``SelfAttentionBlock`` (``nn.MultiheadAttention``). Build
    LayerNorm models under the ensemble regime and run the REAL stacked fwd+bwd for 1 epoch on
    synthetic data — finite per-member predictions ⇒ the flag vmaps under ``torch.func``."""
    import torch

    from src.tuning.ab_ensemble_seeds import ensemble_env, predict_stacked, train_stacked
    from tests.tuning.test_ab_ensemble_seeds import (
        _build_models,
        _captures_for,
        _criterion,
        _synthetic,
        _test_capture,
        _tiny_cfg,
    )

    with ensemble_env(2):  # FF_NN_NORM=layer (buffer-free → vmap-safe), FP32, fixed epochs
        cfg = _tiny_cfg(attn_self_layers=1)
        models = _build_models(cfg, n_members=2)
        criterion = _criterion(cfg)
        feats, _y, loader = _synthetic()
        captures = _captures_for(models, criterion, loader)
        device = torch.device("cpu")
        params, buffers, template = train_stacked(captures, cfg, device, n_epochs=1)
        preds = predict_stacked(template, params, buffers, _test_capture(feats), device)

    assert len(preds) == 2
    assert all(np.all(np.isfinite(v)) for member in preds for v in member.values())
