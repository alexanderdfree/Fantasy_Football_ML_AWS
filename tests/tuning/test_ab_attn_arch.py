"""Unit tests for the stacked attn-arch A/B spec (src/tuning/ab_attn_arch.py).

No training here (that's the GPU-fleet run). Coverage: spec resolution (the ``entropy`` and
``selfattn`` arms are DROPPED for the stacked port — both are vmap-incompatible; attention-only ⇒
``expect_ridge_identical=True``) and the flag mutators (touch only ``KNOWN_FLAG_KEYS`` — guards the
typo'd-no-op footgun).

``selfattn`` is dropped because ``nn.MultiheadAttention``'s fused-SDPA ``attn_bias`` does not
compose with the stacked ensemble's extra ``torch.func.vmap`` "members" batch dim — every real-GPU
stacked seed errors ``attn_bias: wrong shape``. A prior CPU vmap smoke for it PASSED yet missed the
GPU-only SDPA failure ("GPU-guarded code is invisible to CPU unit tests"), so it was removed with
the arm; confirm/delete ``selfattn`` eagerly via ``ablate_attn_arch.py`` instead.
"""

from __future__ import annotations

import pytest

from src.tuning import ab_harness as H

pytestmark = pytest.mark.unit

_FLAGS = {"temp", "seqdrop", "swiglu", "alibi", "alibi_only", "condq"}


def test_attn_arch_spec_resolves_without_dropped_arms():
    """RB-default; baseline + 6 flags with the ``entropy`` and ``selfattn`` arms DROPPED (both
    vmap-incompatible — see module docstring). Every flag is a cfg-mutator-only NN change (no
    frame injector) declaring ``expect_ridge_identical=True`` (the flags feed the NN only — Ridge
    must stay byte-identical); ``baseline`` is identity."""
    spec = H.resolve_spec("src.tuning.ab_attn_arch")
    assert spec.dotted == "src.tuning.ab_attn_arch"
    assert spec.positions == ["RB"]
    assert spec.baseline == "baseline"
    # Both dropped from the stacked port: entropy is an attention-entropy side-channel; selfattn's
    # nn.MultiheadAttention SDPA attn_bias won't compose with the vmap "members" batch dim.
    assert "entropy" not in spec.variants
    assert "selfattn" not in spec.variants
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
