"""A/B spec: the default-OFF attention-architecture extensions — stacked-seed (N=24) form.

The GPU-default N=24 stacked-seed port of [ablate_attn_arch.py](ablate_attn_arch.py). It
re-expresses the seven default-OFF attention flags (PRs #109-121) as ``ab_harness`` Variants so
they inherit the stacked seed-ensemble regime (24 seeds vmapped on one host thread), the
parallel fan-out, artifact isolation, and the Ridge data-identity sentinel — instead of running
eager on ``ablation_runner``.

Scope vs the eager script (which stays for the per-target table and the dropped arm):

* The ``entropy`` flag (``attn_entropy_coeff``) is **DROPPED**. Its forward writes an attention
  entropy side-channel (``self.last_attn_entropy``), a vmap side effect — ``capture_seeds``
  raises on ``attn_entropy_coeff != 0``. Run it eager via ``ablate_attn_arch.py`` if needed.
* **Attention-only by construction**: every flag feeds the NN only, so each arm is
  ``expect_ridge_identical=True`` (the harness Ridge sentinel must stay byte-identical across
  arms; a moved Ridge proves the flag leaked into the shared Ridge/LGBM/NN-static data path).
* **Judge on overall fantasy-point MAE Δ-vs-baseline** (the default metric the harness prints).
  The legacy per-target attention-MAE table is diagnostic, not the verdict — and the per-target
  arrays aren't surfaced under stacking (only ``pred_attn_nn_total`` reaches the metric).
* ``selfattn`` (``attn_self_layers``) vmaps correctly, but ``nn.MultiheadAttention`` currently
  hits a ``torch.func`` batching-rule fallback (a perf drop, not a correctness issue), so that
  one arm may run slower than the others under stacking.

The flag → cfg-override mapping is imported from ``ablate_attn_arch.VARIANTS`` (one source of
truth), minus the dropped ``entropy`` arm.

Run::

    python -m src.tuning.ab_attn_arch --list                         # show the grid, run nothing
    python -m src.tuning.launch_ab --spec src.tuning.ab_attn_arch     # GPU fleet, N=24 stacked
    python -m src.tuning.ab_attn_arch --positions RB --no-stacked-seeds --seeds 42  # eager smoke
    python -m src.tuning.ab_attn_arch --positions QB                  # gated 2nd-position stress
"""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.ablate_attn_arch import BASELINE
from src.tuning.ablate_attn_arch import VARIANTS as _LEGACY_VARIANTS

POSITIONS = ["RB"]  # the documented ablation workhorse; --positions QB is the gated stress test.
# No SEEDS — leave it unset so the grid inherits the GPU-default N=24 stacked width
# (stacked_default_seed_list()) on CUDA and the lean 3-seed DEFAULT_SEEDS on CPU/CI. Defining
# SEEDS here would override and defeat the N=24 inheritance (ab_harness.resolve_spec).

# Vmap-incompatible arms dropped from the stacked port (see module docstring).
_DROP = {"entropy"}


def _apply(override: dict):
    """Build a cfg mutator that applies one variant's flag override(s).

    ``override`` is bound per-variant (default-arg) so the comprehension below produces
    independent mutators rather than all closing over the loop's last value. The harness hands
    every mutator a private deep copy, so the in-place ``update`` is safe.
    """

    def _mut(cfg: dict, _override: dict = override) -> dict:
        cfg.update(_override)
        return cfg

    return _mut


VARIANTS = [Variant(BASELINE, label=_LEGACY_VARIANTS[BASELINE][0])] + [
    Variant(
        name,
        cfg_mutator=_apply(override),
        expect_ridge_identical=True,  # attention-only — Ridge must stay byte-identical
        label=label,
    )
    for name, (label, override) in _LEGACY_VARIANTS.items()
    if name != BASELINE and name not in _DROP
]


if __name__ == "__main__":
    ab_main(__spec__.name)
