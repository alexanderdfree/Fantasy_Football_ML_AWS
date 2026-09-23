> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Nested (K) served-kwargs builder dropped `nn_head_hidden_overrides` the training factory consumes
- **File(s):** [src/shared/registry.py](../../src/shared/registry.py) (`_nested_attn_kwargs_static`), [tests/shared/test_registry_coverage.py](../../tests/shared/test_registry_coverage.py) (parity guard), [tests/k/test_registry.py](../../tests/k/test_registry.py) (shape-parity regression). audit-1499 Tier B PR; issue #1503.
- **What:** `_nested_attn_kwargs_static` forwarded every other `nn_`/`attn_` `PositionConfig` field that `build_multihead_net_with_nested_history` consumes, but never `nn_head_hidden_overrides` — the flat builder forwards it conditionally, the nested one silently dropped it. Latent only because K configures no per-head override; the first K override would have trained one head width and served another, and `load_state_dict` would have failed with a size mismatch (the 2026-06-15 architecture-staleness class, same as the condq `cond_proj` trap). The parity guard mapped config fields to ctor params only by identity or `attn_`-prefix stripping, so the `nn_` spelling was invisible to it.
- **Fix:** Mirror the flat builder (`if pc.nn_head_hidden_overrides: kwargs["head_hidden_overrides"] = dict(...)`), extend the guard to strip `nn_` too, and build its fixture with the conditional knobs populated (an empty fixture false-positives on the flat builder). A K regression test proves red-before/green-after by strict-loading a state_dict built from the override config into the served-kwargs model, with a positive control that the un-overridden kwargs cannot load it. Zero change to today's served kwargs.
- **Lesson:** A train/serve parity guard is only as good as its field-mapping heuristic — enumerate every prefix the config uses (`attn_`, `nn_`), and exercise conditional keys with a populated fixture, or the guard is blind exactly where the conditional branches live. "Latent because the config doesn't set it today" is one tuning change away from a silently NaN'ing position.

**Current equivalence evidence (2026-09-23):** The production K override map is
empty on both main and candidate. The [shared verification receipt](evidence/audit-1499-tier-b-9758de79-20260923.json)
records identical effective training recipes and inference specifications for
all six positions, plus byte-equal K constructor state and forward outputs
from an identically initialized, untrained network. No fitting occurred.
Strict checkpoint-load regression tests cover an explicit `fg_misses=40`
override and prove that the legacy-equivalent constructor cannot load that
state. This establishes default-constructor compatibility; it is not a new
trained-checkpoint accuracy benchmark. The DST incident record describes the
same receipt's frozen-data proof and its limits.
