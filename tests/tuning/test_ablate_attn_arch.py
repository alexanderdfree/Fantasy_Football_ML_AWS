"""Smoke tests for src/tuning/ablate_attn_arch.py.

Cover loadability, the variant→cfg-key mapping, the config-build surface, and the
decision-table sentinel — without invoking the real pipeline ``run`` (a
multi-minute training run, out of scope for unit tests).

The most important guard here is ``test_flag_keys_are_read_by_the_factory``: a
variant that toggles a cfg key the attention factory does not read is a silent
no-op (the key falls through to its OFF default with no error) — exactly the
failure mode this project warns about — so we assert every key the variants touch
still appears in ``neural_net.py``'s factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.shared.neural_net as nn_mod
from src.tuning import ablate_attn_arch as aaa

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    for attr in (
        "VARIANTS",
        "FLAG_VARIANTS",
        "KNOWN_FLAG_KEYS",
        "BASELINE",
        "main",
        "print_summary",
        "run_variant",
        "_extract",
        "_make_cfg",
    ):
        assert hasattr(aaa, attr)


def test_variants_cover_the_seven_prs_plus_baseline_and_alibi_split():
    """Baseline + one variant per PR, with ALiBi split into stacked/replacing PE.

    Adding or removing a variant requires updating this test — it is the
    inventory of what the ablation claims to cover.
    """
    assert aaa.BASELINE == "baseline"
    assert set(aaa.VARIANTS) == {
        "baseline",
        "temp",
        "seqdrop",
        "swiglu",
        "entropy",
        "alibi",
        "alibi_only",
        "selfattn",
        "condq",
    }
    assert aaa.BASELINE not in aaa.FLAG_VARIANTS
    assert set(aaa.FLAG_VARIANTS) == set(aaa.VARIANTS) - {"baseline"}


def test_baseline_override_is_empty():
    """The baseline must be the all-OFF production reference (no overrides)."""
    assert aaa.VARIANTS[aaa.BASELINE][1] == {}


def test_variant_overrides_match_documented_flags():
    """Lock the exact cfg key each variant toggles, including types (a float flag
    set to ``True`` would be a different experiment)."""
    o = {k: v[1] for k, v in aaa.VARIANTS.items()}
    assert o["temp"] == {"attn_learn_temperature": True}
    assert o["seqdrop"] == {"attn_history_dropout": 0.1}
    assert isinstance(o["seqdrop"]["attn_history_dropout"], float)
    assert o["swiglu"] == {"attn_use_swiglu_encoder": True}
    assert o["entropy"] == {"attn_entropy_coeff": 0.01}
    assert isinstance(o["entropy"]["attn_entropy_coeff"], float)
    assert o["alibi"] == {"attn_use_alibi_bias": True}
    # alibi_only must disable the production positional encoding so ALiBi replaces
    # it rather than stacking on top.
    assert o["alibi_only"] == {"attn_use_alibi_bias": True, "attn_positional_encoding": False}
    assert o["selfattn"] == {"attn_self_layers": 1}
    assert isinstance(o["selfattn"]["attn_self_layers"], int)
    assert o["condq"] == {"attn_condition_queries_on_static": True}


def test_every_override_key_is_declared_known():
    """No variant may touch a key outside KNOWN_FLAG_KEYS (typo guard)."""
    for v in aaa.FLAG_VARIANTS:
        assert set(aaa.VARIANTS[v][1]) <= aaa.KNOWN_FLAG_KEYS, v


def test_flag_keys_are_read_by_the_factory():
    """Every KNOWN_FLAG_KEYS entry must still be consumed by the attention factory
    as ``cfg.get("<key>"...)``. If a flag is renamed/dropped in neural_net.py this
    fails loudly instead of silently producing a no-op variant."""
    source = Path(nn_mod.__file__).read_text()
    for key in aaa.KNOWN_FLAG_KEYS:
        assert f'cfg.get("{key}"' in source, key


def test_make_cfg_disables_lightgbm_applies_override_and_deep_copies():
    base = {
        "targets": ["rushing_yards", "rushing_tds"],
        "train_lightgbm": True,
        "huber_deltas": {"rushing_yards": 15.0},
    }
    cfg = aaa._make_cfg(base, {"attn_use_alibi_bias": True})

    assert cfg["train_lightgbm"] is False
    assert cfg["attn_use_alibi_bias"] is True
    # Base config is untouched (deep copy, no leak).
    assert base["train_lightgbm"] is True
    assert "attn_use_alibi_bias" not in base
    # Nested containers are copied, not shared.
    cfg["huber_deltas"]["rushing_yards"] = 1.0
    assert base["huber_deltas"]["rushing_yards"] == 15.0


def test_extract_raises_when_metrics_missing():
    with pytest.raises(RuntimeError, match="missing metrics"):
        aaa._extract({"attn_nn_metrics": {}}, "alibi", 42, ["rushing_yards"])


def _row(variant: str, seed: int, attn: float, base: float, ridge: float) -> dict:
    return {
        "variant": variant,
        "seed": seed,
        "attn_fp_mae": attn,
        "base_fp_mae": base,
        "ridge_fp_mae": ridge,
        "attn_targets": {"rushing_yards": 20.2, "rushing_tds": 0.29},
    }


def test_print_summary_handles_empty_rows():
    """Edge case: empty rows must not raise (no mean-of-empty crash)."""
    assert aaa.print_summary([], ["rushing_yards"], [aaa.BASELINE]) is False


def test_print_summary_sentinel_passes_when_ridge_matches(capsys):
    targets = ["rushing_yards", "rushing_tds"]
    rows = [
        _row("baseline", 42, 4.21, 4.15, 4.72),
        _row("alibi", 42, 4.18, 4.15, 4.72),
    ]
    ok = aaa.print_summary(rows, targets, ["baseline", "alibi"])
    out = capsys.readouterr().out
    assert ok is True
    assert "VERDICT" in out
    assert "Δ vs baseline" in out


def test_print_summary_sentinel_fails_when_ridge_differs(capsys):
    """A Ridge-MAE mismatch means the variants saw different data/seed — flag it
    (False) rather than report a bogus flag delta."""
    targets = ["rushing_yards", "rushing_tds"]
    rows = [
        _row("baseline", 42, 4.21, 4.15, 4.72),
        _row("alibi", 42, 4.18, 4.15, 4.99),
    ]
    ok = aaa.print_summary(rows, targets, ["baseline", "alibi"])
    out = capsys.readouterr().out
    assert ok is False
    assert "MISMATCH" in out
