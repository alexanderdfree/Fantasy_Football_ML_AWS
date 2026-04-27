"""Invariants from CLAUDE.md "Conventions that bite if ignored".

These tests fail loudly when a future change drifts away from a documented
convention. Each test cites the CLAUDE.md section and (where applicable) the
TODO.md archive entry that motivated it.

Lives in the root ``tests/`` directory so it runs in the "shared" CI shard.
All tests are config inspections / AST scans — no model training.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from src.shared.registry import ALL_POSITIONS


def _config(pos: str):
    """Import and return the ``src/{pos}/config`` module."""
    return importlib.import_module(f"src.{pos.lower()}.config")


# ---------------------------------------------------------------------------
# Invariant 1: LOSS_WEIGHTS[t] * HUBER_DELTAS[t] ≈ 2.0 per Huber head.
# ---------------------------------------------------------------------------
# Why: CLAUDE.md "Loss weights are tuned inverse-to-Huber-delta".
# TODO.md Fixed archive entry "Huber delta asymmetry across targets starved
# count heads" documents the regression: pre-rebalance, yards targets
# (delta in [15, 30]) dominated count heads (delta in [0.25, 0.5]) ~20-2500x
# per sample, collapsing the count heads to the mean. The fix paired
# LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS across RB / QB / WR / TE.
# Without this guard, future Huber-delta tuning will silently re-introduce
# the same imbalance.

# Permissive tolerance — the rule is approximate (configs have rounded values
# like 0.133 ≈ 2/15, 0.067 ≈ 2/30) and some heads use slightly different
# weights for tuning reasons. The threshold is set well below the >2x
# imbalance that would actually starve a head.
_LW_HD_PRODUCT_TOLERANCE = 0.5


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_loss_weights_match_huber_deltas(pos: str):
    """Every Huber head must have ``LOSS_WEIGHTS[t] * HUBER_DELTAS[t] ≈ 2.0``.

    Non-Huber heads (Poisson NLL, hurdle-NegBin) have no Huber delta and use
    weight 1.0 by convention — they're skipped here.
    """
    cfg = _config(pos)
    lw = getattr(cfg, "LOSS_WEIGHTS", None)
    hd = getattr(cfg, "HUBER_DELTAS", None)
    if lw is None or hd is None:
        pytest.skip(f"{pos} has no LOSS_WEIGHTS or HUBER_DELTAS")

    head_losses = getattr(cfg, "HEAD_LOSSES", None)

    bad: list[tuple[str, float, float, float]] = []
    for t, weight in lw.items():
        # Skip targets that don't have a Huber delta (those are Poisson /
        # hurdle-NegBin / etc.) — the 2.0/delta rule only applies to Huber.
        if t not in hd:
            continue
        # Belt-and-suspenders: if HEAD_LOSSES is set and explicitly marks
        # this head as non-Huber, skip it. (HUBER_DELTAS shouldn't list a
        # non-Huber target, but the per-head loss family is the source of
        # truth.)
        if head_losses is not None and head_losses.get(t, "huber") != "huber":
            continue
        delta = hd[t]
        product = weight * delta
        if abs(product - 2.0) >= _LW_HD_PRODUCT_TOLERANCE:
            bad.append((t, weight, delta, product))

    assert not bad, (
        f"{pos} Huber heads violate the LOSS_WEIGHTS * HUBER_DELTAS ≈ 2.0 rule "
        f"(CLAUDE.md 'Loss weights are tuned inverse-to-Huber-delta'): "
        + "; ".join(
            f"{t}: w={w} * delta={d} = {p:.3f} (want |p-2.0|<{_LW_HD_PRODUCT_TOLERANCE})"
            for t, w, d, p in bad
        )
    )


# ---------------------------------------------------------------------------
# Invariant 2: ATTN_STATIC_FEATURES ⊆ INCLUDE_FEATURES (or ALL_FEATURES) per pos.
# ---------------------------------------------------------------------------
# Why: CLAUDE.md "Attention static-feature whitelist is separate per position".
# The attention NN's static branch reads columns from the engineered feature
# set. If ``ATTN_STATIC_FEATURES`` references a column that doesn't exist in
# the feature whitelist, training will KeyError or silently NaN.
#
# Position-specific shapes:
#   - QB/RB/WR/TE: ``INCLUDE_FEATURES`` is a dict-of-lists (categorical
#     buckets). ``ATTN_STATIC_FEATURES`` is built by flattening selected
#     buckets via ``ATTN_STATIC_CATEGORIES``. Subset relation must hold.
#   - DST: ``ALL_FEATURES`` is a flat list. ``ATTN_STATIC_FEATURES`` is
#     enumerated directly. Subset relation must hold against ALL_FEATURES.
#   - K: ``ATTN_STATIC_FEATURES`` *intentionally* includes ``ATTN_L1_FEATURES``
#     that are excluded from ``ALL_FEATURES`` (so Ridge / base NN don't see
#     them but the attention static branch does, via ``attn_static_from_df``).
#     Skipped with explanation — the K invariant is enforced separately in
#     tests/test_attn_static_columns.py.


def _flatten_include_features(include_features) -> set[str]:
    """Flatten the dict-of-lists or list shape of ``INCLUDE_FEATURES``."""
    if isinstance(include_features, dict):
        flat: list[str] = []
        for v in include_features.values():
            flat.extend(v)
        return set(flat)
    return set(include_features)


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_attn_static_features_subset_of_include(pos: str):
    """``ATTN_STATIC_FEATURES`` columns must all live in the position's
    feature whitelist (``INCLUDE_FEATURES`` for QB/RB/WR/TE, ``ALL_FEATURES``
    for DST). Otherwise the attention static branch would reference columns
    the feature pipeline doesn't produce.
    """
    cfg = _config(pos)
    attn_static = getattr(cfg, "ATTN_STATIC_FEATURES", None)
    if attn_static is None:
        pytest.skip(f"{pos} has no ATTN_STATIC_FEATURES")

    if pos == "K":
        # K legitimately puts ATTN_L1_FEATURES outside ALL_FEATURES — the
        # attn-static-from-df path reads them straight from the DataFrame.
        # That's enforced in tests/test_attn_static_columns.py; no parity
        # check makes sense here.
        pytest.skip(
            "K's ATTN_L1_FEATURES are intentionally excluded from ALL_FEATURES "
            "(see src/k/config.py 'ATTN_L1_FEATURES' comment + "
            "tests/test_attn_static_columns.py::TestKAttentionStaticFeatures)"
        )

    if hasattr(cfg, "INCLUDE_FEATURES"):
        whitelist_name = "INCLUDE_FEATURES"
        whitelist = _flatten_include_features(cfg.INCLUDE_FEATURES)
    elif hasattr(cfg, "ALL_FEATURES"):
        whitelist_name = "ALL_FEATURES"
        whitelist = set(cfg.ALL_FEATURES)
    else:
        pytest.skip(f"{pos} has no INCLUDE_FEATURES or ALL_FEATURES")

    missing = sorted(set(attn_static) - whitelist)
    assert not missing, (
        f"{pos}.ATTN_STATIC_FEATURES references columns not in {whitelist_name}: "
        f"{missing}. Either add them to {whitelist_name} or drop them from "
        f"ATTN_STATIC_FEATURES (CLAUDE.md 'Attention static-feature whitelist "
        f"is separate per position')."
    )


# ---------------------------------------------------------------------------
# Invariant 3: every direct ``MultiHeadNet(...)`` call site passes
# ``non_negative_targets=`` (or ``**kwargs`` that includes it).
# ---------------------------------------------------------------------------
# Why: CLAUDE.md "non_negative_targets is per-head, not global".
# TODO.md Fixed archive entry "run_cv_pipeline missing non_negative_targets
# on MultiHeadNet" — the CV path was missed once and DST's pts_allowed_bonus
# (range [-4, +10]) was incorrectly clamped to >= 0.
#
# Implementation: walk every ``.py`` under ``src/``, parse with ``ast``,
# find every ``Call`` whose ``func`` ends in ``MultiHeadNet`` (NOT the
# History/NestedHistory variants — those have their own contract). Compliant
# = either ``non_negative_targets=`` is a direct keyword argument, or the call
# uses ``**kwargs`` style starred unpacking (which may contain it; we can't
# statically verify).
#
# The factory ``build_multihead_net`` in src/shared/neural_net.py is the
# canonical source of truth and passes the kwarg directly. All wrappers
# must mirror that contract.

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _multihead_net_calls(tree: ast.AST):
    """Yield every AST ``Call`` node whose func name is exactly ``MultiHeadNet``.

    Matches both ``MultiHeadNet(...)`` (Name) and ``module.MultiHeadNet(...)``
    (Attribute). Excludes ``MultiHeadNetWithHistory`` /
    ``MultiHeadNetWithNestedHistory`` — those have separate factories and
    are not in scope for this invariant.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "MultiHeadNet") or (
            isinstance(func, ast.Attribute) and func.attr == "MultiHeadNet"
        ):
            yield node


def _call_passes_non_negative(call: ast.Call) -> bool:
    """Return True iff the call has ``non_negative_targets=...`` or ``**...``.

    Either explicitly naming the kwarg or unpacking a dict (which may contain
    it) is acceptable — the harness can't statically verify the dict's keys.
    """
    for kw in call.keywords:
        # ``**kwargs`` shows up as a keyword with ``arg=None``.
        if kw.arg is None:
            return True
        if kw.arg == "non_negative_targets":
            return True
    return False


@pytest.mark.unit
def test_every_multihead_net_call_passes_non_negative_targets():
    """Every direct ``MultiHeadNet(...)`` construction in src/ must include
    the ``non_negative_targets`` kwarg (or pass it via ``**kwargs``).
    """
    py_files = sorted(_SRC_ROOT.rglob("*.py"))
    assert py_files, f"No .py files found under {_SRC_ROOT}"

    offenders: list[str] = []
    for path in py_files:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover — would also break ruff/import
            continue
        for call in _multihead_net_calls(tree):
            if _call_passes_non_negative(call):
                continue
            offenders.append(f"{path.relative_to(_SRC_ROOT.parent)}:{call.lineno}")

    assert not offenders, (
        "Direct MultiHeadNet(...) calls missing non_negative_targets kwarg "
        "(CLAUDE.md 'non_negative_targets is per-head, not global'; "
        "TODO.md Fixed archive 'run_cv_pipeline missing non_negative_targets "
        f"on MultiHeadNet'): {offenders}"
    )


# ---------------------------------------------------------------------------
# Invariant 4: training and serving share the same feature-build entry point.
# ---------------------------------------------------------------------------
# Why: CLAUDE.md "Always diff training vs inference paths".
# TODO.md Fixed archive entries:
#   - "Weather/Vegas features missing at inference in src/serving/app.py" —
#     training merged schedule features but serving did not.
#   - the recurring class of bug where the two paths drift silently.
# Today both paths import ``build_position_features`` from
# ``src.shared.feature_build`` — that's the architectural fix that made the
# class of drift impossible. This test pins that import as the contract:
# if anyone ever inlines a parallel feature builder in either path, this
# invariant will fail.


@pytest.mark.unit
def test_training_and_serving_share_feature_builder():
    """Both ``src.shared.pipeline`` (training) and ``src.serving.app``
    (serving) must import ``build_position_features`` from the shared module.
    Centralising the feature build is the architectural guarantee that the
    two paths cannot drift on feature engineering.
    """
    from src.serving import app
    from src.shared import feature_build, pipeline

    canonical = feature_build.build_position_features

    # The training pipeline must use the canonical function.
    assert pipeline.build_position_features is canonical, (
        "src.shared.pipeline rebound build_position_features to a different "
        "object — the training path is no longer guaranteed to use the same "
        "feature builder as serving. CLAUDE.md 'Always diff training vs "
        "inference paths'."
    )

    # The serving app must use the canonical function too.
    assert app.build_position_features is canonical, (
        "src.serving.app rebound build_position_features to a different "
        "object — the serving path is no longer guaranteed to use the same "
        "feature builder as training. TODO.md Fixed archive 'Weather/Vegas "
        "features missing at inference in src/serving/app.py'."
    )
