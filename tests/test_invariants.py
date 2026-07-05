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
# Why: AGENTS.md "Loss weights are tuned inverse-to-Huber-delta".
# TODO.md Fixed archive entry "Huber delta asymmetry across targets starved
# count heads" documents the regression: pre-rebalance, yards targets
# (delta in [15, 30]) dominated count heads (delta in [0.25, 0.5]) ~20-2500x
# per sample, collapsing the count heads to the mean. The fix paired
# LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS across RB / QB / WR / TE.
# Post-#870 (Huber→MSE switch) the yards heads are MSE at 1/δ and no current
# head declares a Huber loss, so this guard is dormant — it exists to catch a
# future Huber head reintroduced without the matching 2.0/δ weight.
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
    # Read from POSITION_CONFIG (per-position config classes don't export
    # UPPERCASE module-level constants — see e.g. src/qb/config.py header).
    pc = _config(pos).POSITION_CONFIG
    lw = pc.loss_weights
    hd = pc.huber_deltas
    head_losses = pc.head_losses

    assert lw, (
        f"{pos}.POSITION_CONFIG.loss_weights is empty — every position trains "
        f"a NN with weighted per-target losses (CLAUDE.md 'Loss weights are "
        f"tuned inverse-to-Huber-delta')."
    )

    bad: list[tuple[str, float, float, float]] = []
    for t, weight in lw.items():
        # Skip targets that don't have a Huber delta (those are Poisson /
        # hurdle-NegBin / etc.) — the 2.0/delta rule only applies to Huber.
        if t not in hd:
            continue
        # Belt-and-suspenders: if head_losses is set and explicitly marks
        # this head as non-Huber, skip it. (huber_deltas shouldn't list a
        # non-Huber target, but the per-head loss family is the source of
        # truth.)
        if head_losses and head_losses.get(t, "huber") != "huber":
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
#   - DST/K: ``ALL_FEATURES`` is a flat list. ``ATTN_STATIC_FEATURES`` is
#     enumerated directly. Subset relation must hold against ALL_FEATURES.


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
    """``POSITION_CONFIG.attn_static_features`` columns must all live in the
    position's feature whitelist (``include_features`` for QB/RB/WR/TE,
    ``all_features`` for DST/K). Otherwise the attention static branch would
    reference columns the feature pipeline doesn't produce.
    """
    pc = _config(pos).POSITION_CONFIG
    attn_static = pc.attn_static_features
    assert attn_static, (
        f"{pos}.POSITION_CONFIG.attn_static_features is empty — every position "
        f"trains the attention NN and must populate the static branch "
        f"whitelist (CLAUDE.md 'Attention static-feature whitelist is "
        f"separate per position')."
    )

    if pc.include_features:
        whitelist_name = "include_features"
        whitelist = _flatten_include_features(pc.include_features)
    elif pc.all_features:
        whitelist_name = "all_features"
        whitelist = set(pc.all_features)
    else:
        raise AssertionError(
            f"{pos}.POSITION_CONFIG has neither include_features nor "
            f"all_features — one of them must be populated (CLAUDE.md "
            f"'Feature whitelist is explicit, not inferred')."
        )

    missing = sorted(set(attn_static) - whitelist)
    assert not missing, (
        f"{pos}.POSITION_CONFIG.attn_static_features references columns not in "
        f"{whitelist_name}: {missing}. Either add them to {whitelist_name} or "
        f"drop them from attn_static_features (CLAUDE.md 'Attention "
        f"static-feature whitelist is separate per position')."
    )


# ---------------------------------------------------------------------------
# Invariant 2b: ATTN_STATIC_FEATURES survive build_position_features at runtime
# with non-degenerate variance.
# ---------------------------------------------------------------------------
# Why: the config-level subset check (invariant 2) only verifies the whitelist
# names — it can't catch H2-class bugs where a column is in the config but
# zeroed by ``merge_schedule_features`` at runtime (H2: DST ``spread_line`` /
# ``div_game`` were dropped by the schedule merge, then re-introduced as
# zero-fills by ``merge_schedule_features``'s own fillna(0) on the merge-back
# loop — so they survived the config check but reached the model as
# constant zeros).
#
# ``build_position_features`` itself no longer silently back-fills missing
# whitelist cols — it now raises ``KeyError``. So the (a) presence check
# below mainly guards against an internal contract bug; the (b) std() > 0
# check still catches H2-class zeroing introduced by individual merge or
# add_features steps that produce the column but with no variance.
#
# This runtime invariant runs ``build_position_features`` against tiny real
# splits per position and asserts every ``attn_static_feature`` column (a)
# exists in the engineered DataFrame and (b) has ``std() > 0`` (i.e. carries
# real signal). A per-position allowlist absorbs columns that are legitimately
# near-constant on the tiny-fixture slice (e.g. prior-season aggregates that
# are zero for first-season rows) so the test doesn't flag known-tiny-fixture
# artifacts.

# Cross-position allowlist of attn_static_features columns that are legitimately
# constant on the tiny-fixture slice from ``load_tiny_splits``. Keys are
# position codes; values are sets of column names known to have ``std() == 0``
# in the tiny synthetic dataset because of fixture truncation (e.g. RB
# prior-season redzone aggregates rely on the season-before history, which is
# absent for the earliest tiny-fixture season).
_TINY_FIXTURE_CONSTANT_ALLOWLIST: dict[str, set[str]] = {
    "RB": {
        "prior_season_total_redzone_touches",
        "prior_season_mean_redzone_touches_per_game",
    },
}


@pytest.mark.e2e
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_attn_static_features_survive_build_position_features(pos: str, tmp_path):
    """Runtime invariant — every ``attn_static_feature`` column must exist on
    the engineered DataFrame and carry non-zero variance after
    ``build_position_features`` runs against tiny real data.

    Catches the H2 failure class: a column listed in
    ``POSITION_CONFIG.attn_static_features`` that is silently dropped or
    back-filled with a constant zero by ``merge_schedule_features`` /
    ``build_position_features``. The config-level subset check (invariant 2)
    cannot detect this — both ``spread_line`` and ``div_game`` were in DST's
    config when H2 fired.
    """
    splits_root = Path(__file__).resolve().parent.parent / "data" / "splits"
    # The K / DST loaders rebuild their per-position dataset from raw caches;
    # the player-position loaders slice ``data/splits/*.parquet``. Either way,
    # require_splits is the canonical gate (DST/K still read raw caches under
    # ``data/`` that the local-setup data pull also populates).
    from tests._skip_helpers import require_splits

    require_splits(splits_root)

    from src.shared.feature_build import build_position_features
    from tests._pipeline_e2e_utils import build_tiny_config, load_tiny_splits

    train, val, test = load_tiny_splits(pos)
    cfg = build_tiny_config(pos)

    pos_train = cfg["compute_targets_fn"](train.copy())
    pos_val = cfg["compute_targets_fn"](val.copy())
    pos_test = cfg["compute_targets_fn"](test.copy())

    feature_cols = cfg["get_feature_columns_fn"]()
    pos_train, _pos_val, _pos_test = build_position_features(
        pos_train, pos_val, pos_test, cfg, feature_cols
    )

    pc = _config(pos).POSITION_CONFIG
    attn_static = list(pc.attn_static_features)
    assert attn_static, f"{pos}.POSITION_CONFIG.attn_static_features is empty"

    # (a) Every attn_static column must be present on the engineered training
    # DataFrame. ``build_position_features`` now raises ``KeyError`` on
    # missing whitelist cols, so reaching this assert with a missing col
    # would indicate the col is in ``attn_static_features`` but not in the
    # broader ``include_features`` set — a config bug, not a data bug.
    missing = [c for c in attn_static if c not in pos_train.columns]
    assert not missing, (
        f"{pos}: attn_static_features columns absent from engineered DataFrame "
        f"after build_position_features: {missing}. Either feature engineering "
        f"failed to produce them or the column name in attn_static_features is wrong."
    )

    # (b) Every attn_static column must carry non-zero variance, except those
    # in the per-position allowlist for known tiny-fixture artifacts.
    allowed_constant = _TINY_FIXTURE_CONSTANT_ALLOWLIST.get(pos, set())
    zero_std: list[str] = []
    for col in attn_static:
        if col in allowed_constant:
            continue
        if float(pos_train[col].std()) == 0.0:
            zero_std.append(col)

    assert not zero_std, (
        f"{pos}: attn_static_features columns reach the model with std()==0 — "
        f"these were silently zeroed by the feature pipeline (likely the "
        f"build_position_features catch-all back-fill, see H2 archive). "
        f"Either fix feature engineering to populate them or extend "
        f"_TINY_FIXTURE_CONSTANT_ALLOWLIST['{pos}'] if the constancy is a "
        f"legitimate artifact of the tiny fixture slice. Zero-std columns: {zero_std}"
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
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    """Both ``src.shared.pipeline`` (training) and ``src.serving.core``
    (serving) must import ``build_position_features`` from the shared module.
    Centralising the feature build is the architectural guarantee that the
    two paths cannot drift on feature engineering.
    """
    from src.serving import core
    from src.shared import feature_build, pipeline

    canonical = feature_build.build_position_features

    # The training pipeline must use the canonical function.
    assert pipeline.build_position_features is canonical, (
        "src.shared.pipeline rebound build_position_features to a different "
        "object — the training path is no longer guaranteed to use the same "
        "feature builder as serving. CLAUDE.md 'Always diff training vs "
        "inference paths'."
    )

    # The serving inference path (src.serving.core) must use the canonical function too.
    assert core.build_position_features is canonical, (
        "src.serving.core rebound build_position_features to a different "
        "object — the serving path is no longer guaranteed to use the same "
        "feature builder as training. TODO.md Fixed archive 'Weather/Vegas "
        "features missing at inference in src/serving/app.py'."
    )
