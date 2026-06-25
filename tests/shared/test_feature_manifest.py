"""Merge-time feature-change drift guard.

This is the *feature-level* counterpart to the file-level ``scope_positions``
detect machinery: it derives a canonical per-position feature manifest from the
live ``POSITION_CONFIG`` objects and fails loud when it drifts from the
committed snapshot (``feature_manifest.snapshot.json``). The snapshot's PR diff
is the human-readable "what features changed on this merge" record; the failure
message names the downstream consumers that must move with the change.

See ``src/scripts/feature_manifest.py`` for the builder + the
``MANIFEST_FIELDS`` / ``IGNORED_FIELDS`` field classification.

Deliberately additive — invariants already covered elsewhere are NOT
re-asserted here:
  • attn_static ⊆ include + no temporal leak  -> tests/test_invariants.py,
    tests/test_attn_static_columns.py
  • loss_weight ≈ 2.0 / huber_delta (values)  -> tests/test_invariants.py
  • config field -> served attn-kwargs (mapped) -> tests/shared/test_registry_coverage.py
  • whitelisted cols present in splits         -> .github/workflows/refresh-splits.yml verify gate
  • train/serve feature/NN/scaler hash drift   -> tests/shared/test_smoke_test.py
"""

import pytest

from src.scripts.feature_manifest import (
    CONSUMER_CHECKLIST,
    IGNORED_FIELDS,
    MANIFEST_FIELDS,
    SNAPSHOT_PATH,
    build_manifest,
    diff_manifests,
    load_snapshot,
    positionconfig_fields,
)
from src.shared.registry import ALL_POSITIONS

# Targets legitimately allowed to skip the per-head non-negativity clamp (a
# future signed-output head). Empty today — all six positions clamp every head.
SIGNED_TARGET_ALLOWLIST: frozenset[str] = frozenset()


@pytest.mark.unit
def test_snapshot_exists():
    assert SNAPSHOT_PATH.exists(), (
        f"missing {SNAPSHOT_PATH}; generate it with "
        "`python -m src.scripts.feature_manifest --write`"
    )


@pytest.mark.unit
def test_manifest_matches_snapshot():
    """THE detector: live config must match the committed feature manifest.

    Fails on any add/remove/rename of a tracked feature, target, attention
    input, or model-shape knob until the snapshot is regenerated (the explicit
    acknowledgement) and the downstream consumers are updated.
    """
    snapshot = load_snapshot()
    assert snapshot is not None, (
        "no feature-manifest snapshot; run `python -m src.scripts.feature_manifest --write`"
    )
    live = build_manifest()
    if live != snapshot:
        report = "\n".join(diff_manifests(snapshot, live))
        pytest.fail(
            "Feature manifest drifted from the committed snapshot.\n\n"
            f"{report}\n\n{CONSUMER_CHECKLIST}",
            pytrace=False,
        )


@pytest.mark.unit
def test_positionconfig_fields_are_classified():
    """Every PositionConfig field must be classified MANIFEST or IGNORED.

    Guards the "new attn_*/nn_* knob added to config but never threaded into
    serving" trap (#121 / the 2026-06-15 staleness NaN): a brand-new field lands
    in neither set and trips this test, forcing the author to decide whether it
    changes the feature/model contract (MANIFEST — capture it + thread arch
    knobs through registry._flat/_nested_attn_kwargs_static) or is a pure
    hyperparameter (IGNORED).
    """
    fields = positionconfig_fields()
    unclassified = fields - (MANIFEST_FIELDS | IGNORED_FIELDS)
    assert not unclassified, (
        f"PositionConfig grew unclassified field(s): {sorted(unclassified)}.\n"
        "Add each to MANIFEST_FIELDS (changes the feature/model contract — also "
        "thread any arch knob through registry._flat/_nested_attn_kwargs_static) "
        "or IGNORED_FIELDS (pure training hyperparameter) in "
        "src/scripts/feature_manifest.py."
    )
    stale = (MANIFEST_FIELDS | IGNORED_FIELDS) - fields
    assert not stale, (
        f"feature_manifest classifies field(s) that no longer exist on "
        f"PositionConfig: {sorted(stale)} — remove them."
    )
    overlap = MANIFEST_FIELDS & IGNORED_FIELDS
    assert not overlap, f"fields classified as BOTH manifest and ignored: {sorted(overlap)}"


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_every_target_has_loss_head_and_clamp(pos):
    """Structural loss-head completeness (additive to test_invariants's value check).

    A target with no head-loss family would silently fall through to a default
    head; a target missing from nn_non_negative_targets would emit signed
    predictions for a non-negative quantity.
    """
    manifest = build_manifest()[pos]
    targets = manifest["targets"]
    head_losses = manifest["loss_heads"]["head_losses"]
    non_negative = set(manifest["loss_heads"]["non_negative_targets"])

    missing_head = [t for t in targets if not head_losses.get(t)]
    assert not missing_head, f"{pos}: targets without a head_losses family: {missing_head}"

    unclamped = set(targets) - non_negative - SIGNED_TARGET_ALLOWLIST
    assert not unclamped, (
        f"{pos}: targets missing from nn_non_negative_targets: {sorted(unclamped)} "
        "(add them, or to SIGNED_TARGET_ALLOWLIST if a signed head is intended)."
    )


@pytest.mark.unit
def test_build_manifest_covers_all_positions():
    manifest = build_manifest()
    assert set(manifest) == set(ALL_POSITIONS)
    for pos, m in manifest.items():
        assert m["targets"], f"{pos}: no targets"
        assert m["features"], f"{pos}: no features"
        assert m["n_features"] == len(m["features"])
        assert m["attn"]["served_kwargs_keys"], f"{pos}: empty served attn-kwargs"
