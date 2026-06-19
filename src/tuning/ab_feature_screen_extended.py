"""A/B: EXTENDED feature-family screen for skill positions (QB/RB/WR/TE).

The sibling of the validated core-8 screen [ab_feature_screen.py](ab_feature_screen.py),
run as a SEPARATE screen so that one stays byte-identical (its #1172 regression
tests pin its design). This one adds the position-defining ``specific`` family
(and ``ewma`` where every screened position populates it) to the screened set, so
the Stage-1 family screen covers the whole ``include_features`` whitelist — not
just the eight families the core screen judged safe to toggle.

Why ``specific`` is screened here, not in the core screen: dropping the whole
``specific`` block can starve the model (it is the position-defining efficiency
signal), so its main effect is informative but its REMOVAL is rarely the answer —
resolve it to individual columns with the Stage-2 sub-screen
([ab_feature_subscreen.py](ab_feature_subscreen.py) ``--family specific``) before
acting. ``ewma`` is populated only for QB, so it is excluded from any multi-skill
screen (an empty drop would false-trip the Ridge sentinel) and screened
position-locally via the sub-screen.

Each variant is one Plackett-Burman row dropping the families flagged ``-1``;
dropping a family filters its columns out of BOTH model paths
(``get_feature_columns_fn`` for Ridge/LightGBM/base-NN and ``attn_static_features``
for the attention static branch). Every row declares
``expect_ridge_identical=False`` — a real drop MUST move Ridge, so a Δ=0 means
the drop silently didn't take and the harness fails the run loudly.

Run (local authoring smoke only — real screen runs on the GPU Batch fleet)::

    python -m src.tuning.ab_feature_screen_extended --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen_extended --stacked-seeds
    python -m src.tuning.ab_feature_screen_extended --positions WR --no-stacked-seeds
"""

from __future__ import annotations

from src.tuning.ab_harness import ab_main
from src.tuning.feature_groups import (
    EXTENDED_FAMILIES,
    SKILL_POSITIONS,
    build_drop_variants,
    main_effects,
    screenable_skill_families,
    skill_family_columns,
)

POSITIONS = ["RB"]  # lead; run QB/WR/TE via --positions (all carry include_features)

# Families non-empty across every skill position (so no drop variant no-ops and
# false-trips the Ridge sentinel): the core 8 + ``specific``. ``ewma`` is QB-only
# and drops out — screen it via the sub-screen.
SCREENED_FAMILIES: list[str] = screenable_skill_families(EXTENDED_FAMILIES, SKILL_POSITIONS)
_GROUP_COLS = skill_family_columns(SCREENED_FAMILIES, SKILL_POSITIONS, drop_empty=True)

VARIANTS, ROW_DROPS = build_drop_variants(_GROUP_COLS)
BASELINE = "baseline"


def feature_main_effects(
    variant_seed_value: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per screened family (MAE or RMSE values).

    Thin wrapper over :func:`feature_groups.main_effects` bound to this spec's
    design (``ROW_DROPS`` / ``SCREENED_FAMILIES``); pass the per-variant
    ``{seed: value}`` map for the model+metric you care about (extract it with
    ``feature_groups.extract_variant_seed_metric``).
    """
    return main_effects(variant_seed_value, ROW_DROPS, SCREENED_FAMILIES)


if __name__ == "__main__":
    ab_main(__spec__.name)
