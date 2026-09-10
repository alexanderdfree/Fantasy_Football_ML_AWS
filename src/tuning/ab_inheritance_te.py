"""A/B: does the static role-inheritance feature (validated on RB/WR, #1053) help TE?

Parity probe — TE is the left-behind skill position (RB/WR shipped inheritance via #1053;
``_INHERITANCE_POSITIONS=("RB","WR")`` excludes TE). A starter TE going Out/Doubtful frees
targets for the next TE up, so the same next-man-up signal *could* help TE. Validated here
BEFORE the expensive production merge (engineer.py → 6-position retrain + refresh-splits).

Only the **static** arm is tested: the ``attn_history`` inheritance token was tested-rejected
on RB (#1053, AGENTS.md stop-rule — a past spot-start is already encoded by the usage tokens),
so re-testing it on TE would re-propose a rejected mechanism. TE role proxy = **targets**
(a TE's value is targets, not snaps — same choice as WR).

* ``baseline``  — production config (the inheritance column is injected but NOT whitelisted,
                  so the model is unchanged and the inheritor slice is identical across arms).
* ``+static``   — ``is_top_available`` + ``inherited_opportunity`` whitelisted into
                  ``include_features`` (→ Ridge + LightGBM + NN-static branch).

Judge on the **inheritor subgroup** (``inherited_opportunity > 0``) bias, NOT overall MAE
(the feature fires on a handful of rows). **Watch ``inh_n``** — the TE inheritor cohort may be
small (TEs sit less often than RBs); if n is tiny, treat the result as inconclusive rather than
forcing a call. Leakage-safe: role(player,W) is the prior-to-W expanding mean (weeks < W only);
the Ridge sentinel checks the feature *took*, not that it's honest.

Run::

    python -m src.tuning.ab_inheritance_te                       # TE, 3 seeds
    python -m src.tuning.ab_inheritance_te --seeds 42 123 7 1 99 2024  # 6-seed confirm
    python -m src.tuning.ab_inheritance_te --list
"""

from __future__ import annotations

from src.tuning._cohort_metrics import inheritance_metrics
from src.tuning._inheritance import inject_inheritance
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["TE"]
SEEDS = [42, 123, 7]

# Inheritance computed WITHIN TE (rank same-position teammates; the splits are all-position).
_POSITIONS = ("TE",)
# TE opportunity proxy = per-game targets (mirrors WR; a TE's value is targets, not snaps).
_ROLE_COL = {"TE": "targets"}
_STATIC = ["is_top_available", "inherited_opportunity"]  # → Ridge + LGBM + NN-static


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean role-inheritance columns (within TE)
# --------------------------------------------------------------------------- #
def _inject_inheritance(train, val, test):
    return inject_inheritance(train, val, test, positions=_POSITIONS, role_columns=_ROLE_COL)


# --------------------------------------------------------------------------- #
# Config mutator
# --------------------------------------------------------------------------- #
def _whitelist_static(cfg):
    get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [*get_cols(), *_STATIC]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *_STATIC]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — per-model overall + inheritor-subgroup bias
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    return inheritance_metrics(result, position)


VARIANTS = [
    Variant(
        "baseline",
        frame_injector=_inject_inheritance,
        label="baseline (model unchanged; inh col carried for slicing)",
    ),
    Variant(
        "+static",
        cfg_mutator=_whitelist_static,
        frame_injector=_inject_inheritance,
        expect_ridge_identical=False,  # a real feature MUST move Ridge
        label="+inheritance static (Ridge+LGBM+NN-static)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
