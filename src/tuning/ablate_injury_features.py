"""Ablation: do the injury / return features actually help, or are they
benchmark-flat? (operator-only CLI).

Companion to :mod:`src.analysis.injury_subgroup_error`. That script slices the
*current* model's error by injury/return subgroup; this one trains each position
twice — **with** and **without** the injury/return features — and reports the MAE
delta both overall and *within the returning subgroup*. The subgroup delta is the
point: per the draft-capital lesson (TODO.md ``[TESTED, REJECTED]``) a feature can
be flat in overall MAE yet move a ~15%-of-rows subgroup, so an overall-only
comparison can't settle "does it help".

Features removed (whichever are present for the position; RB carries no
``is_returning_from_absence``):
    game_status, practice_status, days_rest, is_returning_from_absence

Removal hits **both** model paths so the comparison is honest:
  * linear/tree — wrap ``get_feature_columns_fn`` so its column list excludes them
    (the pipeline slices ``X = df[cols]``);
  * attention NN — filter ``attn_static_features`` (the static-branch whitelist;
    these features sit in the ``contextual`` category, which feeds it).

Standing caveat: ``game_status`` is 96.8% constant in the 2025 test split (Out /
Doubtful self-eliminate — preprocessing drops no-play rows), so its *overall* MAE
delta is near-unmeasurable. The real signal is the ``returning`` subgroup and
``days_rest`` — read the per-subgroup deltas, not just GLOBAL.

Runs the full production config per variant (no skip-NN proxy — a reduced model
can flip the sign; see CLAUDE.md). Single seed, mirroring ablate_rb_gate.py.

Usage:
    python -m src.tuning.ablate_injury_features                          # QB RB WR TE
    python -m src.tuning.ablate_injury_features --positions QB RB        # subset
    python -m src.tuning.ablate_injury_features --seed 7 --no-history
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.analysis.analysis_rb_lgbm_disagreement import (  # noqa: E402  reuse pure helpers
    available_models,
    per_model_metrics,
)
from src.analysis.injury_subgroup_error import SMALL_N, SUBGROUP_SPECS  # noqa: E402
from src.shared.benchmark_utils import (  # noqa: E402
    append_to_history,
    get_git_hash,
    utc_now_iso,
)

ABLATION_NAME = "injury_features"
HISTORY_DIR = "benchmark_history"
# Default to the skill positions: largest returning subgroups and the designation
# features live in their `contextual` category. Extend to K/DST only if A shows
# an effect (per the plan).
DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]

# The injury/return features under test. Whichever are present for a given
# position are removed; absent names are no-ops.
INJURY_FEATURES = frozenset(
    {"game_status", "practice_status", "days_rest", "is_returning_from_absence"}
)

# Subgroups shown in the per-position delta table (subset of A's specs, in order).
_DELTA_SUBGROUPS = ["global", "returning", "ret_1wk", "ret_2wk", "questionable"]


def _drop_injury_features(cfg: dict) -> dict:
    """Deep-copy ``cfg`` and strip the injury/return features from both model
    paths. Module-level callables are atomic to ``deepcopy`` (the wrapped
    function still calls the original), matching the ablate_rb_gate.py pattern."""
    cfg = copy.deepcopy(cfg)
    orig_get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [c for c in orig_get_cols() if c not in INJURY_FEATURES]
    cfg["attn_static_features"] = [
        c for c in cfg["attn_static_features"] if c not in INJURY_FEATURES
    ]
    return cfg


def _dropped_summary(base_cfg: dict, drop_cfg: dict) -> dict:
    """Which features the ablation actually removed from each path. A 0-feature
    drop silently invalidates the comparison (the MAE Δ=0.0000 smell), so this is
    printed and checked before the (expensive) runs."""
    base_cols = set(base_cfg["get_feature_columns_fn"]())
    drop_cols = set(drop_cfg["get_feature_columns_fn"]())
    return {
        "linear_tree_dropped": sorted(base_cols - drop_cols),
        "attn_static_dropped": sorted(
            set(base_cfg["attn_static_features"]) - set(drop_cfg["attn_static_features"])
        ),
    }


def _subgroup_metrics(df, models: dict[str, str]) -> dict:
    """Per-subgroup per-model metrics on ``df``, reusing A's subgroup definitions
    so 'with' and 'without' are sliced identically."""
    out: dict[str, dict] = {}
    n_total = len(df)
    for key, label, needed, mask_fn in SUBGROUP_SPECS:
        if needed is not None and needed not in df.columns:
            continue
        sub = df[mask_fn(df)]
        out[key] = {
            "label": label.strip(),
            "n": len(sub),
            "pct": round(100 * len(sub) / n_total, 1) if n_total else 0.0,
            "models": per_model_metrics(sub, models),
        }
    return out


def run_variant(pos: str, label: str, cfg: dict, seed: int) -> dict:
    run = importlib.import_module(f"src.{pos.lower()}.run_pipeline").run
    print(f"\n{'=' * 72}\n{pos} variant: {label}\n{'=' * 72}")
    result = run(seed=seed, config=cfg)
    df = result["test_df"].copy()
    models = available_models(df)
    return {"label": label, "models": list(models), "subgroups": _subgroup_metrics(df, models)}


def ablate_position(pos: str, seed: int) -> dict:
    mod = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    base_cfg = mod.CONFIG
    drop_cfg = _drop_injury_features(base_cfg)  # pristine deep-copy, taken before any run

    dropped = _dropped_summary(base_cfg, drop_cfg)
    print(f"\n[{pos}] dropped from linear/tree : {dropped['linear_tree_dropped']}")
    print(f"[{pos}] dropped from attn static  : {dropped['attn_static_dropped']}")
    if not dropped["linear_tree_dropped"] and not dropped["attn_static_dropped"]:
        print(
            f"[{pos}] WARNING: no injury features present to drop — ablation is a NO-OP for {pos}."
        )

    with_inj = run_variant(pos, "with injury/return features (baseline)", base_cfg, seed)
    without_inj = run_variant(pos, "WITHOUT injury/return features", drop_cfg, seed)
    _print_ablation_table(pos, with_inj, without_inj)
    return {
        "position": pos,
        "seed": seed,
        "dropped": dropped,
        "with": with_inj,
        "without": without_inj,
    }


def _best_model(variant: dict) -> str:
    g = variant["subgroups"]["global"]["models"]
    return min(g, key=lambda k: g[k]["mae"])


def _print_ablation_table(pos: str, w: dict, wo: dict) -> None:
    best = _best_model(w)
    print(f"\n{'=' * 86}")
    print(f"{pos} injury/return ablation — Δ = without − with  (positive ⇒ features HELP)")
    print(f"best model (baseline GLOBAL MAE): {best}")
    print("=" * 86)

    # 1) All models, GLOBAL — is removal benchmark-flat across the board?
    print("\n  GLOBAL — every model:")
    print(f"    {'model':14}{'with':>9}{'without':>9}{'Δ':>9}")
    for name in w["subgroups"]["global"]["models"]:
        a = w["subgroups"]["global"]["models"][name]["mae"]
        b = wo["subgroups"]["global"]["models"].get(name, {}).get("mae", float("nan"))
        print(f"    {name:14}{a:9.3f}{b:9.3f}{b - a:+9.3f}")

    # 2) Best model across subgroups — where (if anywhere) does it move?
    print(f"\n  {best} — by subgroup:")
    print(f"    {'subgroup':28}{'n':>6}{'with':>9}{'without':>9}{'Δ':>9}")
    for key in _DELTA_SUBGROUPS:
        sw, swo = w["subgroups"].get(key), wo["subgroups"].get(key)
        if not sw or sw["n"] == 0:
            continue
        a = sw["models"][best]["mae"]
        b = swo["models"].get(best, {}).get("mae", float("nan"))
        flag = f"  [small-n<{SMALL_N}]" if sw["n"] < SMALL_N else ""
        print(f"    {sw['label'][:28]:28}{sw['n']:>6}{a:9.3f}{b:9.3f}{b - a:+9.3f}{flag}")


def _print_cross_summary(records: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print("CROSS-POSITION SUMMARY — Δ(without − with) FP MAE, best model")
    print("positive Δ ⇒ removing the features RAISED MAE ⇒ the features HELP that slice")
    print("=" * 78)
    print(f"{'Pos':<5}{'model':12}{'global Δ':>11}{'returning Δ':>14}{'n_ret':>7}")
    print("-" * 49)
    for r in records:
        w, wo = r["with"]["subgroups"], r["without"]["subgroups"]
        best = _best_model(r["with"])
        gd = (
            wo["global"]["models"].get(best, {}).get("mae", float("nan"))
            - w["global"]["models"][best]["mae"]
        )
        ret = w.get("returning")
        if ret and ret["n"] > 0:
            rd = (
                wo["returning"]["models"].get(best, {}).get("mae", float("nan"))
                - ret["models"][best]["mae"]
            )
            n_ret = ret["n"]
        else:
            rd, n_ret = float("nan"), 0
        print(f"{r['position']:<5}{best:12}{gd:+11.3f}{rd:+14.3f}{n_ret:>7}")


def _write_ablation(records: list[dict], seed: int) -> None:
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{ABLATION_NAME}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": ABLATION_NAME,
        "seed": seed,
        "results": records,
    }
    append_to_history(os.path.join(HISTORY_DIR, "ablations"), entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="skip writing results to benchmark_history/ablations/",
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]
    records = [ablate_position(p, args.seed) for p in positions]
    _print_cross_summary(records)
    if not args.no_history:
        _write_ablation(records, args.seed)


if __name__ == "__main__":
    main()
