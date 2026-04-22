"""DEPRECATED: references pre-migration targets (rushing_floor/receiving_floor/td_points); do not run against current data.

Sweep gated TD hyperparameters for RB attention model.
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from RB.run_rb_pipeline import RB_CONFIG
from shared.pipeline import run_pipeline


def run_variant(label, overrides, seed=42):
    cfg = copy.deepcopy(RB_CONFIG)
    cfg.update(overrides)
    t0 = time.time()
    results = run_pipeline("RB", cfg, seed=seed)
    elapsed = time.time() - t0

    attn = results.get("attn_nn_metrics", {})
    attn_rank = results.get("attn_nn_ranking", {})

    row = {
        "label": label,
        "overrides": {k: str(v) for k, v in overrides.items()},
        "attn_mae": attn.get("total", {}).get("mae"),
        "attn_r2": attn.get("total", {}).get("r2"),
        "attn_td_mae": attn.get("td_points", {}).get("mae"),
        "attn_rush_mae": attn.get("rushing_floor", {}).get("mae"),
        "attn_recv_mae": attn.get("receiving_floor", {}).get("mae"),
        "attn_top12": attn_rank.get("season_avg_hit_rate"),
        "elapsed": round(elapsed, 1),
    }
    return row


def print_table(all_results):
    print("\n" + "=" * 115)
    print(
        f"{'Label':<45} {'Total':>7} {'TD MAE':>7} {'Rush':>7} {'Recv':>7} {'R2':>6} {'Top12':>6} {'Time':>5}"
    )
    print("-" * 115)
    for r in all_results:
        print(
            f"{r['label']:<45} "
            f"{r['attn_mae'] or 0:>7.3f} "
            f"{r['attn_td_mae'] or 0:>7.3f} "
            f"{r['attn_rush_mae'] or 0:>7.3f} "
            f"{r['attn_recv_mae'] or 0:>7.3f} "
            f"{r['attn_r2'] or 0:>6.3f} "
            f"{r.get('attn_top12') or 0:>6.3f} "
            f"{r['elapsed']:>5.0f}s"
        )
    print("=" * 115)


def save_results(all_results, path="tune_rb_gate_results.json"):
    """Atomic write: dump to .tmp then rename so a crash can't leave the
    file half-written (which would crash the next step's json.load).
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    results_file = "tune_rb_gate_results.json"
    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"WARNING: {results_file} is corrupt ({e}); starting fresh.")

    if args.step == 0:
        row = run_variant("baseline (w=1.0, h=16, d=3.0)", {})
        all_results.append(row)

    elif args.step == 1:
        for w in [1.5, 2.0, 3.0]:
            row = run_variant(f"gate_weight={w}", {"attn_gate_weight": w})
            all_results.append(row)

    elif args.step == 2:
        step1 = [r for r in all_results if "gate_weight=" in r["label"]]
        baseline = [r for r in all_results if "baseline" in r["label"]]
        candidates = step1 + baseline
        best = min(candidates, key=lambda r: r["attn_td_mae"] or 99)
        best_w = (
            float(best["overrides"].get("attn_gate_weight", "1.0")) if best["overrides"] else 1.0
        )
        print(f"Best gate weight so far: {best_w} (TD MAE={best['attn_td_mae']})")
        for h in [24, 32]:
            row = run_variant(
                f"gate_hidden={h} (w={best_w})",
                {"attn_gate_weight": best_w, "attn_gate_hidden": h},
            )
            all_results.append(row)

    elif args.step == 3:
        all_candidates = [r for r in all_results]
        best = min(all_candidates, key=lambda r: r["attn_td_mae"] or 99)
        best_w = (
            float(best["overrides"].get("attn_gate_weight", "1.0")) if best["overrides"] else 1.0
        )
        best_h = (
            int(best["overrides"].get("attn_gate_hidden", "16"))
            if "attn_gate_hidden" in best.get("overrides", {})
            else 16
        )
        print(f"Best so far: w={best_w}, h={best_h} (TD MAE={best['attn_td_mae']})")
        for d in [2.0, 1.5]:
            deltas = {"rushing_floor": 2.0, "receiving_floor": 2.5, "td_points": d}
            row = run_variant(
                f"huber_delta={d} (w={best_w}, h={best_h})",
                {
                    "attn_gate_weight": best_w,
                    "attn_gate_hidden": best_h,
                    "huber_deltas": deltas,
                },
            )
            all_results.append(row)

    elif args.step == 4:
        all_candidates = [r for r in all_results]
        best = min(all_candidates, key=lambda r: r["attn_td_mae"] or 99)
        best_w = (
            float(best["overrides"].get("attn_gate_weight", "1.0")) if best["overrides"] else 1.0
        )
        best_h = (
            int(best["overrides"].get("attn_gate_hidden", "16"))
            if "attn_gate_hidden" in best.get("overrides", {})
            else 16
        )
        ovr_deltas = best["overrides"].get("huber_deltas", "")
        best_d = 3.0
        if ovr_deltas and "td_points" in ovr_deltas:
            import ast

            best_d = ast.literal_eval(ovr_deltas).get("td_points", 3.0)
        print(f"Best so far: w={best_w}, h={best_h}, d={best_d} (TD MAE={best['attn_td_mae']})")
        deltas = {"rushing_floor": 2.0, "receiving_floor": 2.5, "td_points": best_d}
        for td_lw in [2.5, 3.0]:
            lw = {"rushing_floor": 1.2, "receiving_floor": 1.0, "td_points": td_lw}
            row = run_variant(
                f"td_loss_w={td_lw} (gw={best_w}, h={best_h}, d={best_d})",
                {
                    "attn_gate_weight": best_w,
                    "attn_gate_hidden": best_h,
                    "huber_deltas": deltas,
                    "loss_weights": lw,
                },
            )
            all_results.append(row)

    save_results(all_results)
    print_table(all_results)
