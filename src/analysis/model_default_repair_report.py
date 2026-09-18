"""Reproduce paired development decisions from immutable repair manifests.

This module reads metrics and provenance only. It never fits a model, fetches
training data, or promotes a policy. Confirmation is a separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from src.prediction.bundle import canonical_json

SEEDS = (42, 123, 7)
ORIGINS = (2022, 2023)
VARIANTS = {
    "weights": (
        "baseline",
        "corrected",
        "gate_half",
        "gate_double",
        "reception_half",
        "reception_double",
    ),
    "numerical": ("baseline", "corrected", "stable_numeric"),
    "selector": ("baseline",),
}


def summarize(records, kind):
    positions = ("QB", "WR") if kind == "selector" else ("WR",)
    families = ("attn_nn",) if kind == "selector" else ("nn", "attn_nn")
    variants = VARIANTS[kind]
    index, errors, findings = {}, [], []
    for row in records:
        key = (row["position"], row["origin"], row["seed"], row["variant"])
        if key in index:
            raise ValueError(f"Duplicate immutable evidence for {key}; resolve retry provenance")
        index[key] = row
        if not row.get("inference_parity_passed") or not row.get("batch_job_id"):
            errors.append(f"Missing parity/Batch evidence: {key}")
        for trainer in row["trainers"]:
            numerical = trainer.get("count_numerics", {})
            if numerical.get("active_numerical_defect"):
                findings.append({"cell": list(key), "numerics": numerical})
            if kind == "selector" and trainer["family"] == "attn_nn":
                policy = trainer["policies"]
                if policy.get("guarded_search_epochs") != len(policy["trajectory"]):
                    errors.append(f"Superseded guarded-checkpoint search: {key}")
    expected = {(p, y, s, v) for p in positions for y in ORIGINS for s in SEEDS for v in variants}
    if set(index) != expected:
        errors.append(
            f"Incomplete grid: missing={sorted(expected - set(index))}, extra={sorted(set(index) - expected)}"
        )
    for field in ("source_sha", "data_release"):
        if len({r[field] for r in records}) != 1:
            errors.append(f"Mixed {field} values")
    candidates = ("rmse", "guarded") if kind == "selector" else variants[1:]
    comparisons = []
    for candidate in candidates:
        for position in positions:
            for origin in ORIGINS:
                for family in families:
                    pairs, missing = [], []
                    for seed in SEEDS:
                        base = index.get((position, origin, seed, "baseline"))
                        proposed = (
                            base
                            if kind == "selector"
                            else index.get((position, origin, seed, candidate))
                        )
                        if base is None or proposed is None:
                            missing.append(seed)
                            continue
                        if kind == "selector":
                            policies = base["policy_candidates"]
                            if candidate not in policies:
                                missing.append(seed)
                                continue
                            left, right = policies["legacy"], policies[candidate]
                        else:
                            for field in ("gpu", "prepared_hashes", "row_hash", "truth_hash"):
                                if base[field] != proposed[field]:
                                    errors.append(f"Unmatched {field}: {origin}/{seed}/{candidate}")
                            for control in ("ridge", "lgbm"):
                                for cohort in ("all", "elite_top24"):
                                    for metric in ("mae", "rmse"):
                                        key = f"{cohort}:{control}"
                                        if (
                                            abs(
                                                base["metrics"][key][metric]
                                                - proposed["metrics"][key][metric]
                                            )
                                            > 1e-10
                                        ):
                                            errors.append(
                                                f"Changed control: {origin}/{seed}/{candidate}/{key}/{metric}"
                                            )
                            left = {**base["metrics"][f"all:{family}"], "cohorts": base["cohorts"]}
                            right = {
                                **proposed["metrics"][f"all:{family}"],
                                "cohorts": proposed["cohorts"],
                            }
                        a, b = left["cohorts"]["elite_top24"], right["cohorts"]["elite_top24"]
                        if (
                            a["status"] != "available"
                            or b["status"] != "available"
                            or (a["cohort_hash"], a["n"]) != (b["cohort_hash"], b["n"])
                        ):
                            errors.append(
                                f"Unmatched important-player cohort: {position}/{origin}/{seed}"
                            )
                        if left["n"] != right["n"]:
                            errors.append(f"Unmatched overall rows: {position}/{origin}/{seed}")
                        pairs.append((left, right))
                    row = {
                        "candidate": candidate,
                        "position": position,
                        "origin": origin,
                        "family": family,
                        "missing_qualifying_seeds": missing,
                        "deltas": {},
                        "passes": False,
                    }
                    if not missing:
                        for cohort in ("all", "elite_top24"):
                            for metric in ("mae", "rmse"):
                                values = []
                                for a, b in pairs:
                                    if cohort != "all":
                                        a = a["cohorts"][cohort]["models"][family]
                                        b = b["cohorts"][cohort]["models"][family]
                                    values.append(b[metric] - a[metric])
                                row["deltas"][f"{cohort}:{metric}"] = {
                                    "mean": mean(values),
                                    "std": stdev(values),
                                    "paired": values,
                                }
                        row["passes"] = all(
                            v["mean"] < 0 if k.startswith("all:") else v["mean"] <= 0
                            for k, v in row["deltas"].items()
                        )
                    comparisons.append(row)
    qualified = []
    for candidate in candidates:
        clean_numerics = kind == "selector" or not any(f["cell"][3] == candidate for f in findings)
        if (
            not errors
            and clean_numerics
            and all(r["passes"] for r in comparisons if r["candidate"] == candidate)
        ):
            qualified.append(candidate)
    return {
        "kind": kind,
        "origins": list(ORIGINS),
        "seeds": list(SEEDS),
        "records": len(records),
        "source_shas": sorted({r["source_sha"] for r in records}),
        "data_releases": sorted({r["data_release"] for r in records}),
        "hardware": sorted({r["gpu"] for r in records}),
        "provenance_errors": sorted(set(errors)),
        "numerical_findings": findings,
        "comparisons": comparisons,
        "development_qualified": qualified,
        "merge_authorized": False,
        "limitation": "Development evidence only; confirmation and current review/CI remain mandatory.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=VARIANTS, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, proofs = [], []
    for path in sorted(args.input_dir.glob(args.pattern)):
        row = json.loads(path.read_text())
        digest = hashlib.sha256(canonical_json(row).encode()).hexdigest()
        if not path.name.endswith(f"{digest}-manifest.json"):
            raise ValueError(f"Manifest content-address mismatch: {path}")
        records.append(row)
        proofs.append({"file": path.name, "sha256": digest, "batch_job_id": row["batch_job_id"]})
    report = summarize(records, args.kind)
    report["manifest_proofs"] = proofs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "records": report["records"],
                "provenance_errors": report["provenance_errors"],
                "development_qualified": report["development_qualified"],
                "output": str(args.output),
            }
        )
    )
    return int(bool(report["provenance_errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
