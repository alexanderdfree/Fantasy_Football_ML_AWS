"""Fail-closed paired promotion decision; no training or cloud mutations."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

from src.analysis.repair_evidence import complete_sample_counts, execution_signature, sample_count

SEEDS = (42, 123, 7)
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")


def promotion_gate(records, *, candidate, affected):
    """Each changed model must pass in both years; unchanged models are controls."""
    reasons, comparisons, index = [], [], {}
    for field, length in (
        ("source_sha", 40),
        ("data_release", 64),
        ("frozen_candidate_sha256", 64),
    ):
        values = [record.get(field) for record in records]
        if not values or any(
            not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
            for value in values
        ):
            reasons.append(f"missing immutable {field}")
        elif len(set(values)) != 1:
            reasons.append(f"mixed {field} across the confirmation grid")
    for record in records:
        key = tuple(record.get(k) for k in ("position", "origin", "variant", "seed"))
        if key in index:
            reasons.append(f"duplicate cell: {key}")
        index[key] = record
    if not affected or any(
        p not in POSITIONS or not models or set(models) - set(FAMILIES)
        for p, models in affected.items()
    ):
        reasons.append("affected models must be explicitly classified")
    for position in POSITIONS:
        for origin in (2024, 2025):
            pairs = []
            for seed in SEEDS:
                base = index.get((position, origin, "baseline", seed))
                proposed = index.get((position, origin, candidate, seed))
                if base is None or proposed is None:
                    reasons.append(f"missing pair: {position}/{origin}/{seed}")
                    continue
                valid = True
                try:
                    if execution_signature(base) != execution_signature(proposed):
                        raise ValueError("unmatched execution settings")
                except ValueError as error:
                    reasons.append(f"{error}: {position}/{origin}/{seed}")
                    valid = False
                for key in (
                    "source_sha",
                    "data_release",
                    "gpu",
                    "prepared_hashes",
                    "row_hash",
                    "truth_hash",
                ):
                    if not base.get(key) or base.get(key) != proposed.get(key):
                        reasons.append(f"unmatched {key}: {position}/{origin}/{seed}")
                        valid = False
                for record in (base, proposed):
                    if not isinstance(record.get("metrics"), dict):
                        reasons.append(f"missing metric evidence: {position}/{origin}/{seed}")
                        valid = False
                    if (
                        not record.get("batch_job_id")
                        or not record.get("inference_parity_passed")
                        or not record.get("frozen_candidate_sha256")
                    ):
                        reasons.append(
                            f"missing execution/parity/freeze evidence: {position}/{origin}/{seed}"
                        )
                        valid = False
                    for cohort in ("elite_top24", "weekly_reference_top24"):
                        block = record.get("cohorts", {}).get(cohort, {})
                        if (
                            block.get("status") != "available"
                            or not block.get("n")
                            or not block.get("cohort_hash")
                        ):
                            reasons.append(
                                f"unavailable cohort {cohort}: {position}/{origin}/{seed}"
                            )
                            valid = False
                for cohort in ("elite_top24", "weekly_reference_top24"):
                    a, b = (r.get("cohorts", {}).get(cohort, {}) for r in (base, proposed))
                    if (a.get("cohort_hash"), a.get("n")) != (b.get("cohort_hash"), b.get("n")):
                        reasons.append(f"unmatched cohort {cohort}: {position}/{origin}/{seed}")
                        valid = False
                if valid:
                    pairs.append((base, proposed))
            if len(pairs) != len(SEEDS):
                continue
            for family in FAMILIES:
                changed = family in affected.get(position, ())
                for cohort in ("all", "elite_top24", "weekly_reference_top24"):
                    if not all(
                        complete_sample_counts(
                            a["metrics"].get(f"{cohort}:{family}", {}),
                            b["metrics"].get(f"{cohort}:{family}", {}),
                            sample_count(a["metrics"].get("all:ridge"))
                            if cohort == "all"
                            else a["cohorts"][cohort].get("n"),
                            sample_count(b["metrics"].get("all:ridge"))
                            if cohort == "all"
                            else b["cohorts"][cohort].get("n"),
                        )
                        for a, b in pairs
                    ):
                        reasons.append(
                            f"incomplete model samples: {position}/{origin}/{family}/{cohort}"
                        )
                        continue
                    for metric in ("mae", "rmse"):
                        try:
                            values = [
                                (
                                    a["metrics"][f"{cohort}:{family}"][metric],
                                    b["metrics"][f"{cohort}:{family}"][metric],
                                )
                                for a, b in pairs
                            ]
                            if not all(
                                math.isfinite(v) and v >= 0 for pair in values for v in pair
                            ):
                                raise ValueError("nonfinite metric")
                            delta = [b - a for a, b in values]
                        except (KeyError, TypeError, ValueError):
                            reasons.append(
                                f"missing finite {metric}: {position}/{origin}/{family}/{cohort}"
                            )
                            continue
                        passed = (
                            mean(delta) < 0
                            if changed and cohort == "all"
                            else mean(delta) <= 0
                            if changed
                            else all(abs(d) <= 1e-10 for d in delta)
                        )
                        comparisons.append(
                            dict(
                                position=position,
                                origin=origin,
                                family=family,
                                cohort=cohort,
                                metric=metric,
                                delta_mean=mean(delta),
                                delta_std=stdev(delta),
                                paired_deltas=delta,
                                passed=passed,
                            )
                        )
                        if not passed:
                            reasons.append(
                                f"metric gate failed: {position}/{origin}/{family}/{cohort}/{metric}"
                            )
    return {
        "schema": "model-default-repair-gate/v1",
        "candidate": candidate,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON array of verified cell manifests")
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--affected", type=Path, required=True, help="Position-to-changed-models JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = promotion_gate(
        json.loads(args.records.read_text()),
        candidate=args.candidate,
        affected=json.loads(args.affected.read_text()),
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "reasons": report["reasons"][:20],
                "report": str(args.output),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
