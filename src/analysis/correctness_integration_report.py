"""Read-only correctness/compatibility collector; metric changes never veto it.

Reuse the existing hash-checked archive, independent raw-input truth builder,
checkpoint rescoring and paired-control verifier. No fitting or publication.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from statistics import mean, stdev

import numpy as np

from src.analysis import audit_development_report as audit

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
AFFECTED = ("RB", "WR", "TE")
ORIGINS, SEEDS = (2022, 2023), (42, 123, 7)
ARMS = ("baseline", "precision", "mean", "combined")
require = audit.require


def arms(position):
    return ARMS if position in AFFECTED else ("baseline", "combined")


def expected_cells():
    return {(p, y, s, a) for p in POSITIONS for y in ORIGINS for s in SEEDS for a in arms(p)}


def validate_plan(plan):
    require(plan.get("schema") == "correctness-integration-plan/v1", "Unknown plan schema")
    for name, pattern in (
        ("source_sha", r"[0-9a-f]{40}"),
        ("baseline_sha", r"[0-9a-f]{40}"),
        ("image_digest", r"sha256:[0-9a-f]{64}"),
        ("data_release", r"[0-9a-f]{64}"),
    ):
        require(re.fullmatch(pattern, plan.get(name, "")), f"Invalid {name}")
    require(
        set(plan.get("candidate_heads", {})) == {"1608", "1613"}, "Both PR source pins required"
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{40}", value) for value in plan["candidate_heads"].values()),
        "Invalid PR source pin",
    )
    expected, seen, prefixes = expected_cells(), [], []
    for run in plan["runs"]:
        prefix = run["prefix"].rstrip("/") + "/"
        require(
            prefix.startswith("ab_runs/")
            and all(p not in {"", ".", ".."} for p in prefix[:-1].split("/")),
            "Unsafe run prefix",
        )
        prefixes.append(prefix)
        for position in run["positions"]:
            for seed in run["seeds"]:
                seen.extend((position, run["origin"], seed, arm) for arm in arms(position))
    require(len(prefixes) == len(set(prefixes)), "Duplicate run prefix")
    require(
        len(seen) == len(set(seen)) and set(seen) == expected,
        "Incomplete or overlapping 108-cell plan",
    )
    return deepcopy(plan)


def verify_run(run, planned, plan):
    positions = set(planned["positions"])
    spec = "affected" if positions <= set(AFFECTED) else "qb" if positions == {"QB"} else "native"
    require(
        positions
        <= ({"K", "DST"} if spec == "native" else set(AFFECTED) if spec == "affected" else {"QB"}),
        "Run mixes provider families",
    )
    require(run["spec"] == f"src.tuning.ab_correctness_{spec}", "Wrong diagnostic spec")
    require(
        run["image_sha"] == plan["source_sha"] and run["image_digest"] == plan["image_digest"],
        "Mixed source/image pins",
    )
    require(
        run["positions"] == planned["positions"] and run["seeds"] == planned["seeds"],
        "Wrong run axes",
    )
    require(run["variants"] == list(arms(planned["positions"][0])), "Wrong run arms")
    require(
        run["extra_env"].get("FF_CORRECTNESS_ORIGIN") == str(planned["origin"]), "Wrong run origin"
    )
    require(
        run["extra_env"].get("FF_AMP_DTYPE") == "fp32" and run["cuda_graph"] == "auto",
        "Nonproduction execution settings",
    )


def verify_record(record, truth, read_receipt):
    require(
        record.get("schema") == "count-correctness-integration/v1",
        "Wrong correctness evidence schema",
    )
    require(record.get("candidate_prs") == [1608, 1613], "Wrong candidate PRs")
    require(
        record.get("probability_mean")
        == ("corrected" if record["variant"] in {"mean", "combined"} else "legacy"),
        "Wrong expectation arm",
    )
    if record["position"] in AFFECTED and record["variant"] in {"precision", "combined"}:
        trainers = [t for t in record["trainers"] if t["family"] == "attn_nn"]
        require(len(trainers) == 1, "Missing attention numerical observations")
        numerical = trainers[0].get("count_numerics", {})
        require(
            numerical.get("active_numerical_defect") is False,
            "Corrected loss retains an observed numerical defect",
        )
        with np.load(io.BytesIO(read_receipt(trainers[0]["validation_raw"]))) as values:
            require(
                {"prediction__receptions_value_mu", "prediction__receptions_value_log_alpha"}
                <= set(values.files),
                "Missing observed count parameters",
            )
    # This explicit in-memory schema adapter selects the shared structural
    # verifier, not its promotion policy. Original bytes/metadata remain intact.
    view = deepcopy(record)
    view.update(schema="isolated-audit-development/v1", promotion_eligible=False)
    verified = audit.verify_record(view, truth, read_receipt)
    verified.record = record
    return verified


def summarize(cells, *, errors=()):
    index, problems = {}, list(errors)
    for cell in cells:
        record = cell.record
        key = (record["position"], record["origin"], record["seed"], record["variant"])
        require(key not in index, f"Duplicate cell: {key}")
        index[key] = cell
    expected = expected_cells()
    if set(index) != expected:
        problems.append(
            f"Incomplete grid: missing={sorted(expected - set(index))}, extra={sorted(set(index) - expected)}"
        )
    comparisons = []
    for position in POSITIONS:
        for origin in ORIGINS:
            for arm in arms(position)[1:]:
                pairs = []
                for seed in SEEDS:
                    keys = [(position, origin, seed, variant) for variant in ("baseline", arm)]
                    if not all(key in index for key in keys):
                        continue
                    baseline, candidate = (index[key] for key in keys)
                    try:
                        # Affected arms may change attention only. The existing
                        # verifier compares all raw prediction columns exactly
                        # and unchanged neural checkpoint tensors bit-for-bit.
                        audit._paired(
                            baseline, candidate, "count_precision", repeat=position not in AFFECTED
                        )
                    except (ValueError, AssertionError) as error:
                        problems.append(f"{position}/{origin}/{seed}/{arm}: {error}")
                        continue
                    pairs.append((seed, baseline, candidate))
                row = {
                    "position": position,
                    "origin": origin,
                    "arm": arm,
                    "family": "attn_nn",
                    "seeds": [p[0] for p in pairs],
                    "deltas": {},
                }
                if len(pairs) == 3:
                    for cohort in ("all", "elite_top24"):
                        for metric in ("mae", "rmse"):
                            key = f"{cohort}:attn_nn"
                            before = [a.scores[key][metric] for _, a, _ in pairs]
                            after = [b.scores[key][metric] for _, _, b in pairs]
                            delta = [b - a for a, b in zip(before, after, strict=True)]
                            avg, sd = mean(delta), stdev(delta)
                            half = 4.302652729911275 * sd / np.sqrt(3)
                            row["deltas"][f"{cohort}:{metric}"] = {
                                "paired": delta,
                                "mean": avg,
                                "std": sd,
                                "baseline_mean": mean(before),
                                "candidate_mean": mean(after),
                                "ci95_seed_mean": [avg - half, avg + half],
                            }
                comparisons.append(row)
    return {
        "schema": "correctness-integration-report/v1",
        "verification_passed": not problems,
        "records": len(index),
        "expected_records": len(expected),
        "errors": problems,
        "metric_improvement_veto": False,
        "weekly_reference_top24": {
            str(year): "unavailable: genuine pre-2024 weekly-reference archive absent"
            for year in ORIGINS
        },
        "uncertainty": "Paired seed means; Student t 95% interval with df=2. Historical development, not prospective holdout evidence.",
        "comparisons": comparisons,
        "cells": [
            {
                key: cell.record[key]
                for key in (
                    "position",
                    "origin",
                    "seed",
                    "variant",
                    "source_sha",
                    "data_release",
                    "batch_job_id",
                )
            }
            for cell in cells
        ],
    }


def analyze(plan, archive):
    plan = validate_plan(plan)
    truth_plan = {**plan, "positions": list(POSITIONS), "origins": list(ORIGINS)}
    truths = audit.hydrate_truth(archive, truth_plan)
    cells, errors = [], []
    for planned in plan["runs"]:
        prefix = planned["prefix"].rstrip("/") + "/"
        try:
            run = json.loads(archive.read(prefix + "run.json"))
            verify_run(run, planned, plan)
            keys = archive.keys(prefix)
            wanted = {
                (p, s, a) for p in planned["positions"] for s in planned["seeds"] for a in arms(p)
            }
            seen = set()
            for key in (
                key for key in keys if "/evidence/" in key and key.endswith("-manifest.json")
            ):
                digest = PurePosixPath(key).name.split("-")[0]
                require(re.fullmatch(r"[0-9a-f]{64}", digest), "Metadata is not content addressed")
                record = json.loads(archive.read(key, digest=digest))
                identity = (record["position"], record["seed"], record["variant"])
                require(
                    identity in wanted and identity not in seen, "Unexpected or duplicate manifest"
                )
                seen.add(identity)
                for name in ("source_sha", "baseline_sha", "data_release"):
                    require(record[name] == plan[name], f"Wrong {name}")
                require(
                    record["origin"] == planned["origin"]
                    and record["batch_job_id"] == run["jobs"][record["position"]],
                    "Wrong origin/job",
                )
                small = json.loads(
                    archive.read(
                        prefix
                        + f"cells/{record['position']}-{record['variant']}-{record['seed']}.json"
                    )
                )
                require(
                    all(small[name] == record[name] for name in ("position", "variant", "seed")),
                    "Wrong cell summary identity",
                )
                metrics = dict(small["metrics"])
                require(
                    metrics.pop("repair", None)
                    == {
                        "origin": record["origin"],
                        "trainer_count": 2,
                        "weekly_reference_available": 0,
                    },
                    "Wrong cell summary",
                )
                require(
                    small["ok"] is True and metrics == record["metrics"],
                    "Failed cell or inconsistent scores",
                )
                cells.append(
                    verify_record(
                        record,
                        truths[record["position"]],
                        lambda receipt, prefix=prefix: archive.receipt(receipt, prefix),
                    )
                )
            require(seen == wanted, "Missing completed cells")
            actual_keys = {
                PurePosixPath(key).name
                for key in keys
                if "/cells/" in key and key.endswith(".json")
            }
            require(
                actual_keys == {f"{p}-{a}-{s}.json" for p, s, a in wanted},
                "Extra or missing cell objects",
            )
        except (KeyError, ValueError, AssertionError, FileNotFoundError) as error:
            errors.append(f"{prefix}: {error}")
    report = summarize(cells, errors=errors)
    report.update(plan=plan, object_proofs=archive.proofs)
    return report


def markdown(report):
    lines = [
        "# Count correctness integration",
        "",
        f"Verification: **{'PASS' if report['verification_passed'] else 'FAIL'}**; {report['records']}/{report['expected_records']} cells.",
        "",
        "Metric improvements are not an acceptance condition. Weekly-reference cohorts are unavailable for both 2022 and 2023; elite cohorts use prior-season importance.",
        "",
        report["uncertainty"],
        "",
        "| Position | Year | Arm | Cohort/metric | Baseline | Candidate | Paired delta | SD | 95% CI |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["comparisons"]:
        for metric, value in row["deltas"].items():
            ci = value["ci95_seed_mean"]
            lines.append(
                f"| {row['position']} | {row['origin']} | {row['arm']} | {metric} | {value['baseline_mean']:.8g} | {value['candidate_mean']:.8g} | {value['mean']:.8g} | {value['std']:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] |"
            )
    if report["errors"]:
        lines += ["", "Verification errors:", "", *[f"- {error}" for error in report["errors"]]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(
        json.loads(args.plan.read_text()),
        audit.Archive(args.cache, args.bucket, offline=args.offline),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output / "report.md").write_text(markdown(report))
    print(
        json.dumps(
            {key: report[key] for key in ("verification_passed", "records", "expected_records")}
        )
    )
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
