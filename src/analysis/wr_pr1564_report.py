"""Verify and summarize the PR1564 WR factorial without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS = ("source", "availability", "depth")
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")
SEEDS = (42, 123, 7)
KEYS = ["player_id", "season", "week"]


def metric(frame, family, kind):
    error = frame[f"pred_{family}_total"].to_numpy() - frame.actual.to_numpy()
    return float(
        {
            "mae": lambda: np.abs(error).mean(),
            "rmse": lambda: np.sqrt(np.mean(error**2)),
            "bias": lambda: error.mean(),
        }[kind]()
    )


def arm_name(bits):
    return f"r{bits[0]}a{bits[1]}d{bits[2]}"


def moments(values):
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "seeds": dict(zip(map(str, SEEDS), map(float, values), strict=True)),
    }


def shapley(values):
    """Exact three-factor allocation, averaging every intervention order."""
    result = {}
    for index, name in enumerate(FACTORS):
        effect = 0.0
        others = [other for other in range(3) if other != index]
        for selection in itertools.product((0, 1), repeat=2):
            before = [0, 0, 0]
            for other, bit in zip(others, selection, strict=True):
                before[other] = bit
            after = before.copy()
            after[index] = 1
            size = sum(selection)
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            effect += weight * (values[tuple(after)] - values[tuple(before)])
        result[name] = effect
    if not np.isclose(sum(result.values()), values[(1, 1, 1)] - values[(0, 0, 0)], atol=1e-12):
        raise ValueError("Attribution does not reconcile to the complete change")
    return result


def load(root):
    frames, records = {}, []
    added_reference = None
    for path in sorted(root.glob("evidence/*.json")):
        record = json.loads(path.read_text())
        if record["seed"] not in SEEDS or record["arm"] not in {"r0a0", "r0a1", "r1a0", "r1a1"}:
            raise ValueError("Unexpected fitted cell")
        if record["amp"] is not False or record["cuda_graph"] not in {"0", "false"}:
            raise ValueError("Unexpected execution regime")
        if (
            record.get("tf32") is not True
            or record.get("scoring") != "ppr_shared_projected_components"
        ):
            raise ValueError("Unexpected precision or scoring contract")
        if len(record["parity"]) != 20 or any(
            not np.isfinite(x) or x > 0.001 for x in record["parity"].values()
        ):
            raise ValueError("Missing or invalid saved-inference parity")
        for name, replay in record["replays"].items():
            for cohort, count in (("weekly_reference_top24", 432), ("elite_top24", 325)):
                block = replay.get("cohorts", {}).get(cohort, {})
                if block.get("status") != "available" or block.get("n") != count:
                    raise ValueError(f"Incomplete frozen cohort: {cohort}")
            metadata = replay["rows"]
            content = (root / "rows" / Path(metadata["key"]).name).read_bytes()
            if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
                raise ValueError("Row artifact hash mismatch")
            frame = pd.read_parquet(root / "rows" / Path(metadata["key"]).name)
            if frame.duplicated(KEYS).any() or not np.isfinite(frame.actual).all():
                raise ValueError("Invalid row keys or corrected actuals")
            columns = [
                f"pred_{family}_{target}"
                for family in FAMILIES
                for target in (
                    "total",
                    "receptions",
                    "receiving_yards",
                    "receiving_tds",
                    "fumbles_lost",
                )
            ]
            if not set(columns).issubset(frame) or not np.isfinite(frame[columns].to_numpy()).all():
                raise ValueError("Missing or non-finite model predictions")
            added = frame.loc[~frame.common].sort_values(KEYS).reset_index(drop=True)
            expected_added = 7 if name.startswith("r1") else 0
            if len(added) != expected_added:
                raise ValueError("Unexpected added-observation cohort")
            if expected_added:
                identity = added[[*KEYS, "actual"]]
                if added_reference is None:
                    added_reference = identity
                else:
                    pd.testing.assert_frame_equal(added_reference, identity, check_dtype=False)
            frame = frame.loc[frame.common].sort_values(KEYS).reset_index(drop=True)
            if len(frame) != 2761:
                raise ValueError("Incomplete common cohort")
            if (
                int(frame.weekly_reference_top24.sum()) != 432
                or int(frame.elite_top24.sum()) != 325
            ):
                raise ValueError("Frozen row-level cohort masks disagree with evidence")
            key = (name, record["seed"])
            if key in frames:
                raise ValueError("Duplicate arm/seed evidence")
            frames[key] = frame
        records.append(record)
    expected = {
        (arm_name(bits), seed) for bits in itertools.product((0, 1), repeat=3) for seed in SEEDS
    }
    if set(frames) != expected or len(records) != 12:
        raise ValueError(f"Incomplete factorial: {len(records)} fits, {len(frames)} replays")
    if (
        len({record["source"] for record in records}) != 1
        or len({record["input_archive_sha256"] for record in records}) != 1
    ):
        raise ValueError("Mixed source/input provenance")
    for seed in SEEDS:
        matching = [record for record in records if record["seed"] == seed]
        if len({(r["gpu"], r["tf32"], r["cuda_graph"]) for r in matching}) != 1:
            raise ValueError("A matched seed spans different hardware/settings")
    reference = next(iter(frames.values()))
    for frame in frames.values():
        for cols in (
            KEYS,
            ["actual"],
            [
                "weekly_reference_top24",
                "elite_top24",
                "depth_old",
                "depth_new",
                "availability_old",
                "availability_new",
            ],
        ):
            pd.testing.assert_frame_equal(reference[cols], frame[cols], check_dtype=False)
    return frames, records


def summarize(root):
    frames, records = load(root)
    output = {
        "retrospective": True,
        "test_season": 2025,
        "n_common": 2761,
        "fit_cells": 12,
        "scored_cells": 24,
        "source": records[0]["source"],
        "input_archive_sha256": records[0]["input_archive_sha256"],
        "models": {},
        "mechanism": {},
        "max_saved_inference_difference": max(x for r in records for x in r["parity"].values()),
    }
    display_names = {"ridge": "Ridge", "nn": "NN", "attn_nn": "Attention NN", "lgbm": "LightGBM"}
    output["added_observations"] = {
        "n": 7,
        "basis": "Descriptive only: baseline lacks these player-weeks; excluded from paired attribution",
        "arms": {},
    }
    for name in ("r1a0d0", "r1a0d1", "r1a1d0", "r1a1d1"):
        selected = {
            record["seed"]: record["replays"][name]["metrics"]["added"]
            for record in records
            if name in record["replays"]
        }
        if set(selected) != set(SEEDS) or any(block["n"] != 7 for block in selected.values()):
            raise ValueError("Incomplete added-observation metrics")
        output["added_observations"]["arms"][name] = {
            family: {
                kind: moments([selected[seed]["models"][display][kind] for seed in SEEDS])
                for kind in ("mae", "rmse", "bias")
            }
            for family, display in display_names.items()
        }
    for family in FAMILIES:
        result = {
            "baseline": {},
            "fixed": {},
            "delta": {},
            "shapley": {},
            "depth_conditional": {},
            "interactions": {},
            "subgroups": {},
            "raw_heads": {},
        }
        for kind in ("mae", "rmse", "bias"):
            cubes = [
                {
                    bits: metric(frames[(arm_name(bits), seed)], family, kind)
                    for bits in itertools.product((0, 1), repeat=3)
                }
                for seed in SEEDS
            ]
            result["baseline"][kind] = moments([v[(0, 0, 0)] for v in cubes])
            result["fixed"][kind] = moments([v[(1, 1, 1)] for v in cubes])
            result["delta"][kind] = moments([v[(1, 1, 1)] - v[(0, 0, 0)] for v in cubes])
            allocated = [shapley(v) for v in cubes]
            result["shapley"][kind] = {
                name: moments([v[name] for v in allocated]) for name in FACTORS
            }
            result["depth_conditional"][kind] = {
                f"r{r}a{a}": moments([v[(r, a, 1)] - v[(r, a, 0)] for v in cubes])
                for r in (0, 1)
                for a in (0, 1)
            }
            result["interactions"][kind] = {
                "source_availability": moments(
                    [v[(1, 1, 0)] - v[(1, 0, 0)] - v[(0, 1, 0)] + v[(0, 0, 0)] for v in cubes]
                ),
                "source_depth": moments(
                    [v[(1, 0, 1)] - v[(1, 0, 0)] - v[(0, 0, 1)] + v[(0, 0, 0)] for v in cubes]
                ),
                "availability_depth": moments(
                    [v[(0, 1, 1)] - v[(0, 1, 0)] - v[(0, 0, 1)] + v[(0, 0, 0)] for v in cubes]
                ),
                "three_way": moments(
                    [
                        v[(1, 1, 1)]
                        - v[(1, 1, 0)]
                        - v[(1, 0, 1)]
                        - v[(0, 1, 1)]
                        + v[(1, 0, 0)]
                        + v[(0, 1, 0)]
                        + v[(0, 0, 1)]
                        - v[(0, 0, 0)]
                        for v in cubes
                    ]
                ),
            }
        for target, weight in {
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        }.items():
            changes = {"prediction_shift": [], "point_shift": [], "mae_delta": [], "rmse_delta": []}
            for seed in SEEDS:
                before = frames[("r0a0d0", seed)]
                after = frames[("r1a1d1", seed)]
                column = f"pred_{family}_{target}"
                shift = float((after[column] - before[column]).mean())
                first_error = before[column] - before[target]
                last_error = after[column] - after[target]
                changes["prediction_shift"].append(shift)
                changes["point_shift"].append(weight * shift)
                changes["mae_delta"].append(
                    float(last_error.abs().mean() - first_error.abs().mean())
                )
                changes["rmse_delta"].append(
                    float(np.sqrt((last_error**2).mean()) - np.sqrt((first_error**2).mean()))
                )
            result["raw_heads"][target] = {key: moments(values) for key, values in changes.items()}
        for label in (
            "depth_changed",
            "depth_unchanged",
            "availability_changed",
            "week1",
            "weekly_reference_top24",
            "elite_top24",
        ):
            first = frames[("r0a0d0", 42)]
            if label.startswith("depth_"):
                mask = first.depth_old.ne(first.depth_new)
                if label == "depth_unchanged":
                    mask = ~mask
            elif label == "availability_changed":
                mask = first.availability_old.ne(first.availability_new)
            elif label == "week1":
                mask = first.week.eq(1)
            else:
                mask = first[label]
            result["subgroups"][label] = {
                "n": int(mask.sum()),
                **{
                    kind: moments(
                        [
                            metric(frames[("r1a1d1", seed)].loc[mask], family, kind)
                            - metric(frames[("r0a0d0", seed)].loc[mask], family, kind)
                            for seed in SEEDS
                        ]
                    )
                    for kind in ("mae", "rmse", "bias")
                },
            }
        output["models"][family] = result
        bins = []
        first = frames[("r1a1d0", 42)]
        for lower, upper in ((-float("inf"), 2), (2, 5), (5, 10), (10, 20), (20, float("inf"))):
            mask = (
                first.actual.ge(lower)
                & first.actual.lt(upper)
                & first.depth_old.ne(first.depth_new)
            )
            shifts = []
            per_target = {
                target: []
                for target in ("receptions", "receiving_yards", "receiving_tds", "fumbles_lost")
            }
            mae, mse = [], []
            for seed in SEEDS:
                before = frames[("r1a1d0", seed)].loc[mask]
                after = frames[("r1a1d1", seed)].loc[mask]
                shifts.append(
                    float((after[f"pred_{family}_total"] - before[f"pred_{family}_total"]).mean())
                )
                for target in per_target:
                    per_target[target].append(
                        float(
                            (
                                after[f"pred_{family}_{target}"] - before[f"pred_{family}_{target}"]
                            ).mean()
                        )
                    )
                mae.append(metric(after, family, "mae") - metric(before, family, "mae"))
                mse.append(metric(after, family, "rmse") ** 2 - metric(before, family, "rmse") ** 2)
            bins.append(
                {
                    "actual_bin": f"[{lower},{upper})",
                    "n": int(mask.sum()),
                    "prediction_shift": moments(shifts),
                    "mae_delta": moments(mae),
                    "mse_delta": moments(mse),
                    "raw_target_shift": {t: moments(v) for t, v in per_target.items()},
                }
            )
        output["mechanism"][family] = {"fixed_model_depth_replay_bins": bins}
    first = frames[("r1a1d0", 42)]
    second = frames[("r1a1d1", 42)]
    output["depth_clipping"] = {
        column: {
            "before_fraction_abs_gt4": float(first[column].abs().gt(4).mean()),
            "after_fraction_abs_gt4": float(second[column].abs().gt(4).mean()),
        }
        for column in first
        if column.endswith("_depth_z")
    }
    return output


def markdown(report):
    lines = [
        "# WR PR1564 causal investigation",
        "",
        "Retrospective 2025 comparison: 2,761 identical player-weeks, corrected shared-component actuals, three seeds, four full-production model families. Twelve fitted cells produce 24 scored intervention cells.",
        "",
        "## Complete correction",
        "",
        "| Model | Baseline MAE | Fixed MAE | Delta MAE | Delta RMSE | Delta bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, result in report["models"].items():
        lines.append(
            f"| {family} | {result['baseline']['mae']['mean']:.4f} | {result['fixed']['mae']['mean']:.4f} | {result['delta']['mae']['mean']:+.4f} ± {result['delta']['mae']['sd']:.4f} | {result['delta']['rmse']['mean']:+.4f} ± {result['delta']['rmse']['sd']:.4f} | {result['delta']['bias']['mean']:+.4f} |"
        )
    lines += [
        "",
        "## Attribution of MAE change",
        "",
        "Exact Shapley allocation averages all six factor orders, allocating interactions rather than ignoring them. Values sum to each model's complete MAE change.",
        "",
        "| Model | Depth normalization | Availability semantics | Other source/identity changes |",
        "|---|---:|---:|---:|",
    ]
    for family, result in report["models"].items():
        s = result["shapley"]["mae"]
        lines.append(
            f"| {family} | {s['depth']['mean']:+.4f} | {s['availability']['mean']:+.4f} | {s['source']['mean']:+.4f} |"
        )
    lines += [
        "",
        "## Fixed-model mechanism",
        "",
        "Depth changes below rescore the fully corrected fitted models; weights and fitted preprocessing stay fixed. Actual-output bins describe outcomes retrospectively and are not model-selected evaluation pools.",
    ]
    for family, values in report["mechanism"].items():
        lines += [
            "",
            f"### {family}",
            "",
            "| Actual points | n | Forecast shift | Delta MAE | Delta MSE |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in values["fixed_model_depth_replay_bins"]:
            lines.append(
                f"| {row['actual_bin']} | {row['n']} | {row['prediction_shift']['mean']:+.4f} | {row['mae_delta']['mean']:+.4f} | {row['mse_delta']['mean']:+.4f} |"
            )
    lines += [
        "",
        "## Seven added player-weeks",
        "",
        "These rows are absent from the baseline and excluded from every paired delta. The table describes the fully corrected arm only; the JSON retains all four source-corrected arms.",
        "",
        "| Model | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|",
    ]
    for family, metrics in report["added_observations"]["arms"]["r1a1d1"].items():
        lines.append(
            f"| {family} | {metrics['mae']['mean']:.4f} | {metrics['rmse']['mean']:.4f} | {metrics['bias']['mean']:+.4f} |"
        )
    lines += [
        "",
        "## Provenance and limits",
        "",
        f"Worker source: `{report['source']}`. Input archive SHA-256: `{report['input_archive_sha256']}`. Maximum saved-inference discrepancy: `{report['max_saved_inference_difference']}`.",
        "",
        "The inputs reconstruct archived production generations. They are not claimed to be the deleted local CPU experiment artifacts. All new comparisons use one pinned CUDA FP32 eager regime; seed standard deviations do not measure uncertainty across NFL seasons. No model changes, promotion, deployment, or retuning are implied.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (args.output / "results.md").write_text(markdown(report))
    print(json.dumps({family: result["delta"] for family, result in report["models"].items()}))


if __name__ == "__main__":
    main()
