"""Verify and summarize isolated AWS development cells without fitting models.

An explicit run plan separates real smoke evidence from the complete three-seed
development matrix. All source objects are read-only; downloads and reports live
in the caller's private cache. A passing development screen never authorizes a
default change or substitutes for the separate confirmation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean, stdev

import numpy as np
import pandas as pd

from src.analysis.repair_evidence import execution_signature
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.comparison_scoring import (
    comparison_actuals,
    comparison_model_totals,
    scoring_components,
)
from src.shared.evaluation_cohorts import _identity, ranked_rows, regular_season_rows

KEYS = ["player_id", "season", "week"]
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")
POSITIONS = {
    "count_precision": ("RB", "WR", "TE"),
    "bagging": ("QB", "RB", "WR", "TE", "K", "DST"),
    "stint_reset": ("WR", "TE"),
}
AFFECTED = {
    "count_precision": ("attn_nn",),
    "bagging": ("lgbm",),
    "stint_reset": ("ridge", "nn", "lgbm"),
}
SPECS = {
    "count_precision": ("src.tuning.ab_audit_count_development",),
    "bagging": (
        "src.tuning.ab_audit_bagging_development",
        "src.tuning.ab_audit_native_bagging_development",
    ),
    "stint_reset": ("src.tuning.ab_audit_stint_development",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(plan):
    plan = deepcopy(plan)
    require(
        plan.get("schema") in {"audit-development-plan/v1", "audit-development-plan/v2"},
        "Unknown plan schema",
    )
    if plan["schema"] == "audit-development-plan/v1":
        source = plan.pop("source_sha")
        plan["sources"] = {source: {"image_digest": plan.pop("image_digest")}}
        for run in plan["runs"]:
            run["source_sha"] = source
        plan["schema"] = "audit-development-plan/v2"
    candidate = plan.get("candidate")
    require(candidate in POSITIONS, "Unknown candidate")
    require(
        plan.get("purpose") in {"smoke", "development"},
        "Explicit smoke/development purpose required",
    )
    for field, length in (
        ("baseline_sha", 40),
        ("candidate_sha", 40),
        ("data_release", 64),
    ):
        require(re.fullmatch(rf"[0-9a-f]{{{length}}}", plan.get(field, "")), f"Invalid {field}")
    require(
        isinstance(plan.get("sources"), dict) and plan["sources"],
        "Explicit source/image pins required",
    )
    for source, pin in plan["sources"].items():
        require(re.fullmatch(r"[0-9a-f]{40}", source), "Invalid source SHA")
        require(
            re.fullmatch(r"sha256:[0-9a-f]{64}", pin.get("image_digest", "")),
            "Invalid image digest",
        )
    if len(plan["sources"]) > 1:
        require(
            sum("bridge" not in pin for pin in plan["sources"].values()) == 1,
            "Mixed sources require an explicit observer-only bridge",
        )
    require(set(plan["positions"]) <= set(POSITIONS[candidate]), "Unexpected position")
    for name in ("positions", "origins", "seeds"):
        require(plan[name] and len(plan[name]) == len(set(plan[name])), f"Invalid {name} axis")
    require(set(plan["origins"]) <= {2022, 2023}, "Development cannot inspect confirmation")
    require(set(plan["seeds"]) <= {42, 123, 7}, "Unexpected seed")
    if plan["purpose"] == "development":
        require(
            set(plan["positions"]) == set(POSITIONS[candidate]),
            "Incomplete affected-position matrix",
        )
        require(set(plan["origins"]) == {2022, 2023}, "Both development years required")
        require(set(plan["seeds"]) == {42, 123, 7}, "All three paired seeds required")
    else:
        require(
            len(plan["seeds"]) == len(plan["origins"]) == 1, "Smoke must be one seed and origin"
        )
    expected = {
        (p, y, s) for p in plan["positions"] for y in plan["origins"] for s in plan["seeds"]
    }
    coverage = []
    prefixes = []
    position_sources = {}
    campaign = plan["campaign_prefix"].rstrip("/") + "/"
    require(campaign.startswith("ab_runs/"), "Isolated ab_runs campaign required")
    for run in plan["runs"]:
        require(run["source_sha"] in plan["sources"], "Run has undeclared source pin")
        included = run.setdefault("include_positions", list(run["positions"]))
        superseded = run.setdefault("superseded_positions", {})
        require(
            set(included).isdisjoint(superseded)
            and set(included) | set(superseded) == set(run["positions"]),
            "Unexplained excluded run positions",
        )
        require(
            all(
                p == "TE" and reason == "observer_position_identity"
                for p, reason in superseded.items()
            ),
            "Unsupported supersession reason",
        )
        require(all(seed in plan["seeds"] for seed in run["seeds"]), "Unexpected run seed")
        prefix = run["prefix"].rstrip("/") + "/"
        require(prefix.startswith(campaign) and prefix != campaign, "Mixed campaigns")
        prefixes.append(prefix)
        coverage.extend((p, run["origin"], s) for p in included for s in run["seeds"])
        for position in included:
            position_sources.setdefault(position, set()).add(run["source_sha"])
    require(len(prefixes) == len(set(prefixes)), "Duplicate run prefixes")
    require(
        len(coverage) == len(set(coverage)) and set(coverage) == expected,
        "Incomplete or overlapping run plan",
    )
    require(
        all(len(pins) == 1 for pins in position_sources.values()),
        "A position mixes source pins across its paired seeds/seasons",
    )
    require(
        {run["source_sha"] for run in plan["runs"]} == set(plan["sources"]),
        "Unused source declaration",
    )
    return plan


class Archive:
    """Read-only S3 archive with hash-checked local copies; no publication API."""

    def __init__(self, root, bucket, *, offline=False, client=None):
        self.root, self.bucket, self.offline = Path(root), bucket, offline
        self.client = client
        self.proofs = {}

    def _s3(self):
        if self.client is None:
            import boto3

            self.client = boto3.client("s3")
        return self.client

    def read(self, key, *, digest=None, size=None):
        path = PurePosixPath(key)
        require(
            not path.is_absolute() and all(p not in {"", ".", ".."} for p in path.parts),
            "Unsafe object key",
        )
        local = self.root / "objects" / key
        if not local.is_file():
            require(not self.offline, f"Missing offline object: {key}")
            payload = self._s3().get_object(Bucket=self.bucket, Key=key)["Body"].read()
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(payload)
        payload = local.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        require(digest is None or actual == digest, f"Checksum mismatch: {key}")
        require(size is None or len(payload) == size, f"Size mismatch: {key}")
        self.proofs[key] = {"sha256": actual, "bytes": len(payload)}
        return payload

    def keys(self, prefix):
        index = self.root / "indexes" / (hashlib.sha256(prefix.encode()).hexdigest() + ".json")
        if self.offline:
            require(index.is_file(), f"Missing offline listing: {prefix}")
            return json.loads(index.read_text())
        keys = [
            v["Key"]
            for page in self._s3()
            .get_paginator("list_objects_v2")
            .paginate(Bucket=self.bucket, Prefix=prefix)
            for v in page.get("Contents", [])
        ]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(keys))
        return keys

    def receipt(self, entry, prefix):
        base = f"s3://{self.bucket}/"
        require(entry["uri"].startswith(base + prefix), "Evidence outside declared run")
        key = entry["uri"][len(base) :]
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid evidence digest")
        require(
            PurePosixPath(key).name.startswith(entry["sha256"] + "-"),
            "Evidence is not content addressed",
        )
        require(type(entry["bytes"]) is int and entry["bytes"] > 0, "Invalid evidence size")
        return self.read(key, digest=entry["sha256"], size=entry["bytes"])


def hydrate_truth(archive, plan):
    from src.analysis.audit_development_truth import build_truth, required_files

    release = plan["data_release"]
    base = f"data/releases/{release}/"
    manifest = json.loads(archive.read(base + "manifest.json", digest=release))
    names = sorted({name for position in plan["positions"] for name in required_files(position)})

    def fetch(name):
        require(name in manifest["files"], f"Missing genuine release source: {name}")
        meta = manifest["files"][name]
        archive.read(base + name, digest=meta["sha256"], size=meta["bytes"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, names))
    root = archive.root / "objects" / base
    seasons = sorted(set(plan["origins"]) | {year - 1 for year in plan["origins"]})
    return {
        position: build_truth(position, root, seasons=seasons) for position in plan["positions"]
    }


def _numbers(error):
    values = np.asarray(error, dtype=float)
    require(len(values) > 0 and np.isfinite(values).all(), "Incomplete/nonfinite model samples")
    return {
        "n": len(values),
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
    }


def _same_scores(actual, expected, label, *, tolerance=1e-11):
    require(
        type(expected.get("n")) is int and actual["n"] == expected["n"],
        f"Incomplete samples: {label}",
    )
    for metric in ("mae", "rmse"):
        require(
            np.isfinite(expected[metric])
            and np.isclose(actual[metric], expected[metric], rtol=tolerance, atol=tolerance),
            f"Rescore mismatch: {label}/{metric}",
        )


@dataclass
class Verified:
    record: dict
    rows: pd.DataFrame
    scores: dict
    states: dict


def verify_record(record, truth, read_receipt):
    """Rescore saved predictions and checkpoint-validation outputs independently."""
    import torch

    require(record.get("schema") == "isolated-audit-development/v1", "Unknown evidence schema")
    require(record.get("promotion_eligible") is False, "Isolated diagnosis cannot claim promotion")
    position, origin = record["position"], record["origin"]
    targets = tuple(
        importlib.import_module(f"src.{position.lower()}.config").POSITION_CONFIG.targets
    )
    execution_signature(record)
    require(
        record.get("gpu")
        and record.get("tf32_matmul") is True
        and record.get("tf32_cudnn") is True,
        "Missing paired FP32/TF32 evidence",
    )
    hashes = record["prepared_hashes"]
    needed = {s for s in ("train", "val", "test")}
    needed |= {f"X_{s}" for s in ("train", "val", "test")}
    needed |= {f"y_{s}/{t}" for s in ("train", "val", "test") for t in targets}
    require(
        needed <= set(hashes) and all(re.fullmatch(r"[0-9a-f]{64}", hashes[k]) for k in needed),
        "Incomplete prepared-input fingerprints",
    )
    rows = (
        pd.read_parquet(io.BytesIO(read_receipt(record["rows"])))
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    require(
        set(rows.season.unique()) == {origin} and not rows.duplicated(KEYS).any(),
        "Wrong origin or duplicate rows",
    )
    for field, cols in (("row_hash", KEYS), ("truth_hash", [*KEYS, "comparison_actual"])):
        require(
            hashlib.sha256(
                pd.util.hash_pandas_object(rows[cols], index=False).values.tobytes()
            ).hexdigest()
            == record[field],
            f"Invalid {field}",
        )
    expected = regular_season_rows(truth)
    expected["player_id"] = expected.player_id.astype(str)
    expected["fantasy_points"] = comparison_actuals(expected, position)
    require(not expected.duplicated(KEYS).any(), "Duplicate independent truth rows")
    assessment = expected[expected.season.eq(origin)].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[KEYS], assessment[KEYS], check_dtype=False)
    np.testing.assert_allclose(
        rows.comparison_actual, assessment.fantasy_points, rtol=0, atol=0, equal_nan=True
    )
    frame = comparison_model_totals(rows, position, rescore=True)
    if position != "DST":
        require(
            set(targets) == set(scoring_components(position)),
            "Native aggregation has different projected components",
        )
        for family in FAMILIES:
            # K's production sign-vector aggregation retains the raw neural
            # head dtype. Promoting FP32 heads before their sum would change
            # the reported errors, even though the components are identical.
            frame[f"pred_{family}_total"] = predictions_to_fantasy_points(
                position,
                {target: rows[f"pred_{family}_{target}"].to_numpy() for target in targets},
                "ppr",
            )
    frame["fantasy_points"] = comparison_actuals(frame, position)
    np.testing.assert_allclose(
        frame.fantasy_points, assessment.fantasy_points, rtol=0, atol=0, equal_nan=True
    )
    for family in FAMILIES:
        for target in (*targets, "total"):
            require(
                np.isfinite(rows[f"pred_{family}_{target}"]).all(),
                f"Missing forecasts: {family}/{target}",
            )
        np.testing.assert_allclose(
            rows[f"pred_{family}_total"], frame[f"pred_{family}_total"], rtol=1e-12, atol=1e-12
        )
    frame = frame[frame.fantasy_points.notna()].copy()
    prior = (
        expected[expected.season.eq(origin - 1)]
        .groupby(["player_id", "season"])
        .fantasy_points.mean()
    )
    require(not prior.empty, "Missing genuine prior-season cohort coverage")
    lookup = pd.MultiIndex.from_arrays([frame.player_id, frame.season - 1])
    frame["prior_points"] = prior.reindex(lookup).to_numpy()
    players = ranked_rows(
        frame.drop_duplicates(["season", "player_id"]), "prior_points", ["season"], 24
    )
    selected = pd.MultiIndex.from_frame(players[["season", "player_id"]])
    elite = frame[pd.MultiIndex.from_frame(frame[["season", "player_id"]]).isin(selected)]
    cohort = record["cohorts"]["elite_top24"]
    require(
        cohort["status"] == "available"
        and cohort["n"] == len(elite) > 0
        and cohort["cohort_hash"] == _identity(elite),
        "Unverifiable protected cohort",
    )
    weekly = record["cohorts"]["weekly_reference_top24"]
    require(
        weekly["status"] == "unavailable" and weekly["n"] is None and not weekly["models"],
        "Pre-2024 weekly cohort must be explicitly unavailable",
    )
    scores = {}
    for name, subset in (("all", frame), ("elite_top24", elite)):
        for family in FAMILIES:
            key = f"{name}:{family}"
            scores[key] = _numbers(subset[f"pred_{family}_total"] - subset.fantasy_points)
            _same_scores(scores[key], record["metrics"][key], key)
            if name == "elite_top24":
                _same_scores(scores[key], cohort["models"][family], key)
    require(record["inference_parity_passed"] is True, "Missing saved-inference parity")
    parity = record["inference_parity_max_absolute_error"]
    require(
        set(parity) == {f"pred_{f}_{t}" for f in FAMILIES for t in (*targets, "total")},
        "Incomplete saved-inference comparison",
    )
    for column, error in parity.items():
        # The pinned observer checks per-row allclose; reject receipts outside
        # even its largest permitted absolute error at this prediction scale.
        require(
            np.isfinite(error) and 0 <= error <= 1e-5 + 1e-5 * rows[column].abs().max(),
            "Failed saved-inference tolerance",
        )
    states = {}
    for trainer in record["trainers"]:
        family = trainer["family"]
        require(
            trainer["graph"] is (not (position == "K" and family == "attn_nn")),
            "Non-production graph policy",
        )
        data = np.load(io.BytesIO(read_receipt(trainer["validation_raw"])))
        predictions = {t: data[f"prediction__{t}"] for t in targets}
        actuals = {t: data[f"truth__{t}"] for t in targets}
        if "prediction__receptions_value_mu" in data:
            numerical = trainer["count_numerics"]
            positive = data["truth__receptions"] > 0
            require(
                numerical["n_positive"] == int(positive.sum()) > 0,
                "Incomplete count numerical observations",
            )
            require(
                str(numerical["production_device"]).startswith("cuda")
                and numerical["reference_device"] == "cpu",
                "Wrong numerical-reference execution",
            )
            for label, key in (
                ("mu", "prediction__receptions_value_mu"),
                ("log_alpha", "prediction__receptions_value_log_alpha"),
            ):
                values = data[key][positive]
                np.testing.assert_allclose(
                    numerical["observed_ranges"][label],
                    [values.min(), values.max()],
                    rtol=0,
                    atol=0,
                )
            require(
                set(numerical["errors"]) == {"log_probability", "d_mu", "d_log_alpha"},
                "Missing likelihood/gradient comparisons",
            )
            failures = []
            for value in numerical["errors"].values():
                count = value["n_outside_tolerance"]
                maximum = value["max_scaled_error"]
                require(
                    type(count) is int and 0 <= count <= int(positive.sum()),
                    "Invalid gradient discrepancy count",
                )
                require(
                    maximum is None or np.isfinite(maximum) and maximum >= 0,
                    "Invalid gradient discrepancy",
                )
                require(
                    bool(count) == (maximum is None or maximum > 1e-4),
                    "Inconsistent numerical tolerance receipt",
                )
                failures.append(bool(count))
            require(
                numerical["active_numerical_defect"] is any(failures),
                "Inconsistent numerical defect flag",
            )
        errors = predictions_to_fantasy_points(position, predictions, "ppr").astype(
            float
        ) - predictions_to_fantasy_points(position, actuals, "ppr").astype(float)
        scores_at_restore = _numbers(errors)
        _same_scores(
            scores_at_restore,
            {"n": len(errors), **trainer["restored_scores"]},
            f"restored/{family}",
            tolerance=2e-5,
        )
        checkpoint = trainer["checkpoint_selection"]
        epoch, curve = checkpoint["epoch"], checkpoint["validation_curve"]
        require(
            type(epoch) is int and 1 <= epoch <= len(curve) == trainer["epochs_executed"],
            "Invalid selected checkpoint epoch",
        )
        require(
            trainer["stop_reason"] in {"budget", "patience"} and not checkpoint["fixed_epochs"],
            "Changed stopping policy",
        )
        require(
            checkpoint["score"] == curve[epoch - 1] == min(v for v in curve if v is not None),
            "Checkpoint score does not match returned epoch",
        )
        for metric in ("mae", "rmse"):
            expected_score = checkpoint["validation_metrics"][f"val_fantasy_{metric}_ppr"]
            require(
                np.isclose(expected_score, scores_at_restore[metric], rtol=2e-5, atol=2e-6),
                "Selected checkpoint independently rescored differently",
            )
        states[family] = torch.load(
            io.BytesIO(read_receipt(trainer["checkpoint"])), map_location="cpu", weights_only=True
        )
    return Verified(record, rows, scores, states)


def _paired(base, other, candidate, *, repeat=False):
    import torch

    a, b = base.record, other.record
    if candidate == "stint_reset":
        from src.analysis.audit_input_proof import verify_stint_pair

        verify_stint_pair(a, b, repeat=repeat)
    require(execution_signature(a) == execution_signature(b), "Unmatched execution settings")
    for field in (
        "source_sha",
        "data_release",
        "gpu",
        "tf32_cudnn",
        "row_hash",
        "truth_hash",
        "batch_job_id",
    ):
        require(a[field] == b[field], f"Unpaired {field}")
    fields = set(a["prepared_hashes"]) | set(b["prepared_hashes"])
    if candidate == "stint_reset" and not repeat:
        fields = {key for key in fields if key.startswith("y_")}
        allowed = {"opportunity_index_L3", "redzone_target_share_L3"}
        raw = [c for c in base.rows if not c.startswith("pred_") and c not in allowed]
        pd.testing.assert_frame_equal(base.rows[raw], other.rows[raw], check_exact=True)
    require(
        all(a["prepared_hashes"].get(k) == b["prepared_hashes"].get(k) for k in fields),
        "Unpaired prepared inputs",
    )
    for name in ("elite_top24", "weekly_reference_top24"):
        left, right = a["cohorts"][name], b["cohorts"][name]
        require(
            {k: v for k, v in left.items() if k != "models"}
            == {k: v for k, v in right.items() if k != "models"},
            "Unpaired cohort identity",
        )
    controls = FAMILIES if repeat else tuple(f for f in FAMILIES if f not in AFFECTED[candidate])
    for family in controls:
        columns = [c for c in base.rows if c.startswith(f"pred_{family}_")]
        pd.testing.assert_frame_equal(base.rows[columns], other.rows[columns], check_exact=True)
        if family in ("nn", "attn_nn"):
            require(
                base.states[family].keys() == other.states[family].keys(),
                "Changed control checkpoint schema",
            )
            require(
                all(
                    torch.equal(value, other.states[family][key])
                    for key, value in base.states[family].items()
                ),
                "Changed control checkpoint tensors",
            )


def summarize(verified, plan, *, errors=()):
    plan = validate_plan(plan)
    candidate = plan["candidate"]
    problems = list(errors)
    index = {}
    for cell in verified:
        r = cell.record
        key = (r["position"], r["origin"], r["seed"], r["variant"])
        require(key not in index, f"Duplicate evidence, possibly mixed smoke/full runs: {key}")
        index[key] = cell
    expected = {
        (p, y, s, v)
        for p in plan["positions"]
        for y in plan["origins"]
        for s in plan["seeds"]
        for v in ("baseline", "baseline_rep", candidate)
    }
    if set(index) != expected:
        problems.append(
            f"Incomplete grid: missing={sorted(expected - set(index))}, extra={sorted(set(index) - expected)}"
        )
    comparisons = []
    for position in plan["positions"]:
        for origin in plan["origins"]:
            paired = []
            for seed in plan["seeds"]:
                keys = [
                    (position, origin, seed, v) for v in ("baseline", "baseline_rep", candidate)
                ]
                if not all(k in index for k in keys):
                    continue
                baseline, repeated, proposed = [index[k] for k in keys]
                try:
                    _paired(baseline, repeated, candidate, repeat=True)
                    _paired(baseline, proposed, candidate)
                except (ValueError, AssertionError) as error:
                    problems.append(f"{position}/{origin}/{seed}: {error}")
                    continue
                paired.append((seed, baseline, proposed))
            for family in AFFECTED[candidate]:
                row = {
                    "position": position,
                    "origin": origin,
                    "family": family,
                    "seeds": [p[0] for p in paired],
                    "deltas": {},
                    "passes": False,
                }
                if len(paired) == len(plan["seeds"]):
                    for cohort in ("all", "elite_top24"):
                        for metric in ("mae", "rmse"):
                            values = [
                                b.scores[f"{cohort}:{family}"][metric]
                                - a.scores[f"{cohort}:{family}"][metric]
                                for _, a, b in paired
                            ]
                            delta, sd = mean(values), stdev(values) if len(values) > 1 else None
                            half = 4.302652729911275 * sd / np.sqrt(3) if len(values) == 3 else None
                            row["deltas"][f"{cohort}:{metric}"] = {
                                "paired": values,
                                "mean": delta,
                                "std": sd,
                                "baseline_mean": mean(
                                    a.scores[f"{cohort}:{family}"][metric] for _, a, _ in paired
                                ),
                                "candidate_mean": mean(
                                    b.scores[f"{cohort}:{family}"][metric] for _, _, b in paired
                                ),
                                "ci95_seed_mean": None
                                if half is None
                                else [delta - half, delta + half],
                            }
                    row["passes"] = all(
                        v["mean"] < 0 if key.startswith("all:") else v["mean"] <= 0
                        for key, v in row["deltas"].items()
                    )
                comparisons.append(row)
    numerical_findings = [
        {
            "position": cell.record["position"],
            "origin": cell.record["origin"],
            "seed": cell.record["seed"],
            "variant": cell.record["variant"],
            "family": trainer["family"],
            "numerics": trainer["count_numerics"],
        }
        for cell in verified
        for trainer in cell.record["trainers"]
        if trainer.get("count_numerics", {}).get("active_numerical_defect")
    ]
    numerical_pass = candidate != "count_precision" or not any(
        f["variant"] == candidate for f in numerical_findings
    )
    return {
        "schema": "audit-development-report/v1",
        "purpose": plan["purpose"],
        "candidate": candidate,
        "records": len(index),
        "expected_records": len(expected),
        "verification_passed": not problems,
        "errors": sorted(set(problems)),
        "comparisons": comparisons,
        "numerical_findings": numerical_findings,
        "candidate_numerical_checks_passed": numerical_pass,
        "cells": [
            {
                "position": cell.record["position"],
                "origin": cell.record["origin"],
                "seed": cell.record["seed"],
                "variant": cell.record["variant"],
                "batch_job_id": cell.record["batch_job_id"],
                "gpu": cell.record["gpu"],
                "source_sha": cell.record["source_sha"],
                "image_digest": plan["sources"][cell.record["source_sha"]]["image_digest"],
                "metrics": cell.scores,
                "inference_max_absolute_error": max(
                    cell.record["inference_parity_max_absolute_error"].values()
                ),
                "selected_checkpoints": {
                    t["family"]: {
                        "epoch": t["checkpoint_selection"]["epoch"],
                        "epochs_executed": t["epochs_executed"],
                        "stop_reason": t["stop_reason"],
                        "restored_scores": t["restored_scores"],
                    }
                    for t in cell.record["trainers"]
                },
            }
            for cell in verified
        ],
        "development_qualified": plan["purpose"] == "development"
        and not problems
        and numerical_pass
        and all(r["passes"] for r in comparisons),
        "merge_authorized": False,
        "confirmation_complete": False,
        "weekly_reference_top24": "Unavailable before 2024; required confirmation remains separate.",
        "uncertainty": "Paired sample standard deviations and Student-t intervals describe three training seeds within each retrospective season, not independent season uncertainty.",
    }


def analyze(plan, archive):
    from src.analysis.audit_source_bridge import verify_bridge

    plan = validate_plan(plan)
    bridges = []
    anchors = [source for source, pin in plan["sources"].items() if "bridge" not in pin]
    for source, pin in plan["sources"].items():
        if "bridge" in pin:
            proof = verify_bridge(source, pin["bridge"])
            require(
                len(plan["sources"]) == 1 or proof["baseline_source_sha"] == anchors[0],
                "Source bridge has a different baseline",
            )
            bridges.append(proof)
    truths = hydrate_truth(archive, plan)
    verified, errors = [], []
    candidate = plan["candidate"]
    for run_plan in plan["runs"]:
        prefix = run_plan["prefix"].rstrip("/") + "/"
        source = run_plan["source_sha"]
        image = plan["sources"][source]["image_digest"]
        try:
            run = json.loads(archive.read(prefix + "run.json"))
            require(run["spec"] in SPECS[candidate], "Wrong experiment spec")
            require(
                run["image_sha"] == source and run["image_digest"] == image,
                "Mixed source/image pins",
            )
            require(run["variants"] == ["baseline", "baseline_rep", candidate], "Unexpected arms")
            require(
                run["positions"] == run_plan["positions"] and run["seeds"] == run_plan["seeds"],
                "Unexpected run axes",
            )
            require(
                run["extra_env"]["FF_AUDIT_ORIGIN"] == str(run_plan["origin"]), "Wrong run origin"
            )
            require(
                run["extra_env"].get("FF_AMP_DTYPE") == "fp32" and run["cuda_graph"] == "auto",
                "Wrong run execution policy",
            )
            keys = archive.keys(prefix)
            manifests = [k for k in keys if "/evidence/" in k and k.endswith("-manifest.json")]
            wanted_cells = {
                (p, s, v)
                for p in run_plan["include_positions"]
                for s in run_plan["seeds"]
                for v in run["variants"]
            }
            seen = set()
            for key in manifests:
                raw = archive.read(key, digest=PurePosixPath(key).name.split("-")[0])
                record = json.loads(raw)
                cell_key = (record["position"], record["seed"], record["variant"])
                require(
                    cell_key in wanted_cells and cell_key not in seen,
                    "Unexpected or duplicate manifest",
                )
                seen.add(cell_key)
                require(record["source_sha"] == source, "Wrong source_sha")
                for field in ("data_release", "baseline_sha"):
                    require(record[field] == plan[field], f"Wrong {field}")
                require(
                    record["candidate_pins"][candidate] == plan["candidate_sha"],
                    "Wrong candidate pin",
                )
                require(
                    record["origin"] == run_plan["origin"]
                    and record["batch_job_id"] == run["jobs"][record["position"]],
                    "Wrong origin/job",
                )
                small = json.loads(
                    archive.read(
                        prefix
                        + f"cells/{record['position']}-{record['variant']}-{record['seed']}.json"
                    )
                )
                metrics = dict(small["metrics"])
                auxiliary = metrics.pop("repair", None)
                require(
                    auxiliary
                    == {
                        "origin": record["origin"],
                        "trainer_count": 2,
                        "weekly_reference_available": 0,
                    },
                    "Missing or inconsistent cell summary",
                )
                require(
                    small["ok"] is True and metrics == record["metrics"],
                    "Failed cell or inconsistent cell metrics",
                )
                verified.append(
                    verify_record(
                        record,
                        truths[record["position"]],
                        lambda entry, prefix=prefix: archive.receipt(entry, prefix),
                    )
                )
            require(seen == wanted_cells, f"Missing completed cells: {sorted(wanted_cells - seen)}")
            all_cells = {
                f"{p}-{v}-{s}.json"
                for p in run_plan["positions"]
                for s in run_plan["seeds"]
                for v in run["variants"]
            }
            for key in (k for k in keys if "/cells/" in k and k.endswith(".json")):
                require(PurePosixPath(key).name in all_cells, "Extra cell objects")
                if (
                    PurePosixPath(key).name.startswith("TE-")
                    and "TE" in run_plan["superseded_positions"]
                ):
                    ignored = json.loads(archive.read(key))
                    require(
                        ignored["ok"] is False
                        and "Repair result does not match its execution context"
                        in ignored.get("error", ""),
                        "Supersession would discard a usable or differently failed cell",
                    )
        except (KeyError, ValueError, AssertionError, FileNotFoundError) as error:
            errors.append(f"{prefix}: {error}")
    report = summarize(verified, plan, errors=errors)
    report["plan"] = plan
    report["source_bridges"] = bridges
    report["source_pins"] = plan["sources"]
    report["object_proofs"] = archive.proofs
    return report


def markdown(report):
    rows = [
        "# Isolated development evidence",
        "",
        f"Candidate: `{report['candidate']}`. Purpose: **{report['purpose']}**.",
        "",
        f"Verified grid: {report['records']}/{report['expected_records']} cells. Verification: {'PASS' if report['verification_passed'] else 'FAIL'}. Development qualification: {report['development_qualified']}.",
        "",
        "These results do not authorize a merge or establish confirmation.",
        "",
        "| Position | Season | Model | MAE delta | RMSE delta | Elite MAE delta | Elite RMSE delta | Screen |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in report["comparisons"]:
        values = [
            f"{r['deltas'][k]['mean']:+.6f}" if k in r["deltas"] else "unavailable"
            for k in ("all:mae", "all:rmse", "elite_top24:mae", "elite_top24:rmse")
        ]
        rows.append(
            f"| {r['position']} | {r['origin']} | {r['family']} | "
            + " | ".join(values)
            + f" | {'pass' if r['passes'] else 'hold'} |"
        )
    rows += ["", report["weekly_reference_top24"], "", report["uncertainty"]]
    if report["errors"]:
        rows += ["", "Verification errors:", "", *[f"- {error}" for error in report["errors"]]]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = analyze(
        json.loads(args.plan.read_text()),
        Archive(args.cache_dir, args.bucket, offline=args.offline),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "records",
                    "expected_records",
                    "verification_passed",
                    "development_qualified",
                    "errors",
                )
            }
        )
    )
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
