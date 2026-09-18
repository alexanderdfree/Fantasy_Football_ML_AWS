"""Paired model-response reports for two replays of the same sampled donors.

A treatment cohort applies transforms after sampling with the same seed as its
baseline, so case ``i`` of both shares the donor window. The report pairs cases
by index, verifies the donor keys, and summarizes how each model family's
response moved. Responses have no observed outcome: this is model response to
constructed histories, never forecast accuracy, and the vocabulary stays that
way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.synthetic_history import publish_artifact_dir
from src.analysis.synthetic_history_schema import position_schema
from src.prediction.frames import SCORING_FORMATS
from src.shared.aggregate_targets import predictions_to_fantasy_points

REPORT_KIND = "synthetic_model_response"
SEMANTICS = (
    "paired model response to constructed histories; no observed outcome; not forecast accuracy"
)
PAIRING_KEY = ("case_index", "donor_player_id", "donor_season", "forecast_week")
FAMILY_LABELS = {
    "ridge": "Ridge",
    "nn": "Base NN",
    "attn_nn": "Attention NN",
    "lgbm": "LightGBM",
}


def load_replay(directory: Path) -> tuple[dict, pd.DataFrame]:
    directory = Path(directory)
    manifest = json.loads((directory / "replay_manifest.json").read_text())
    predictions = pd.read_parquet(directory / "predictions.parquet")
    for key in ("recipe", "source_values_sha256", "sampling_identity_sha256", "cohort_name"):
        if key not in manifest:
            raise ValueError(f"replay manifest is missing {key}; regenerate the replay")
    return manifest, predictions


def _families(frame: pd.DataFrame) -> list[str]:
    return [
        column[len("pred_") : -len("_total")]
        for column in frame.columns
        if column.startswith("pred_") and column.endswith("_total")
    ]


def pair_replays(
    baseline: tuple[dict, pd.DataFrame], treatment: tuple[dict, pd.DataFrame]
) -> pd.DataFrame:
    """Align two replays case by case; refuse different donors or sources."""
    (base_manifest, base), (treat_manifest, treat) = baseline, treatment
    if base_manifest["source_values_sha256"] != treat_manifest["source_values_sha256"]:
        raise ValueError("baseline and treatment were not generated from the same source values")
    if base_manifest["sampling_identity_sha256"] != treat_manifest["sampling_identity_sha256"]:
        raise ValueError(
            "baseline and treatment were not sampled from the same donors: their sampling "
            "recipes (seed, cases, history, window, bounds, mode) differ"
        )
    if len(base) != len(treat):
        raise ValueError("baseline and treatment replay different numbers of cases")
    base = base.sort_values("case_index").reset_index(drop=True)
    treat = treat.sort_values("case_index").reset_index(drop=True)
    for key in PAIRING_KEY:
        if not base[key].equals(treat[key]):
            raise ValueError(f"baseline and treatment disagree on {key}")
    return base.merge(treat, on=list(PAIRING_KEY), suffixes=("_baseline", "_treatment"))


def _delta_summary(values: np.ndarray) -> dict:
    median = float(np.median(values))
    signed = np.sign(values)
    consistency = float(np.mean(signed == np.sign(median))) if median != 0 else None
    return {
        "mean": float(np.mean(values)),
        "median": median,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "share_positive": float(np.mean(values > 0)),
        "share_negative": float(np.mean(values < 0)),
        "share_zero": float(np.mean(values == 0)),
        "sign_consistency": consistency,
    }


def _totals(pairs: pd.DataFrame, family: str, side: str, targets, position: str, scoring: str):
    column = (
        f"pred_{family}_total_{side}"
        if scoring == "ppr"
        else f"pred_{family}_total_{scoring}_{side}"
    )
    if column in pairs:
        return pairs[column].to_numpy(dtype=float)
    raw = {t: pairs[f"pred_{family}_{t}_{side}"].to_numpy(dtype=float) for t in targets}
    return predictions_to_fantasy_points(position, raw, scoring)


def summarize_response(
    pairs: pd.DataFrame, *, families, targets, position: str, scoring: str = "ppr"
) -> dict:
    summary = {}
    for family in families:
        base = _totals(pairs, family, "baseline", targets, position, scoring)
        treat = _totals(pairs, family, "treatment", targets, position, scoring)
        delta_targets = {}
        for target in targets:
            columns = (f"pred_{family}_{target}_baseline", f"pred_{family}_{target}_treatment")
            if all(column in pairs for column in columns):
                delta = pairs[columns[1]].to_numpy(dtype=float) - pairs[columns[0]].to_numpy(
                    dtype=float
                )
                delta_targets[target] = {
                    "mean": float(np.mean(delta)),
                    "median": float(np.median(delta)),
                }
        summary[family] = {
            "label": FAMILY_LABELS.get(family, family),
            "n": int(len(pairs)),
            "baseline_mean_total": float(np.mean(base)),
            "treatment_mean_total": float(np.mean(treat)),
            "delta_total": _delta_summary(treat - base),
            "delta_targets": delta_targets,
        }
    return summary


def build_report(baseline_dir: Path, treatment_dir: Path, *, scoring: str = "ppr") -> dict:
    baseline = load_replay(baseline_dir)
    treatment = load_replay(treatment_dir)
    pairs = pair_replays(baseline, treatment)
    position = baseline[0]["recipe"]["position"]
    schema = position_schema(position)
    families = [f for f in _families(baseline[1]) if f in _families(treatment[1])]
    if not families:
        raise ValueError("baseline and treatment share no replayed model family")
    excluded = {
        family: reason
        for manifest, _ in (baseline, treatment)
        for family, reason in manifest.get("families_excluded", {}).items()
        if family not in families
    }
    history = {
        "donor_mean": float(pairs["donor_history_ppg_baseline"].mean()),
        "baseline_generated_mean": float(pairs["generated_history_ppg_baseline"].mean()),
        "treatment_generated_mean": float(pairs["generated_history_ppg_treatment"].mean()),
    }
    history["generated_delta_mean"] = (
        history["treatment_generated_mean"] - history["baseline_generated_mean"]
    )

    def _side(manifest: dict, directory: Path) -> dict:
        return {
            "replay_dir": str(directory),
            "cohort_name": manifest["cohort_name"],
            "history_kind": manifest.get("history_kind", "donor"),
            "fixture": bool(manifest.get("fixture", False)),
            "recipe_sha256": manifest["recipe_sha256"],
            "cohort_manifest_sha256": manifest["cohort_manifest_sha256"],
        }

    return {
        "report_kind": REPORT_KIND,
        "semantics": SEMANTICS,
        "scoring_format": scoring,
        "scoring_scope": schema.scoring_scope,
        "position": position,
        "baseline": _side(baseline[0], baseline_dir),
        "treatment": _side(treatment[0], treatment_dir),
        "pairing": {
            "key": list(PAIRING_KEY),
            "n_pairs": int(len(pairs)),
            "unique_donor_windows": int(
                pairs.drop_duplicates(["donor_player_id", "donor_season", "forecast_week"]).shape[0]
            ),
            "sampling_identity_sha256": baseline[0]["sampling_identity_sha256"],
        },
        "history_ppg": history,
        "families": summarize_response(
            pairs, families=families, targets=schema.targets, position=position, scoring=scoring
        ),
        "excluded_families": excluded,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Model response: {report['treatment']['cohort_name']} vs {report['baseline']['cohort_name']}",
        "",
        f"{report['semantics']}. Scoring: {report['scoring_format']} ({report['scoring_scope']}).",
        "",
        f"Pairs: {report['pairing']['n_pairs']} cases over "
        f"{report['pairing']['unique_donor_windows']} unique donor windows; history points "
        f"donor {report['history_ppg']['donor_mean']:.2f}, baseline "
        f"{report['history_ppg']['baseline_generated_mean']:.2f}, treatment "
        f"{report['history_ppg']['treatment_generated_mean']:.2f}.",
        "",
        "| Family | n | Baseline mean | Treatment mean | Δ mean | Δ median | Δ std | Sign consistency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _family, entry in report["families"].items():
        delta = entry["delta_total"]
        consistency = (
            "n/a" if delta["sign_consistency"] is None else f"{delta['sign_consistency']:.2f}"
        )
        lines.append(
            f"| {entry['label']} | {entry['n']} | {entry['baseline_mean_total']:.3f} | "
            f"{entry['treatment_mean_total']:.3f} | {delta['mean']:+.3f} | {delta['median']:+.3f} | "
            f"{delta['std']:.3f} | {consistency} |"
        )
    targets = sorted({t for entry in report["families"].values() for t in entry["delta_targets"]})
    if targets:
        lines += ["", "| Family | " + " | ".join(f"Δ {t}" for t in targets) + " |"]
        lines.append("|---|" + "---|" * len(targets))
        for _family, entry in report["families"].items():
            cells = [
                f"{entry['delta_targets'][t]['mean']:+.3f}" if t in entry["delta_targets"] else "—"
                for t in targets
            ]
            lines.append(f"| {entry['label']} | " + " | ".join(cells) + " |")
    if report["excluded_families"]:
        lines += ["", "Families not replayed on both sides:"]
        lines += [f"- {family}: {reason}" for family, reason in report["excluded_families"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline replay directory")
    parser.add_argument("--treatment", type=Path, required=True, help="Treatment replay directory")
    parser.add_argument(
        "--output", type=Path, default=None, help="New report directory; prints markdown if omitted"
    )
    parser.add_argument("--scoring", choices=SCORING_FORMATS, default="ppr")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        report = build_report(args.baseline, args.treatment, scoring=args.scoring)
        markdown = render_markdown(report)
        if args.output is not None:

            def _write(directory: Path) -> None:
                (directory / "response_report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
                )
                (directory / "response_report.md").write_text(markdown)

            publish_artifact_dir(args.output, _write, {"report_kind": REPORT_KIND})
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"synthetic-response: {exc}\n")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
