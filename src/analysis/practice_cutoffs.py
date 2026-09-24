"""Compare archived live forecasts at identical, game-relative 48/24-hour cutoffs.

No historical weekly feed is accepted as prospective evidence. Missing source
times stay unknown; observation and forecast availability bound what was known.

    python -m src.analysis.practice_cutoffs --baseline-dir <archive> \
        --candidate-dir <shadow-archive> --actuals <raw-target-actuals.parquet> \
        --output <comparison.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts.practice_archive import SCHEMA_VERSION
from src.evaluation.metrics import compute_metrics
from src.features.practice_context import reason_features
from src.shared.comparison_scoring import comparison_actuals
from src.shared.evaluation_cohorts import regular_season_rows

KEYS = ["player_id", "season", "week", "position"]
MODELS = ("ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred")


def load_snapshots(directory):
    for path in sorted(Path(directory).rglob("*.json")):
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != path.stem:
            raise ValueError(f"Practice archive content hash mismatch: {path}")
        snapshot = json.loads(body)
        if snapshot.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported practice archive schema: {path}")
        yield snapshot


def _time(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def select_cutoff_rows(snapshots, *, hours=48, scoring="ppr") -> pd.DataFrame:
    """Select latest eligible observation per player/game, never a later revision."""
    if hours not in (24, 48):
        raise ValueError("Practice evaluation supports the preregistered 48/24-hour cutoffs")
    chosen = {}
    for snapshot in snapshots:
        if snapshot.get("evidence_kind") != "observed_live_snapshot":
            raise ValueError("Cutoff evaluation requires observed live snapshots")
        forecast = snapshot["forecast"]
        if not forecast.get("available"):
            continue
        timestamps = [
            _time(snapshot.get("observed_at")),
            _time(snapshot.get("forecast_available_at")),
            _time(forecast.get("generated_at")),
        ]
        if any(pd.isna(value) for value in timestamps):
            raise ValueError("Archive lacks observation or forecast availability time")
        available = max(timestamps)
        games = {game["team"]: _time(game.get("kickoff")) for game in snapshot["games"]}
        reports = {row["player_id"]: row for row in snapshot["practice"]["observations"]}
        for row in forecast["scoring"][scoring]:
            if row["position"] not in {"QB", "RB", "WR", "TE"}:
                continue
            kickoff = games.get(row["team"])
            if kickoff is None or pd.isna(kickoff):
                continue  # No inferred Sunday kickoff for Thursday/Monday games.
            cutoff = kickoff - pd.Timedelta(hours=hours)
            report = reports.get(row["player_id"])
            if report is None:
                continue
            if (report["season"], report["week"]) != (snapshot["season"], snapshot["week"]):
                raise ValueError("Practice report week differs from archived forecast")
            observed = _time(report.get("observed_at"))
            reported = _time(report.get("reported_at"))
            if pd.isna(observed) or max(available, observed) > cutoff:
                continue
            if pd.notna(reported) and reported > cutoff:
                continue
            key = (row["player_id"], snapshot["season"], snapshot["week"], row["position"])
            candidate = {
                **dict(zip(KEYS, key, strict=True)),
                **{model: row.get(model) for model in MODELS},
                **reason_features(report["injury_descriptions"], coverage=report["coverage"]),
                "cutoff_hours": hours,
                "kickoff": kickoff.isoformat(),
                "available_at": max(available, observed).isoformat(),
                "input_signature": forecast.get("input_signature"),
            }
            previous = chosen.get(key)
            if previous is None or candidate["available_at"] > previous["available_at"]:
                chosen[key] = candidate
    return pd.DataFrame(chosen.values(), columns=[*KEYS, *MODELS] if not chosen else None)


def compare_cutoff(baseline, candidate, actuals, *, scoring="ppr") -> dict:
    """Same player-weeks/components in both arms, with explicit missing coverage."""
    joined = baseline.merge(
        candidate,
        on=KEYS,
        how="outer",
        suffixes=("_base", "_candidate"),
        validate="1:1",
        indicator=True,
    )
    coverage = joined["_merge"].value_counts().to_dict()
    paired = joined[joined["_merge"].eq("both")].drop(columns="_merge")
    if not paired.empty:
        if not paired["kickoff_base"].eq(paired["kickoff_candidate"]).all():
            raise ValueError("Arms use different game kickoffs; resolve schedule revisions first")
        if not paired["cutoff_hours_base"].eq(paired["cutoff_hours_candidate"]).all():
            raise ValueError("Arms use different forecast cutoffs")
    actuals = regular_season_rows(actuals)
    scored = []
    for position, frame in actuals.groupby("position"):
        if position in {"QB", "RB", "WR", "TE"}:
            scored.append(frame[KEYS].assign(actual=comparison_actuals(frame, position, scoring)))
    truth = pd.concat(scored) if scored else pd.DataFrame(columns=[*KEYS, "actual"])
    paired = paired.merge(truth, on=KEYS, how="left", validate="1:1")
    output = {
        "coverage": {str(key): int(value) for key, value in coverage.items()},
        "missing_actuals": int(paired["actual"].isna().sum()),
        "positions": {},
        "acceptance": "requires protected-cohort review; never auto-promotes a model",
    }
    for position, frame in paired.groupby("position"):
        metrics = {}
        for model in MODELS:
            columns = ["actual", f"{model}_base", f"{model}_candidate"]
            values = frame[columns].apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(values).all(axis=1)
            pair = values[valid]
            item = {"n": len(pair), "unavailable": int((~valid).sum())}
            if not pair.empty:
                for arm in ("base", "candidate"):
                    prediction = pair[f"{model}_{arm}"]
                    item[arm] = {
                        **{
                            k: v
                            for k, v in compute_metrics(pair["actual"], prediction).items()
                            if k != "r2"
                        },
                        "bias": float((prediction - pair["actual"]).mean()),
                    }
                item["delta"] = {
                    key: item["candidate"][key] - item["base"][key]
                    for key in ("mae", "rmse", "bias")
                }
            metrics[model] = item
        output["positions"][position] = metrics
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--actuals",
        required=True,
        help="Parquet with player/week/position and observed raw target components",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--scoring", choices=("ppr", "half_ppr", "standard"), default="ppr")
    args = parser.parse_args()
    baseline = list(load_snapshots(args.baseline_dir))
    candidate = list(load_snapshots(args.candidate_dir))
    actuals = pd.read_parquet(args.actuals)
    result = {
        str(hours): compare_cutoff(
            select_cutoff_rows(baseline, hours=hours, scoring=args.scoring),
            select_cutoff_rows(candidate, hours=hours, scoring=args.scoring),
            actuals,
            scoring=args.scoring,
        )
        for hours in (48, 24)
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
