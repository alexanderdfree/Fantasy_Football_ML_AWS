"""Declared per-game rewrites of sampled donor histories (ADR-0029 slices 2 and 3).

An op names the columns it changes. The position schema declares which
columns may be rewritten, which are opaque external signals governed only by
the recipe's ``opaque_signal_policy``, which team totals move with a player
stat, and which game context is held. Relations are checked on the exact
rewritten values before counts are rounded, so an extreme factor fails loudly
instead of being silently capped; pure rounding artifacts are capped at their
bound and counted. A transformed history is a fixture: it has no observed
outcome and never claims one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from src.analysis.synthetic_history_schema import PositionHistorySchema
from src.shared.aggregate_targets import predictions_to_fantasy_points

TRANSFORM_CONTRACT_VERSION = 1
OPAQUE_POLICIES = ("keep_donor", "mark_missing")
COUNT_ROUNDING = (
    "half_to_even (numpy.rint); dependents capped at their rounded bound only for "
    "rounding artifacts"
)
TEAM_ACCOUNTING_NOTE = (
    "team totals move by the player delta; PAT and two-point plays are not modeled"
)


class TransformUnsupported(ValueError):
    """The position schema declines this op with a stated reason."""


@dataclass(frozen=True)
class ScaleOp:
    stats: tuple[str, ...]
    factor: float
    steps: tuple[int, int] | None = None
    op: str = "scale"


@dataclass(frozen=True)
class SetHistoryPpgOp:
    stats: tuple[str, ...]
    target_ppg: float
    op: str = "set_history_ppg"


OPS = {"scale": ScaleOp, "set_history_ppg": SetHistoryPpgOp}


def scoring_weights(schema: PositionHistorySchema) -> dict[str, float]:
    """Per-target points per unit, probed from the shared scoring function."""
    weights = {}
    for target in schema.targets:
        probe = {name: np.zeros(1) for name in schema.targets}
        probe[target] = np.ones(1)
        weights[target] = float(predictions_to_fantasy_points(schema.position, probe)[0])
    return weights


def _finite_number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value) + 0.0


def _stats(value, schema: PositionHistorySchema, op: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{op} requires a nonempty stats list")
    stats = tuple(value)
    if len(set(stats)) != len(stats) or not all(isinstance(s, str) for s in stats):
        raise ValueError(f"{op} stats must be unique column names")
    for stat in stats:
        if stat in schema.opaque_columns:
            raise ValueError(
                f"{stat} is an opaque external signal; it is governed by "
                f"opaque_signal_policy, not by {op}"
            )
        if stat in schema.team_accounting_columns:
            raise ValueError(f"{stat} is a team-accounting column; it moves with the player stat")
        if stat in schema.held_context_columns:
            raise ValueError(f"{stat} is donor game context and is held fixed")
        if stat not in schema.transformable_columns:
            raise ValueError(f"{stat} is not a transformable {schema.position} history column")
    return stats


def parse_transform(value, schema: PositionHistorySchema, history_games: int):
    """Turn a recipe entry (JSON object or op) into a validated frozen op."""
    if isinstance(value, tuple(OPS.values())):
        value = asdict(value)
    if not isinstance(value, dict) or "op" not in value:
        raise ValueError("each transform must be an object with an op")
    op = value["op"]
    if op not in OPS:
        raise ValueError(f"unknown transform op {op!r}; expected one of {sorted(OPS)}")
    reason = schema.transform_support.get(op, f"{op} is not declared for {schema.position}")
    if reason is not None:
        raise TransformUnsupported(f"{schema.position} does not support {op}: {reason}")
    unknown = set(value) - {f.name for f in fields(OPS[op])}
    if unknown:
        raise ValueError(f"unknown {op} fields: {sorted(unknown)}")
    stats = _stats(value.get("stats"), schema, op)
    if op == "scale":
        factor = _finite_number(value.get("factor"), "factor")
        if factor <= 0:
            raise ValueError("factor must be positive; a zero-usage game is a different archetype")
        steps = value.get("steps")
        if steps is not None:
            if (
                not isinstance(steps, (list, tuple))
                or len(steps) != 2
                or any(type(s) is not int for s in steps)
                or not 1 <= steps[0] <= steps[1] <= history_games
            ):
                raise ValueError("steps must be [from, to] within 1..history_games")
            steps = (steps[0], steps[1])
        return ScaleOp(stats=stats, factor=factor, steps=steps)
    target_ppg = _finite_number(value.get("target_ppg"), "target_ppg")
    if not any(stat in schema.targets for stat in stats):
        raise ValueError("set_history_ppg must scale at least one scoring target")
    return SetHistoryPpgOp(stats=stats, target_ppg=target_ppg)


def _targeted(frame: pd.DataFrame, op) -> pd.Series:
    steps = getattr(op, "steps", None)
    if steps is None:
        return pd.Series(True, index=frame.index)
    return frame["history_step"].between(steps[0], steps[1])


def relation_violations(frame: pd.DataFrame, schema: PositionHistorySchema) -> list[str]:
    """Human-readable counts of every violated relation or derived check."""
    violations = []
    for smaller, larger in schema.relations:
        count = int((frame[smaller] > frame[larger]).sum())
        if count:
            violations.append(f"{smaller} <= {larger} in {count} rows")
    for name, violated in schema.derived_checks:
        count = int(violated(frame).sum())
        if count:
            violations.append(f"{name} in {count} rows")
    return violations


def _ppg_factors(frame: pd.DataFrame, op: SetHistoryPpgOp, weights: dict[str, float]):
    """Per-case factor so the window's mean projected points hit the target."""
    zero = pd.Series(0.0, index=frame.index)
    listed = sum((weights[t] * frame[t] for t in op.stats if t in weights), zero)
    unlisted = sum((weights[t] * frame[t] for t in weights if t not in op.stats), zero)
    per_case = pd.DataFrame({"listed": listed, "unlisted": unlisted, "case_id": frame["case_id"]})
    means = per_case.groupby("case_id", sort=False)[["listed", "unlisted"]].mean()
    factors = (op.target_ppg - means["unlisted"]) / means["listed"]
    bad = means[(means["listed"] <= 0) | ~(factors > 0)]
    if not bad.empty:
        case, row = next(iter(bad.iterrows()))
        raise ValueError(
            f"target_ppg {op.target_ppg} is not reachable by scaling {list(op.stats)} for case "
            f"{case}: listed components contribute {row['listed']:.3g}, unlisted "
            f"{row['unlisted']:.3g}"
        )
    return frame["case_id"].map(factors), factors


def apply_transforms(
    games: pd.DataFrame, ops, *, schema: PositionHistorySchema, policy: str
) -> tuple[pd.DataFrame, dict]:
    """Rewrite sampled games in place of the donor values; return games and a report.

    ``games`` carries one row per case and history step with the schema's
    history columns and targets as float64. Ops apply in order to every case.
    """
    if policy not in OPAQUE_POLICIES:
        raise ValueError(f"opaque_signal_policy must be one of {OPAQUE_POLICIES}")
    weights = scoring_weights(schema)
    frame = games.copy()
    frame["transformed"] = False
    report_ops = []
    for number, op in enumerate(ops, start=1):
        targeted = _targeted(frame, op)
        stats = list(op.stats)
        ppg_block = None
        if op.op == "scale":
            factors = pd.Series(op.factor, index=frame.index)
        else:
            factors, per_case = _ppg_factors(frame, op, weights)
            ppg_block = {"requested": op.target_ppg, "factors": per_case.round(6).to_dict()}
        before = frame.loc[targeted, stats]
        exact = before.mul(factors[targeted], axis=0)
        trial = frame.copy()
        trial.loc[targeted, stats] = exact
        # Team totals follow the exact player deltas before any relation is judged.
        for column in stats:
            if column in schema.team_accounting:
                team_column, coefficient = schema.team_accounting[column]
                trial.loc[targeted, team_column] += coefficient * (exact[column] - before[column])
        violations = relation_violations(trial, schema)
        if violations:
            raise ValueError(
                f"transformed history violates {'; '.join(violations)} after op {number} "
                f"({op.op} on {stats}); a production factor must be matched by usage or "
                "lowered. Synthetic histories never relax relations."
            )
        rounded = exact.copy()
        cells_rounded = 0
        for column in stats:
            if column in schema.count_columns:
                values = np.rint(exact[column])
                cells_rounded += int((values != exact[column]).sum())
                rounded[column] = values
        trial.loc[targeted, stats] = rounded
        # Re-derive the team totals from the rounded deltas; two stats may feed one column.
        touched = {schema.team_accounting[c][0] for c in stats if c in schema.team_accounting}
        for team_column in touched:
            trial.loc[targeted, team_column] = frame.loc[targeted, team_column]
        for column in stats:
            if column in schema.team_accounting:
                team_column, coefficient = schema.team_accounting[column]
                trial.loc[targeted, team_column] += coefficient * (rounded[column] - before[column])
        cells_capped = 0
        for smaller, larger in schema.relations:
            if smaller in stats:
                bound = trial.loc[targeted, larger]
                over = rounded[smaller] > bound
                cells_capped += int(over.sum())
                rounded.loc[over, smaller] = bound[over]
        for column, bound_fn in schema.derived_caps:
            if column in stats:
                bound = bound_fn(trial).loc[targeted]
                over = rounded[column] > bound
                cells_capped += int(over.sum())
                rounded.loc[over, column] = bound[over]
        cells_clamped = 0
        for column, low, high in schema.bounded_columns:
            if column in stats:
                clipped = rounded[column].clip(low, high)
                cells_clamped += int((clipped != rounded[column]).sum())
                rounded[column] = clipped
        accounting = []
        for column in stats:
            if column in schema.team_accounting:
                team_column, coefficient = schema.team_accounting[column]
                frame.loc[targeted, team_column] += coefficient * (rounded[column] - before[column])
                accounting.append(f"{team_column} += {coefficient:g} * delta {column}")
        frame.loc[targeted, stats] = rounded
        frame.loc[targeted, "transformed"] = True
        entry = {
            "op": asdict(op),
            "fields_changed": stats,
            "derived_recomputed": ["fantasy_points"],
            "team_accounting": accounting,
            "rows_targeted": int(targeted.sum()),
            "cells_rounded": cells_rounded,
            "cells_capped_by_relation": cells_capped,
            "cells_clamped": cells_clamped,
        }
        if ppg_block is not None:
            entry["ppg_target"] = ppg_block
        report_ops.append(entry)
    marked = 0
    if policy == "mark_missing":
        rows = frame["transformed"]
        for column in schema.opaque_columns:
            marked += int(rows.sum() - frame.loc[rows, column].isna().sum())
            frame.loc[rows, column] = np.nan
    frame["fantasy_points"] = predictions_to_fantasy_points(
        schema.position, {t: frame[t].to_numpy() for t in schema.targets}
    )
    report = {
        "transform_contract_version": TRANSFORM_CONTRACT_VERSION,
        "transforms": report_ops,
        "opaque_signal_policy": policy,
        "opaque_columns": list(schema.opaque_columns),
        "opaque_cells_marked_missing": marked,
        "count_rounding": COUNT_ROUNDING,
        "team_accounting_note": TEAM_ACCOUNTING_NOTE,
        "scoring_weights": weights,
    }
    return frame, report
