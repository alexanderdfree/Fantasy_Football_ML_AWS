"""Stage-2/3 orchestration for the staged feature-selection system.

Stage 1 ([ab_feature_screen*.py](ab_feature_screen.py)) ranks whole families and
writes per-position reports to ``todo/feature_selection/{pos}.json``. This module is
the orchestration that turns those into a smooth Stage-2 (sub-family zoom) + Stage-3
(combined-drop confirm) workflow, wired into the [feature_selection.py](feature_selection.py)
driver as the ``substage`` / ``substage-report`` / ``confirm`` / ``confirm-report``
subcommands. It is pure planning + S3 post-processing — it **never trains and never
fires a Batch job**; it prints the exact ``launch_ab`` commands a human runs later.

Four pieces (the 4 subcommands map 1:1):

  1. SELECT      — :func:`select_subscreens` reads a Stage-1 ``{pos}.json`` and
     auto-picks the ``(position, family)`` sub-screens worth running (Comprehensive:
     every decomposable family that is a drop-candidate/borderline OR large/
     heterogeneous; skip atomic + clean all-signal KEEP).
  2. LAUNCH      — :func:`subscreen_launch_command` / :func:`smoke_command` print the
     exact ``launch_ab`` lines (run-id, cells, cost, ``--max-cells``, stacked flags);
     the ``substage`` subcommand prints them + writes a ``plan.json``.
  3. REPORT      — :func:`build_subscreen_spec` + :func:`collect_effects` collect each
     sub-screen run from S3 and :func:`write_stage2_report` consolidates them into one
     per-position report (per-model MAE+RMSE, same shape as Stage 1).
  4. CONFIRM     — :func:`confirm_launch_command` re-runs a chosen drop-set TOGETHER on
     the **production PCA-Ridge** config ([ab_feature_confirm.py](ab_feature_confirm.py));
     :func:`build_confirm_spec` collects it for a faithful final decision.

Two correctness invariants baked in (see the module's tests):

  * **Seeds are always explicit.** ``ab_batch`` resolves seeds with no stacked default
    and the sub-screen spec declares no ``SEEDS``, so a stacked run launched without
    ``--seeds`` would silently fall back to 3 seeds. Every command passes ``--seeds
    42..65 --stacked-seeds`` (skill) / ``--seeds 42..49`` (K/DST eager), which also
    makes cell keys deterministic for collection.
  * **No env-import for collection.** The sub-screen/confirm specs build their groups
    from env AT IMPORT, so collecting several families in one process via
    ``resolve_spec(<dotted>)`` would pin to the first. :func:`build_subscreen_spec` /
    :func:`build_confirm_spec` construct the ``Spec`` directly from
    :mod:`feature_groups` instead.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning.feature_groups import (  # noqa: E402
    build_confirm_variants,
    build_drop_variants,
    subfamily_groups,
)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SPECIAL_POSITIONS = ("K", "DST")

SUBSCREEN_SPEC = "src.tuning.ab_feature_subscreen"
CONFIRM_SPEC = "src.tuning.ab_feature_confirm"
STAGE1_DIR = "todo/feature_selection"
STAGE2_DIR = "todo/feature_selection/stage2"
PLAN_FILE = "plan.json"
CONFIRM_PLAN_FILE = "confirm_plan.json"

# Skill families always worth zooming even on a family-level KEEP: large +
# heterogeneous (so the family verdict can hide a droppable sub-group), plus the
# cross-position drop-candidate ``trend``.
ALWAYS_ZOOM_SKILL = frozenset({"rolling", "prior_season", "specific", "trend"})
# A K/DST partition this wide is "large/heterogeneous" enough to zoom regardless of
# the (3-seed-noisy) family verdict — the 8-seed sub-screen gives a cleaner read.
KDST_MIN_COLS_TO_ZOOM = 4
# Seed widths. Skill stacks 24 (the measured per-seed optimum, ab_ensemble_seeds);
# K/DST can't vmap-stack, so they screen eager — bumped to 8 (Stage-1's 3-seed
# mid-tier is ~1σ from zero) to clear the 0.02 FP-MAE noise floor.
DEFAULT_STACKED_N = 24
DEFAULT_KDST_N = 8

# Rough cost preview (a gut-check, NOT a billing oracle). launch_ab's own docstring
# estimates ~3-8 min/cell on g6/L4 Spot for the full skill pipeline; K/DST cells are
# lighter (less data, fewer features), and a stacked group is ~one cell's non-attention
# cost + a 24-seed vmap attention bundle. Treat the dollar figures as ±2x.
SPOT_USD_PER_HR = 0.70
EAGER_CELL_MIN = 3.0
STACKED_GROUP_MIN = 6.0
# An 8-seed K/DST eager sub-screen is ~100 sequential cells, which can exceed
# launch_ab's default 3h attemptDurationSeconds. Bake a generous single-attempt cap
# into the eager commands (per-cell S3 checkpointing resumes anything beyond it on a
# retry); stacked skill jobs are short, so they keep the launch_ab default.
EAGER_ATTEMPT_TIMEOUT_S = 21600  # 6h


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
@dataclass
class FamilyPick:
    """One selected ``(position, family)`` sub-screen + everything needed to launch
    and later collect it."""

    position: str
    family: str
    group_cols: dict[str, frozenset[str]]
    n_subgroups: int
    n_variants: int
    seeds: list[int]
    stacked: bool
    verdict: str
    reason: str
    riskiest_variant: str
    run_id: str = ""

    @property
    def cells(self) -> int:
        return self.n_variants * len(self.seeds)


def _stacked_seed_list(n: int) -> list[int]:
    """The canonical n-seed stacked grid (``42..42+n``) the container also uses."""
    from src.tuning.ab_ensemble_seeds import stacked_default_seed_list

    return stacked_default_seed_list(n)


def _eager_seed_list(n: int) -> list[int]:
    """A deterministic n-seed eager grid (42-based, matching the stacked convention)."""
    return list(range(42, 42 + n))


def _stage1_families(payload: dict) -> list[str]:
    """Family/group names present in a Stage-1 payload's ``effects`` (any model)."""
    out: set[str] = set()
    for by_metric in payload.get("effects", {}).values():
        out.update(by_metric.get("mae", {}))
    return sorted(out)


def _should_zoom(
    position: str,
    family: str,
    verdict: str,
    suggested: set[str],
    group_cols: dict[str, frozenset[str]],
) -> tuple[bool, str]:
    """The Comprehensive selection rule for a DECOMPOSABLE family. Returns
    ``(select, reason)``."""
    if position in SKILL_POSITIONS and family in ALWAYS_ZOOM_SKILL:
        return True, f"large/heterogeneous, always-zoom ({len(group_cols)} sub-groups)"
    if family in suggested:
        return True, f"drop-candidate (in suggested_cut, verdict={verdict})"
    if verdict in ("DROP-CAND", "MIXED"):
        return True, f"borderline (verdict={verdict})"
    if position in SPECIAL_POSITIONS:
        n_cols = sum(len(c) for c in group_cols.values())
        if n_cols >= KDST_MIN_COLS_TO_ZOOM:
            return True, f"large K/DST partition ({n_cols} cols, verdict={verdict} is 3-seed-noisy)"
    return False, f"all-signal KEEP (verdict={verdict})"


def _pick_priority(pick: FamilyPick) -> tuple:
    """Sort key: actionable drop-candidates first, then the widest families."""
    rank = {"DROP-CAND": 0, "MIXED": 1, "KEEP": 2}.get(pick.verdict, 3)
    return (rank, -pick.n_subgroups, pick.family)


def select_subscreens(
    position: str,
    stage1_payload: dict,
    *,
    stacked_n: int = DEFAULT_STACKED_N,
    kdst_n: int = DEFAULT_KDST_N,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    max_families: int | None = None,
) -> tuple[list[FamilyPick], list[dict]]:
    """Auto-pick the sub-screens worth running for one position from its Stage-1
    ``{pos}.json`` payload. Returns ``(picks, skipped)`` where ``skipped`` is a list
    of ``{position, family, reason}`` dicts. ``only`` / ``skip`` are operator
    overrides; ``max_families`` caps the per-position fan-out (lowest-priority
    families spill into ``skipped``)."""
    from src.tuning import feature_selection as fs

    pos = position.upper()
    stacked = pos in SKILL_POSITIONS
    seeds = _stacked_seed_list(stacked_n) if stacked else _eager_seed_list(kdst_n)
    effects = stage1_payload.get("effects", {})
    suggested = set(stage1_payload.get("suggested_cut", []))
    only_set = {f for f in (only or [])}
    skip_set = {f for f in (skip or [])}

    picks: list[FamilyPick] = []
    skipped: list[dict] = []

    def _skip(family: str, reason: str) -> None:
        skipped.append({"position": pos, "family": family, "reason": reason})

    for family in _stage1_families(stage1_payload):
        if only_set and family not in only_set:
            _skip(family, "not in --only-family")
            continue
        if family in skip_set:
            _skip(family, "excluded via --skip-family")
            continue
        group_cols = subfamily_groups(pos, family)
        n_sub = len(group_cols)
        verdict = fs._verdict(effects, family, noise=fs.NOISE_FLOOR)
        if n_sub < 2:
            # Nothing finer to resolve: an atomic KEEP is "all-signal", an atomic
            # drop-candidate is "dead-as-a-block" (apply the whole column wholesale).
            if verdict == "DROP-CAND":
                _skip(family, "dead-as-a-block, atomic (apply the column wholesale)")
            elif verdict == "KEEP":
                _skip(family, "all-signal, atomic (1 sub-group)")
            else:
                _skip(family, f"atomic ({n_sub} sub-group, cannot decompose)")
            continue
        select, reason = _should_zoom(pos, family, verdict, suggested, group_cols)
        if not select:
            _skip(family, reason)
            continue
        variants, row_drops = build_drop_variants(group_cols)
        riskiest = (
            max(row_drops, key=lambda v: len(row_drops[v])) if row_drops else variants[-1].name
        )
        picks.append(
            FamilyPick(
                position=pos,
                family=family,
                group_cols=group_cols,
                n_subgroups=n_sub,
                n_variants=len(variants),
                seeds=list(seeds),
                stacked=stacked,
                verdict=verdict,
                reason=reason,
                riskiest_variant=riskiest,
            )
        )

    picks.sort(key=_pick_priority)
    if max_families is not None and len(picks) > max_families:
        for p in picks[max_families:]:
            _skip(p.family, f"over --max-families {max_families} (lower priority)")
        picks = picks[:max_families]
    return picks, skipped


# --------------------------------------------------------------------------- #
# Cost + run-ids + launch commands
# --------------------------------------------------------------------------- #
def estimate_cost(pick: FamilyPick) -> float:
    """Rough USD for one sub-screen Batch job (sequential cells/groups in-container).

    Stacked skill trains ~``n_variants`` vmap groups; eager K/DST runs every cell."""
    minutes = pick.n_variants * STACKED_GROUP_MIN if pick.stacked else pick.cells * EAGER_CELL_MIN
    return round(minutes / 60.0 * SPOT_USD_PER_HR, 2)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def assign_run_ids(
    picks: list[FamilyPick], *, image_sha: str | None = None, stamp: str | None = None
) -> str:
    """Stamp each pick with a deterministic, recorded run-id. Returns the stamp used
    (one per ``substage`` invocation, so a position's families share a namespace)."""
    stamp = stamp or _utc_stamp()
    sha7 = (image_sha or "local")[:7]
    for p in picks:
        p.run_id = f"subscreen-{p.position.lower()}-{p.family}-{stamp}-{sha7}"
    return stamp


def _cmd(parts: list[str]) -> str:
    return " ".join(p for p in parts if p)


def split_env_prefixed_command(cmd: str) -> tuple[dict[str, str], list[str]]:
    """Split a generated command's leading ``KEY=VAL`` env-prefix from its argv.

    The printed commands start with a local env-prefix (``FF_SUBSCREEN_POSITION=RB
    ... python -m ...``) that a shell interprets but ``subprocess.run(cmd.split())``
    would not (it would try to exec ``FF_SUBSCREEN_POSITION=RB`` as the program). The
    ``--exec`` path uses this to run them faithfully: leading ``KEY=VAL`` tokens (valid
    identifier keys, not flags) become env overrides; the rest is the argv."""
    tokens = cmd.split()
    env: dict[str, str] = {}
    i = 0
    while (
        i < len(tokens)
        and not tokens[i].startswith("-")
        and "=" in tokens[i]
        and tokens[i].split("=", 1)[0].isidentifier()
    ):
        key, _, value = tokens[i].partition("=")
        env[key] = value
        i += 1
    return env, tokens[i:]


def subscreen_launch_command(pick: FamilyPick, *, image_sha: str | None = None) -> str:
    """The exact ``launch_ab`` command for one sub-screen (explicit seeds + stacked +
    its own run-id + a ``--max-cells`` cap matching the grid).

    The ``FF_SUBSCREEN_*`` env is set BOTH locally (the leading ``VAR=val`` prefix) and
    via ``--env``: ``--env`` only reaches the Batch container, but launch_ab's own
    submitter imports the spec to size ``--max-cells`` / the manifest / its ``--wait``
    collection, and the spec reads its groups from the LOCAL env at import — without the
    prefix the submitter would resolve the default (RB/rolling) grid (a real mismatch for
    leave-one-out families, whose ``drop_*`` variant names differ from the default's PB
    ``pb*``)."""
    return _cmd(
        [
            f"FF_SUBSCREEN_POSITION={pick.position} FF_SUBSCREEN_FAMILY={pick.family} "
            "python -m src.tuning.launch_ab",
            f"--spec {SUBSCREEN_SPEC}",
            f"--positions {pick.position}",
            f"--env FF_SUBSCREEN_POSITION={pick.position}",
            f"--env FF_SUBSCREEN_FAMILY={pick.family}",
            f"--seeds {' '.join(str(s) for s in pick.seeds)}",
            "--stacked-seeds" if pick.stacked else "",
            f"--run-id {pick.run_id}",
            f"--max-cells {pick.cells}",
            "" if pick.stacked else f"--attempt-timeout {EAGER_ATTEMPT_TIMEOUT_S}",
            f"--image-sha {image_sha}" if image_sha else "",
        ]
    )


def smoke_command(pick: FamilyPick, *, image_sha: str | None = None) -> str:
    """Smoke ONE real Batch cell first: the riskiest arm (most sub-groups dropped) at
    a single eager seed (baseline is always kept, so 2 cells). Degenerate arms only
    crash live — ``--list`` / unit tests validate grid construction, not the pipeline
    (#1187 -> #1212). The feature/Ridge crash path is identical eager vs stacked, so a
    1-seed eager smoke is the cheap, sufficient guard."""
    return _cmd(
        [
            f"FF_SUBSCREEN_POSITION={pick.position} FF_SUBSCREEN_FAMILY={pick.family} "
            "python -m src.tuning.launch_ab",
            f"--spec {SUBSCREEN_SPEC}",
            f"--positions {pick.position}",
            f"--env FF_SUBSCREEN_POSITION={pick.position}",
            f"--env FF_SUBSCREEN_FAMILY={pick.family}",
            f"--only {pick.riskiest_variant}",
            f"--seeds {pick.seeds[0]}",
            f"--run-id {pick.run_id}-smoke",
            "--max-cells 2",
            f"--image-sha {image_sha}" if image_sha else "",
        ]
    )


def confirm_regime(
    position: str,
    *,
    eager: bool = False,
    stacked_n: int = DEFAULT_STACKED_N,
    kdst_n: int = DEFAULT_KDST_N,
) -> tuple[bool, list[int]]:
    """``(stacked, seeds)`` for a Stage-3 confirm. Skill stacks 24 (production-faithful
    Ridge/LGBM on PCA-Ridge; ``--eager`` forces the faithful-attention eager-8 path);
    K/DST always eager 8 (can't vmap-stack)."""
    pos = position.upper()
    stacked = (pos in SKILL_POSITIONS) and not eager
    seeds = _stacked_seed_list(stacked_n) if stacked else _eager_seed_list(kdst_n)
    return stacked, seeds


def confirm_launch_command(
    position: str,
    drop_cols: list[str],
    *,
    eager: bool = False,
    image_sha: str | None = None,
    run_id: str | None = None,
    stacked_n: int = DEFAULT_STACKED_N,
    kdst_n: int = DEFAULT_KDST_N,
) -> str:
    """The exact ``launch_ab`` command for a Stage-3 confirm (combined drop-set on the
    production PCA-Ridge config)."""
    pos = position.upper()
    stacked, seeds = confirm_regime(pos, eager=eager, stacked_n=stacked_n, kdst_n=kdst_n)
    run_id = run_id or f"confirm-{pos.lower()}"
    cols = ",".join(sorted(drop_cols))
    return _cmd(
        [
            # local env (for the submitter's resolve_spec) + --env (for the container);
            # see subscreen_launch_command for why both are needed.
            f"FF_CONFIRM_POSITION={pos} FF_CONFIRM_DROP_COLS={cols} python -m src.tuning.launch_ab",
            f"--spec {CONFIRM_SPEC}",
            f"--positions {pos}",
            f"--env FF_CONFIRM_POSITION={pos}",
            f"--env FF_CONFIRM_DROP_COLS={cols}",
            f"--seeds {' '.join(str(s) for s in seeds)}",
            "--stacked-seeds" if stacked else "",
            f"--run-id {run_id}",
            f"--max-cells {2 * len(seeds)}",
            f"--image-sha {image_sha}" if image_sha else "",
        ]
    )


def confirm_smoke_command(
    position: str, drop_cols: list[str], *, image_sha: str | None = None, run_id: str | None = None
) -> str:
    """Smoke the confirm's drop arm (production PCA-Ridge) at one seed before the
    high-seed fan-out — a drop below ``ridge_pca_components`` would crash PCA live."""
    pos = position.upper()
    run_id = (run_id or f"confirm-{pos.lower()}") + "-smoke"
    cols = ",".join(sorted(drop_cols))
    return _cmd(
        [
            f"FF_CONFIRM_POSITION={pos} FF_CONFIRM_DROP_COLS={cols} python -m src.tuning.launch_ab",
            f"--spec {CONFIRM_SPEC}",
            f"--positions {pos}",
            f"--env FF_CONFIRM_POSITION={pos}",
            f"--env FF_CONFIRM_DROP_COLS={cols}",
            "--only drop_confirmed",
            "--seeds 42",
            f"--run-id {run_id}",
            "--max-cells 2",
            f"--image-sha {image_sha}" if image_sha else "",
        ]
    )


# --------------------------------------------------------------------------- #
# Spec construction (direct, NO env-module import) + collection
# --------------------------------------------------------------------------- #
def build_subscreen_spec(position: str, family: str, seeds: list[int]):
    """A collection ``Spec`` for one sub-screen built directly from
    :mod:`feature_groups` — never importing the env-parametrized
    :mod:`ab_feature_subscreen` (whose groups are pinned at import, so a multi-family
    report in one process would collect the wrong cells). Returns
    ``(spec, group_cols, row_drops)``."""
    from collections import OrderedDict

    from src.tuning.ab_harness import Spec, default_metric_fn

    pos = position.upper()
    group_cols = subfamily_groups(pos, family)
    if not group_cols:
        raise ValueError(f"no sub-groups for position={pos!r} family={family!r}")
    variants, row_drops = build_drop_variants(group_cols)
    spec = Spec(
        variants=OrderedDict((v.name, v) for v in variants),
        baseline="baseline",
        positions=[pos],
        seeds=[int(s) for s in seeds],
        metric_fn=default_metric_fn,
        dotted=SUBSCREEN_SPEC,
        name=f"subscreen_{pos}_{family}",
    )
    return spec, group_cols, row_drops


def build_confirm_spec(position: str, drop_cols: list[str], seeds: list[int]):
    """A collection ``Spec`` for a Stage-3 confirm, built directly from
    :func:`feature_groups.build_confirm_variants` (production PCA-Ridge baseline vs the
    combined-drop arm). Returns ``(spec, group_names, row_drops)`` with the single
    synthetic group ``confirmed_drop``."""
    from collections import OrderedDict

    from src.tuning.ab_harness import Spec, default_metric_fn

    pos = position.upper()
    variants, row_drops = build_confirm_variants(frozenset(drop_cols))
    spec = Spec(
        variants=OrderedDict((v.name, v) for v in variants),
        baseline="baseline",
        positions=[pos],
        seeds=[int(s) for s in seeds],
        metric_fn=default_metric_fn,
        dotted=CONFIRM_SPEC,
        name=f"confirm_{pos}",
    )
    return spec, ["confirmed_drop"], row_drops


def collect_effects(
    spec,
    position: str,
    group_names: list[str],
    row_drops: dict,
    run_id: str,
    *,
    s3_prefix: str = "ab_runs",
) -> dict:
    """Download a run's per-cell JSONs from S3 and compute per-model MAE+RMSE effects
    (reuses ``launch_ab.collect_results`` + ``feature_selection.position_effects``)."""
    import boto3

    from src.batch.launch import AWS_REGION, S3_BUCKET
    from src.tuning import feature_selection as fs
    from src.tuning.launch_ab import collect_results

    s3 = boto3.client("s3", region_name=AWS_REGION)
    results = collect_results(
        spec, bucket=S3_BUCKET, s3_prefix=s3_prefix, run_id=run_id, s3_client=s3
    )
    return fs.position_effects(results, position.upper(), group_names, row_drops)


# --------------------------------------------------------------------------- #
# Consolidated Stage-2 report
# --------------------------------------------------------------------------- #
def combined_drop_cols(family_reports: list[dict]) -> list[str]:
    """Union of each family's suggested sub-cut (sub-groups neutral-or-helpful for
    every model), mapped back to columns — the combined drop-set to confirm."""
    from src.tuning import feature_selection as fs

    cols: set[str] = set()
    for fr in family_reports:
        sub_cut = fs.suggested_cut(fr["effects"], list(fr["group_cols"]))
        for grp in sub_cut:
            cols.update(fr["group_cols"].get(grp, ()))
    return sorted(cols)


def _family_section(position: str, fr: dict, *, noise: float) -> list[str]:
    """One family's H2 sub-section: per-sub-group verdict + per-model MAE|RMSE."""
    from src.tuning import feature_selection as fs

    effects = fr["effects"]
    group_cols = fr["group_cols"]
    group_names = sorted(group_cols)
    models = list(effects)
    lines = [f"## `{fr['family']}` sub-groups", ""]
    lines.append(f"- run-id: `{fr['run_id']}`  seeds: {fr['seeds']}")
    sub_cut = fs.suggested_cut(effects, group_names, noise=noise)
    drop_cols = sorted({c for g in sub_cut for c in group_cols.get(g, ())})
    lines.append(
        f"- suggested sub-cut: {', '.join(f'`{g}`' for g in sub_cut) or '_none_'}"
        + (f"  -> columns: {', '.join(f'`{c}`' for c in drop_cols)}" if drop_cols else "")
    )
    lines.append("")
    if not models:
        lines += ["_no cells collected for this family yet._", ""]
        return lines
    header = ["sub-group", "verdict"]
    for m in models:
        header += [f"{m} MAE", f"{m} RMSE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def _sort_key(grp: str) -> float:
        maes = [
            bm["mae"][grp]["mean_effect"] for bm in effects.values() if grp in bm.get("mae", {})
        ]
        return min(maes) if maes else 0.0

    for grp in sorted(group_names, key=_sort_key):
        row = [f"`{grp}`", fs._verdict(effects, grp, noise=noise)]
        for m in models:
            row.append(fs._fmt_effect(effects[m].get("mae", {}).get(grp)))
            row.append(fs._fmt_effect(effects[m].get("rmse", {}).get(grp)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _stage2_caveats() -> list[str]:
    return [
        "## Caveats",
        "",
        "- **Sign:** `+` = dropping the sub-group RAISES error = it carries signal (keep). "
        "`-` = drop candidate.",
        "- **Skip-PCA screen.** Ridge here runs on raw features (PCA off) for clean "
        "attribution; production RB/WR/DST ship PCA-Ridge. The combined drop-set MUST be "
        "confirmed on the production config (Stage 3) before `apply`.",
        "- **Stacked vs eager / subgroup bias.** Skill stacks (vmap, FP32/LN/fixed-epochs); "
        "K/DST eager. Judge borderline sub-groups by subgroup *bias*, not overall MAE, and "
        "confirm the combined drop at high seed count — PB main effects assume additivity.",
        "",
    ]


def render_stage2_md(
    position: str, family_reports: list[dict], *, noise: float | None = None
) -> str:
    """Consolidated per-position Stage-2 markdown: combined drop-set + Stage-3 confirm /
    apply commands, then a per-family sub-group table, then shared caveats."""
    from src.tuning import feature_selection as fs

    noise = fs.NOISE_FLOOR if noise is None else noise
    combined = combined_drop_cols(family_reports)
    lines = [f"# Feature-selection Stage-2 (sub-family zoom) — {position.upper()}", ""]
    fams = ", ".join(f"`{fr['family']}`" for fr in family_reports) or "_none_"
    lines.append(f"- families zoomed: {fams}")
    lines.append(f"- noise floor: {noise} FP (AGENTS.md)")
    lines.append("")
    lines.append("## Combined suggested drop columns (review — not auto-applied)")
    lines.append("")
    if combined:
        lines.append(
            "Union of every zoomed family's suggested sub-cut (neutral-or-helpful for "
            "**every** model). CONFIRM them together on the production PCA-Ridge config "
            "(Stage 3) before applying — the screen is skip-PCA and PB assumes additivity:"
        )
        lines.append("")
        lines += [f"- `{c}`" for c in combined]
        lines += [
            "",
            "Confirm (Stage 3 — production PCA-Ridge, high seed count):",
            "",
            "```",
            f"python -m src.tuning.feature_selection confirm --position {position.upper()} --from-stage2",
            "```",
            "",
            "Then apply the cut YOU choose after a clean confirm (draft PR):",
            "",
            "```",
            f"python -m src.tuning.feature_selection apply --position {position.upper()} "
            f"--drop {' '.join(combined)} --pr",
            "```",
            "",
        ]
    else:
        lines += [
            "_No sub-group is neutral-or-helpful across all models — nothing suggested for "
            "removal. The zoomed families carry signal at the sub-group level._",
            "",
        ]
    for fr in family_reports:
        lines += _family_section(position, fr, noise=noise)
    lines += _stage2_caveats()
    return "\n".join(lines)


def write_stage2_report(
    position: str, family_reports: list[dict], *, out_dir: str = STAGE2_DIR
) -> tuple[str, str]:
    """Write ``{out_dir}/{pos}.md`` + ``{pos}.json`` for the consolidated Stage-2
    report; return the two paths."""
    from src.tuning import feature_selection as fs

    pos = position.upper()
    md = render_stage2_md(pos, family_reports)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{pos.lower()}.md"
    json_path = out / f"{pos.lower()}.json"
    md_path.write_text(md)
    families = {}
    for fr in family_reports:
        sub_cut = fs.suggested_cut(fr["effects"], list(fr["group_cols"]))
        families[fr["family"]] = {
            "run_id": fr["run_id"],
            "seeds": fr["seeds"],
            "spec": SUBSCREEN_SPEC,
            "group_cols": {g: sorted(c) for g, c in fr["group_cols"].items()},
            "effects": fr["effects"],
            "suggested_subcut": sub_cut,
            "suggested_drop_cols": sorted(
                {c for g in sub_cut for c in fr["group_cols"].get(g, ())}
            ),
        }
    payload = {
        "position": pos,
        "stage": 2,
        "noise_floor": fs.NOISE_FLOOR,
        "families": families,
        "combined_suggested_drop_cols": combined_drop_cols(family_reports),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return str(md_path), str(json_path)


# --------------------------------------------------------------------------- #
# Stage-3 confirm report
# --------------------------------------------------------------------------- #
def render_confirm_md(
    position: str, drop_cols: list[str], run_id: str, seeds: list[int], effects: dict
) -> str:
    """The combined-drop confirm report (production PCA-Ridge), per-model MAE+RMSE."""
    from src.tuning import feature_selection as fs

    pos = position.upper()
    lines = [f"# Feature-selection Stage-3 confirm — {pos}", ""]
    lines.append(f"- spec: `{CONFIRM_SPEC}`  run-id: `{run_id}`  seeds: {seeds}")
    lines.append(
        f"- columns dropped together ({len(drop_cols)}): {', '.join(f'`{c}`' for c in drop_cols)}"
    )
    lines.append(
        "- **Production config — PCA-Ridge ON.** This is the faithful gate the skip-PCA "
        "screen feeds; judge the combined drop here, not on the screen's raw-Ridge column."
    )
    lines.append(
        "- **Sign:** `+` = dropping the set RAISES error = the set carries signal (KEEP it). "
        f"`-`/flat (≤ {fs.NOISE_FLOOR} FP) = the combined drop is safe."
    )
    lines.append("")
    models = list(effects)
    grp = "confirmed_drop"
    lines.append("## Combined drop-set effect by model (MAE | RMSE)")
    lines.append("")
    lines.append("| model | verdict | MAE | RMSE |")
    lines.append("|---|---|---|---|")
    for m in models:
        v = fs._verdict({m: effects[m]}, grp, noise=fs.NOISE_FLOOR)
        lines.append(
            f"| {m} | {v} | {fs._fmt_effect(effects[m].get('mae', {}).get(grp))} "
            f"| {fs._fmt_effect(effects[m].get('rmse', {}).get(grp))} |"
        )
    lines.append("")
    overall = fs._verdict(effects, grp, noise=fs.NOISE_FLOOR)
    if overall == "DROP-CAND":
        lines.append(
            "**Confirmed:** the combined drop is neutral-or-helpful for every model on the "
            "production config — safe to `apply` (draft PR; fires a 6-position retrain — "
            "review the benchmark delta)."
        )
    elif overall == "KEEP":
        lines.append(
            "**Not confirmed:** dropping the set together RAISES error on production PCA-Ridge "
            "(an interaction the additive screen missed) — do NOT drop as a block."
        )
    else:
        lines.append(
            "**Mixed:** helps some models, hurts others on the production config — operator's "
            "call; consider a smaller subset and re-confirm."
        )
    lines.append("")
    return "\n".join(lines)


def write_confirm_report(
    position: str,
    drop_cols: list[str],
    run_id: str,
    seeds: list[int],
    effects: dict,
    *,
    out_dir: str = STAGE2_DIR,
) -> tuple[str, str]:
    """Write ``{out_dir}/{pos}.confirm.md`` + ``.confirm.json``; return the two paths."""
    from src.tuning import feature_selection as fs

    pos = position.upper()
    md = render_confirm_md(pos, drop_cols, run_id, seeds, effects)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{pos.lower()}.confirm.md"
    json_path = out / f"{pos.lower()}.confirm.json"
    md_path.write_text(md)
    payload = {
        "position": pos,
        "stage": 3,
        "spec": CONFIRM_SPEC,
        "run_id": run_id,
        "seeds": seeds,
        "drop_cols": sorted(drop_cols),
        "noise_floor": fs.NOISE_FLOOR,
        "verdict": fs._verdict(effects, "confirmed_drop", noise=fs.NOISE_FLOOR),
        "effects": effects,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return str(md_path), str(json_path)


# --------------------------------------------------------------------------- #
# Plan I/O (the orchestration source of truth)
# --------------------------------------------------------------------------- #
def pick_to_dict(pick: FamilyPick) -> dict:
    return {
        "position": pick.position,
        "family": pick.family,
        "run_id": pick.run_id,
        "spec": SUBSCREEN_SPEC,
        "seeds": pick.seeds,
        "stacked": pick.stacked,
        "n_subgroups": pick.n_subgroups,
        "n_variants": pick.n_variants,
        "cells": pick.cells,
        "verdict": pick.verdict,
        "reason": pick.reason,
        "riskiest_variant": pick.riskiest_variant,
        "est_cost_usd": estimate_cost(pick),
        "group_cols": {g: sorted(c) for g, c in pick.group_cols.items()},
    }


def write_plan(
    picks: list[FamilyPick],
    skipped: list[dict],
    *,
    out_path: str,
    image_sha: str | None = None,
    stamp: str | None = None,
) -> dict:
    """Write the substage ``plan.json`` — the source of truth ``substage-report`` reads
    (records each pick's run-id + seeds so collection finds the right cells regardless
    of the launcher manifest)."""
    plan = {
        "stage": 2,
        "created_at": stamp or datetime.now(UTC).isoformat(),
        "image_sha": image_sha,
        "picks": [pick_to_dict(p) for p in picks],
        "skipped": skipped,
        "total_est_cost_usd": round(sum(estimate_cost(p) for p in picks), 2),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, default=str))
    return plan


def load_plan(path: str) -> dict:
    return json.loads(Path(path).read_text())


def write_confirm_plan(
    position: str,
    drop_cols: list[str],
    run_id: str,
    seeds: list[int],
    stacked: bool,
    *,
    eager: bool = False,
    out_dir: str = STAGE2_DIR,
    image_sha: str | None = None,
) -> str:
    """Record a Stage-3 confirm run (keyed by position) so ``confirm-report`` can
    collect it without re-typing the drop-set + seeds. Merges into an existing file."""
    pos = position.upper()
    path = Path(out_dir) / CONFIRM_PLAN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data[pos] = {
        "position": pos,
        "spec": CONFIRM_SPEC,
        "run_id": run_id,
        "drop_cols": sorted(drop_cols),
        "seeds": seeds,
        "stacked": stacked,
        "eager": eager,
        "image_sha": image_sha,
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def load_confirm_plan(position: str, *, out_dir: str = STAGE2_DIR) -> dict | None:
    path = Path(out_dir) / CONFIRM_PLAN_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get(position.upper())
