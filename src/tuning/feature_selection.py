"""Feature-selection driver: plan the screens, turn their results into a
per-position report, and (on request) open a draft config-edit PR.

This is the orchestration layer over the staged feature-selection screens. The
screens themselves are shared-harness A/B specs that run on the GPU Batch fleet
(local training SIGSEGVs on the macOS torch+lightgbm+sklearn libomp triple-load);
this driver never trains — it plans launches, collects per-cell results from S3,
and post-processes them.

Stages (coarse -> fine, judged on the real pipeline per the 0.02 FP-MAE noise
floor — single-feature leave-one-out is mostly noise, which is why the screens
work on families/sub-groups):

  1. FAMILY screen   — [ab_feature_screen.py](ab_feature_screen.py) (core 8) +
     [ab_feature_screen_extended.py](ab_feature_screen_extended.py) (+ specific)
     for skill; [ab_feature_screen_k.py](ab_feature_screen_k.py) /
     [ab_feature_screen_dst.py](ab_feature_screen_dst.py) for K/DST.
  2. SUB-family zoom — [ab_feature_subscreen.py](ab_feature_subscreen.py) on the
     families Stage 1 flags neutral/borderline.
  3. CONFIRM         — re-run the candidate drop-set together at a high seed
     count (just another screen run with a hand-built --only set).

Per the design decisions: the report is **per-model** (Ridge / LightGBM / base-NN
/ Attention NN separately) with **both MAE and RMSE**, and carries NO automatic
drop rule — it ranks candidates and flags a clearly-labelled *suggested*
conservative cut, but the final whitelist is the operator's call. ``apply`` then
takes the operator's chosen drop-set and opens a DRAFT PR (CI + benchmark gated;
never auto-merge — editing include_features fires a 6-position retrain).

Usage::

    # 1. print the staged launch plan (cells + cost + exact launch_ab commands)
    python -m src.tuning.feature_selection plan --positions RB WR K DST

    # 2. after a launch finishes, build the report from S3
    python -m src.tuning.feature_selection report \\
        --spec src.tuning.ab_feature_screen_extended --run-id <id> --positions RB

    # 3. apply the cut YOU chose from the report (draft PR for review)
    python -m src.tuning.feature_selection apply --position RB \\
        --drop trend defense --pr
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning import feature_selection_stage2 as fs2  # noqa: E402
from src.tuning.attn_knob_experiments import plackett_burman_design  # noqa: E402
from src.tuning.feature_groups import (  # noqa: E402
    extract_variant_seed_metric,
    main_effects,
)

# Per AGENTS.md: judge FP-MAE deltas against a 0.02 noise floor; reuse it as the
# (rough) bar for "dropping this group doesn't hurt" on both MAE and RMSE.
NOISE_FLOOR = 0.02
METRICS = ("mae", "rmse")
# Report models in a stable, meaningful order; only those present are shown.
MODEL_ORDER = ("Ridge", "LightGBM", "NN", "Attention NN")
DEFAULT_OUT_DIR = "todo/feature_selection"
_DROP_BLOCK_RE = re.compile(
    r"\n*# >>> feature_selection drops.*?# <<< feature_selection drops <<<\n",
    re.DOTALL,
)
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SPECIAL_POSITIONS = ("K", "DST")

# The staged screens, for `plan`. (spec, applicable positions, stacked-capable).
_STAGE1_SPECS = [
    ("src.tuning.ab_feature_screen", ["QB", "RB", "WR", "TE"], True),
    ("src.tuning.ab_feature_screen_extended", ["QB", "RB", "WR", "TE"], True),
    ("src.tuning.ab_feature_screen_k", ["K"], False),
    ("src.tuning.ab_feature_screen_dst", ["DST"], False),
]


# --------------------------------------------------------------------------- #
# Design reconstruction (group_names + row_drops) for any screen spec
# --------------------------------------------------------------------------- #
def spec_design(spec_dotted: str) -> tuple[list[str], dict[str, frozenset[str]]]:
    """Return ``(group_names, row_drops)`` for a screen spec.

    New screens expose ``SCREENED_FAMILIES`` + ``ROW_DROPS`` directly. The
    validated core ``ab_feature_screen`` exposes only ``SCREENED_FAMILIES`` (its
    internals are pinned by tests), so reconstruct its PB row-drops the same way
    its ``feature_main_effects`` does.
    """
    mod = importlib.import_module(spec_dotted)
    group_names = list(mod.SCREENED_FAMILIES)
    row_drops = getattr(mod, "ROW_DROPS", None)
    if row_drops is None:
        design = plackett_burman_design(len(group_names))
        row_drops = {
            f"pb{idx:02d}": frozenset(g for g, s in zip(group_names, signs, strict=True) if s < 0)
            for idx, signs in enumerate(design, start=1)
        }
    return group_names, dict(row_drops)


def _set_subscreen_env(spec_dotted: str, position: str | None, family: str | None) -> None:
    """The sub-screen builds its groups from env at import — set them before the
    importlib.import_module in :func:`spec_design` / ``resolve_spec`` so the run's
    (position, family) groups are reconstructed correctly."""
    if not spec_dotted.endswith("ab_feature_subscreen"):
        return
    if position:
        os.environ["FF_SUBSCREEN_POSITION"] = position.upper()
    if family:
        os.environ["FF_SUBSCREEN_FAMILY"] = family


# --------------------------------------------------------------------------- #
# Collect results from S3 (reuses the launch_ab collector)
# --------------------------------------------------------------------------- #
def collect_run(
    spec_dotted: str,
    run_id: str,
    *,
    positions: list[str] | None,
    seeds: list[int] | None,
    s3_prefix: str,
):
    """Resolve the spec and download its per-cell result JSONs from S3.

    Returns ``(resolved_spec, results)``. The run manifest (if present) is
    authoritative for the grid, mirroring ``launch_ab --collect-only``.
    """
    import boto3

    from src.batch.launch import AWS_REGION, S3_BUCKET
    from src.tuning.ab_harness import resolve_spec
    from src.tuning.launch_ab import collect_results, load_run_manifest

    s3 = boto3.client("s3", region_name=AWS_REGION)
    manifest = load_run_manifest(s3, bucket=S3_BUCKET, s3_prefix=s3_prefix, run_id=run_id)
    if manifest:
        spec_dotted = manifest.get("spec", spec_dotted)
        positions = positions or manifest.get("positions")
        seeds = seeds or manifest.get("seeds")
        extra = manifest.get("extra_env") or {}
        _set_subscreen_env(
            spec_dotted, extra.get("FF_SUBSCREEN_POSITION"), extra.get("FF_SUBSCREEN_FAMILY")
        )
    spec = resolve_spec(spec_dotted, positions=positions, seeds=seeds)
    results = collect_results(
        spec, bucket=S3_BUCKET, s3_prefix=s3_prefix, run_id=run_id, s3_client=s3
    )
    return spec, results


# --------------------------------------------------------------------------- #
# Per-position effect computation
# --------------------------------------------------------------------------- #
def position_effects(
    results: list[dict],
    position: str,
    group_names: list[str],
    row_drops: dict[str, frozenset[str]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Per-model, per-metric group main effects for one position.

    Returns ``{model: {metric: {group: {mean_effect, std_effect, n_seeds}}}}``.
    Effects are metric-agnostic high-minus-low contrasts (see
    ``feature_groups.main_effects``); positive = dropping the group RAISES the
    metric = the group carries signal for that model.
    """
    models_present = sorted(
        {
            m
            for r in results
            if r.get("ok") and r.get("position") == position
            for m in r.get("metrics", {})
        },
        key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m),
    )
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model in models_present:
        out[model] = {}
        for metric in METRICS:
            vsv = extract_variant_seed_metric(results, position, model, metric)
            out[model][metric] = main_effects(vsv, row_drops, group_names)
    return out


def suggested_cut(
    effects: dict[str, dict[str, dict[str, dict[str, float]]]],
    group_names: list[str],
    *,
    noise: float = NOISE_FLOOR,
) -> list[str]:
    """Groups whose removal is neutral-or-helpful (MAE mean_effect <= noise) for
    EVERY model present. A *suggestion* only — the operator decides. Empty models
    are ignored; a group with no measured effect anywhere is not suggested."""
    suggestion: list[str] = []
    for grp in group_names:
        verdicts = []
        for by_metric in effects.values():
            eff = by_metric.get("mae", {}).get(grp)
            if eff is not None:
                verdicts.append(eff["mean_effect"] <= noise)
        if verdicts and all(verdicts):
            suggestion.append(grp)
    return suggestion


# --------------------------------------------------------------------------- #
# Static-audit corroboration (best-effort)
# --------------------------------------------------------------------------- #
def static_audit_flags(position: str, *, audit_dir: str = "analysis_output") -> dict[str, str]:
    """Best-effort: map a column to a redundancy note from a static feature-audit
    JSON (high VIF / |r|>0.95), if one exists. Never raises — corroboration only."""
    flags: dict[str, str] = {}
    p = Path(audit_dir)
    if not p.is_dir():
        return flags
    for path in list(p.glob(f"*{position.lower()}*audit*.json")) + list(
        p.glob("*feature_audit*.json")
    ):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for col, note in _extract_audit_notes(data).items():
            flags.setdefault(col, note)
    return flags


def _extract_audit_notes(data: object) -> dict[str, str]:
    """Pull {column: note} from a few common audit-JSON shapes (defensive — the
    audits' schemas vary). Looks for high-VIF and high-correlation listings."""
    notes: dict[str, str] = {}
    if not isinstance(data, dict):
        return notes
    vif = data.get("high_vif") or data.get("vif")
    if isinstance(vif, dict):
        for col, val in vif.items():
            with_val = f"VIF={val:.1f}" if isinstance(val, int | float) else "high VIF"
            notes[col] = with_val
    pairs = data.get("high_correlation") or data.get("correlated_pairs")
    if isinstance(pairs, list):
        for entry in pairs:
            if isinstance(entry, dict):
                a, b = entry.get("a") or entry.get("col1"), entry.get("b") or entry.get("col2")
                r = entry.get("r") or entry.get("corr")
                rtxt = f"r={r:.3f}" if isinstance(r, int | float) else "high r"
                for c, other in ((a, b), (b, a)):
                    if c:
                        notes.setdefault(c, f"{rtxt} with {other}")
    return notes


# --------------------------------------------------------------------------- #
# Report rendering (markdown + JSON)
# --------------------------------------------------------------------------- #
def _fmt_effect(eff: dict[str, float] | None) -> str:
    if not eff:
        return "    —"
    return f"{eff['mean_effect']:+.3f}±{eff['std_effect']:.3f}"


def _verdict(effects: dict[str, dict], grp: str, *, noise: float) -> str:
    """One-word verdict from the per-model MAE effects of a group."""
    maes = [
        by_metric["mae"][grp]["mean_effect"]
        for by_metric in effects.values()
        if grp in by_metric.get("mae", {})
    ]
    if not maes:
        return "n/a"
    if max(maes) <= noise:
        return "DROP-CAND"  # neutral-or-helpful for every model
    if min(maes) > noise:
        return "KEEP"  # carries signal for every model
    return "MIXED"  # helps some models, hurts others — operator's call


def render_report_md(
    position: str,
    spec_dotted: str,
    run_id: str,
    seeds: list[int],
    effects: dict[str, dict[str, dict[str, dict[str, float]]]],
    group_names: list[str],
    *,
    audit_flags: dict[str, str] | None = None,
    group_cols: dict[str, frozenset[str]] | None = None,
    noise: float = NOISE_FLOOR,
) -> str:
    """Render the per-position markdown report (per-model MAE+RMSE effect tables,
    ranked drop candidates, suggested conservative cut, caveats)."""
    audit_flags = audit_flags or {}
    lines: list[str] = []
    lines.append(f"# Feature-selection report — {position}")
    lines.append("")
    lines.append(f"- spec: `{spec_dotted}`")
    lines.append(f"- run-id: `{run_id}`")
    lines.append(f"- seeds: {seeds}")
    lines.append(
        f"- noise floor: {noise} FP (AGENTS.md); effect = MAE/RMSE delta when the group is DROPPED."
    )
    lines.append(
        "- **Sign:** `+` = dropping RAISES error = group carries signal (keep). `-` = dropping LOWERS error = drop candidate."
    )
    lines.append(
        "- **Ridge:** PCA is disabled for the screen (raw features) for robustness + clean "
        "attribution. Production RB/WR/DST ship PCA-Ridge — confirm a final cut on the production "
        "config (Stage 3 / the benchmark gate), not this screen's Ridge column."
    )
    lines.append("")
    models = list(effects)

    # Per-group summary across models (sorted: best drop candidates first).
    def _sort_key(grp: str) -> float:
        maes = [
            by_metric["mae"][grp]["mean_effect"]
            for by_metric in effects.values()
            if grp in by_metric.get("mae", {})
        ]
        return min(maes) if maes else 0.0

    lines.append("## Per-group effect by model (MAE | RMSE)")
    lines.append("")
    header = ["group", "verdict"]
    for m in models:
        header += [f"{m} MAE", f"{m} RMSE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for grp in sorted(group_names, key=_sort_key):
        row = [f"`{grp}`", _verdict(effects, grp, noise=noise)]
        for m in models:
            row.append(_fmt_effect(effects[m].get("mae", {}).get(grp)))
            row.append(_fmt_effect(effects[m].get("rmse", {}).get(grp)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    cut = suggested_cut(effects, group_names, noise=noise)
    lines.append("## Suggested conservative cut (review — not auto-applied)")
    lines.append("")
    if cut:
        lines.append(
            "Groups neutral-or-helpful (MAE effect ≤ noise) for **every** model present. "
            "Dropping them should not regress any model beyond the noise floor — but confirm "
            "with a combined-drop run at high seed count (Stage 3) before applying:"
        )
        lines.append("")
        for grp in cut:
            cols = sorted(group_cols.get(grp, [])) if group_cols else []
            col_txt = f" — columns: {', '.join(f'`{c}`' for c in cols)}" if cols else ""
            lines.append(f"- `{grp}`{col_txt}")
        if group_cols:
            drop_cols = sorted({c for grp in cut for c in group_cols.get(grp, [])})
            lines.append("")
            lines.append("Apply (after review):")
            lines.append("")
            lines.append("```")
            lines.append(
                f"python -m src.tuning.feature_selection apply --position {position} "
                f"--drop {' '.join(drop_cols)} --pr"
            )
            lines.append("```")
    else:
        lines.append(
            "_No group is neutral-or-helpful across all models — nothing suggested for removal._"
        )
    lines.append("")

    if audit_flags:
        lines.append("## Static-audit corroboration (VIF / correlation)")
        lines.append("")
        lines.append("Columns the static feature audit independently flagged as redundant:")
        lines.append("")
        for col, note in sorted(audit_flags.items()):
            lines.append(f"- `{col}` — {note}")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Neutral overall ≠ useless.** A group flat on overall MAE/RMSE may still carry "
        "subgroup signal (rookies/RB, returners). Judge subgroup value by *bias*, not overall MAE "
        "(draft-capital lesson). Check `result['test_df']` cohorts before dropping a borderline group."
    )
    lines.append(
        "- **Stacked vs eager:** skill positions screen stacked (vmap, FP32/LN/fixed-epochs); "
        "K/DST screen eager. Compare stacked arms only against stacked arms — never seed-by-seed "
        "against an eager run."
    )
    lines.append(
        "- **Confirm before applying.** PB main effects assume additivity; re-run the chosen "
        "drop-set together (Stage 3) at high seed count to catch interactions."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    position: str,
    spec_dotted: str,
    run_id: str,
    seeds: list[int],
    effects: dict,
    group_names: list[str],
    *,
    group_cols: dict[str, frozenset[str]] | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
) -> tuple[str, str]:
    """Write ``{out_dir}/{pos}.md`` + ``{pos}.json``; return the two paths."""
    audit_flags = static_audit_flags(position)
    md = render_report_md(
        position,
        spec_dotted,
        run_id,
        seeds,
        effects,
        group_names,
        audit_flags=audit_flags,
        group_cols=group_cols,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{position.lower()}.md"
    json_path = out / f"{position.lower()}.json"
    md_path.write_text(md)
    payload = {
        "position": position,
        "spec": spec_dotted,
        "run_id": run_id,
        "seeds": seeds,
        "noise_floor": NOISE_FLOOR,
        "effects": effects,
        "suggested_cut": suggested_cut(effects, group_names),
        "static_audit_flags": audit_flags,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return str(md_path), str(json_path)


# --------------------------------------------------------------------------- #
# apply: edit config + (optional) draft PR
# --------------------------------------------------------------------------- #
def _drop_block(position: str, drops: list[str]) -> str:
    """The self-contained, import-free config block that filters ``drops`` out of
    the constructed POSITION_CONFIG's feature fields (uniform across positions;
    no fragile literal surgery). Idempotent — :func:`apply_drops` strips any prior
    block first."""
    pos = position.upper()
    drop_lines = "\n".join(f'    "{c}",' for c in sorted(drops))
    head = (
        "\n# >>> feature_selection drops — generated; edit or remove this block "
        f"(see {DEFAULT_OUT_DIR}/{pos.lower()}.md) >>>\n"
        "_FS_DROPS = {\n" + drop_lines + "\n}\n"
    )
    if pos in SKILL_POSITIONS:
        body = (
            "POSITION_CONFIG.include_features = {\n"
            "    _k: [_c for _c in _v if _c not in _FS_DROPS]\n"
            "    for _k, _v in POSITION_CONFIG.include_features.items()\n"
            "}\n"
            "POSITION_CONFIG.specific_features = [\n"
            "    _c for _c in POSITION_CONFIG.specific_features if _c not in _FS_DROPS\n"
            "]\n"
        )
    else:  # K / DST flat lists
        body = (
            "POSITION_CONFIG.specific_features = [\n"
            "    _c for _c in POSITION_CONFIG.specific_features if _c not in _FS_DROPS\n"
            "]\n"
            "POSITION_CONFIG.contextual_features = [\n"
            "    _c for _c in POSITION_CONFIG.contextual_features if _c not in _FS_DROPS\n"
            "]\n"
            "POSITION_CONFIG.all_features = [\n"
            "    _c for _c in POSITION_CONFIG.all_features if _c not in _FS_DROPS\n"
            "]\n"
        )
    tail = (
        "POSITION_CONFIG.attn_static_features = [\n"
        "    _c for _c in POSITION_CONFIG.attn_static_features if _c not in _FS_DROPS\n"
        "]\n"
        "# <<< feature_selection drops <<<\n"
    )
    return head + body + tail


def apply_drops(position: str, drops: list[str], *, open_pr: bool = False) -> str:
    """Edit ``src/{pos}/config.py`` to drop ``drops``, verify the edit took, and
    optionally open a DRAFT PR. Returns the config path. Never auto-merges."""
    pos = position.upper()
    if pos not in (*SKILL_POSITIONS, *SPECIAL_POSITIONS):
        raise ValueError(f"unknown position {pos!r}")
    if not drops:
        raise ValueError("nothing to drop")
    cfg_path = Path(f"src/{pos.lower()}/config.py")
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    src = cfg_path.read_text()
    src = _DROP_BLOCK_RE.sub("\n", src)  # idempotent: remove any prior block
    if not src.endswith("\n"):
        src += "\n"
    cfg_path.write_text(src + _drop_block(pos, drops))

    _verify_apply(pos, drops)
    print(
        f"[fs] edited {cfg_path} — dropped {sorted(drops)} (verified absent from both model paths)"
    )
    if open_pr:
        _open_draft_pr(pos, drops)
    else:
        print(
            "[fs] review the diff, then open a DRAFT PR for benchmark review:\n"
            f"     git checkout -b feat/feature-select-{pos.lower()} && "
            f"git add {cfg_path} && git commit && gh pr create --draft"
        )
    return str(cfg_path)


def _verify_apply(position: str, drops: list[str]) -> None:
    """Positive control: import the edited config in a fresh subprocess and assert
    the dropped columns are gone from get_feature_columns() AND
    attn_static_features. Aborts loudly if the edit silently didn't take."""
    pos = position.lower()
    drop_set = json.dumps(sorted(drops))
    code = (
        f"from src.{pos}.config import POSITION_CONFIG as p\n"
        f"from src.{pos}.features import get_feature_columns as g\n"
        f"import json\n"
        f"d=set(json.loads('{drop_set}'))\n"
        f"feat=set(g()); stat=set(p.attn_static_features)\n"
        f"assert not (feat & d), f'still in features: {{feat & d}}'\n"
        f"assert not (stat & d), f'still in attn_static: {{stat & d}}'\n"
        f"print('verify-ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0 or "verify-ok" not in proc.stdout:
        raise RuntimeError(
            f"apply verification FAILED for {position}; the drop did not take.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )


def _open_draft_pr(position: str, drops: list[str]) -> None:
    """Branch, commit the config edit + report, open a DRAFT PR. Worktree-safe;
    never merges. Surfaces (not swallows) git/gh failures."""
    pos = position.lower()
    branch = f"feat/feature-select-{pos}"
    cfg = f"src/{pos}/config.py"
    report = f"{DEFAULT_OUT_DIR}/{pos}.md"
    body = (
        f"Apply the feature-selection cut for {position.upper()} "
        f"({len(drops)} columns dropped from `include_features` / `all_features` and the "
        "attention static branch).\n\n"
        f"Dropped: {', '.join(f'`{c}`' for c in sorted(drops))}\n\n"
        f"Chosen from the per-model MAE+RMSE effect report (`{report}`). **Behavioral — fires a "
        "6-position retrain.** DRAFT: review the benchmark delta before un-drafting; do not "
        "auto-merge.\n"
    )
    to_add = [cfg] + ([report] if Path(report).is_file() else [])
    cmds = [
        ["git", "checkout", "-b", branch],
        ["git", "add", *to_add],
        ["git", "commit", "-m", f"feat({pos}): apply feature-selection cut ({len(drops)} cols)"],
        ["git", "push", "-u", "origin", branch],
        [
            "gh",
            "pr",
            "create",
            "--draft",
            "--title",
            f"feat({pos}): feature-selection cut",
            "--body",
            body,
        ],
    ]
    for cmd in cmds:
        print(f"[fs] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")
    print("[fs] DRAFT PR opened — review the benchmark delta before un-drafting; never auto-merge.")


# --------------------------------------------------------------------------- #
# plan: print the staged launch commands + cell counts
# --------------------------------------------------------------------------- #
def _grid_size(spec_dotted: str, positions: list[str], seeds_per_cell: int) -> int:
    from src.tuning.ab_harness import resolve_spec

    spec = resolve_spec(spec_dotted, positions=positions)
    return len(positions) * len(spec.variants) * seeds_per_cell


# launch_ab's DEFAULT_MAX_CELLS cost guard; a stacked screen's per-seed cell
# count (24 seeds x 13 variants) overshoots it even though the real compute is
# ~13 cheap vmap groups, so the printed command must raise the cap explicitly.
_LAUNCH_MAX_CELLS_DEFAULT = 120


def cmd_plan(args) -> int:
    want = [p.upper() for p in args.positions]
    print("Staged feature-selection plan (one Batch job per position; cells run sequentially):\n")
    print("Stage 1 — family screen:")
    for spec, applies, stackable in _STAGE1_SPECS:
        pos = [p for p in applies if p in want]
        if not pos:
            continue
        seeds = 24 if (stackable and args.stacked) else 3
        cells = _grid_size(spec, pos, seeds)
        stack_flag = " --stacked-seeds" if (stackable and args.stacked) else ""
        # Raise the cost guard when the per-seed cell count exceeds it (stacked
        # runs ~len(variants) cheap vmap groups regardless of the seed count).
        cap = f" --max-cells {cells}" if cells > _LAUNCH_MAX_CELLS_DEFAULT else ""
        print(
            f"  ~{cells} cells  python -m src.tuning.launch_ab --spec {spec} "
            f"--positions {' '.join(pos)}{stack_flag}{cap}"
        )
    print("\nStage 2 — sub-family zoom (auto-selected from the Stage-1 reports):")
    print(f"  python -m src.tuning.feature_selection substage --positions {' '.join(want)}")
    print(
        "  python -m src.tuning.feature_selection substage-report --positions <pos>  # after they finish"
    )
    print("\nStage 3 — confirm the chosen drop-set together on production PCA-Ridge:")
    print("  python -m src.tuning.feature_selection confirm --position <pos> --from-stage2")
    print(
        "  python -m src.tuning.feature_selection confirm-report --position <pos>      # after it finishes"
    )
    print(
        "\nNote: substage/confirm only PRINT commands (--exec to submit). Each launch fires "
        "real Spot jobs; --dry-run on launch_ab to preview. Stacked (skill) ≈ 24 seeds; K/DST eager."
    )
    return 0


def cmd_report(args) -> int:
    _set_subscreen_env(args.spec, args.positions[0] if args.positions else None, args.family)
    spec, results = collect_run(
        args.spec, args.run_id, positions=args.positions, seeds=args.seeds, s3_prefix=args.s3_prefix
    )
    group_names, row_drops = spec_design(spec.dotted)
    group_cols = _group_cols_for(spec.dotted, args.family)
    written = []
    for position in spec.positions:
        effects = position_effects(results, position, group_names, row_drops)
        md_path, json_path = write_report(
            position,
            spec.dotted,
            args.run_id,
            spec.seeds,
            effects,
            group_names,
            group_cols=group_cols,
            out_dir=args.out_dir,
        )
        written.append(md_path)
        print(f"[fs] {position}: wrote {md_path} + {json_path}")
    print(f"[fs] {len(written)} report(s) under {args.out_dir}/")
    return 0


def _group_cols_for(spec_dotted: str, family: str | None) -> dict[str, frozenset[str]]:
    """Best-effort {group: columns} for the report's column listings."""
    try:
        mod = importlib.import_module(spec_dotted)
        gc = getattr(mod, "_GROUP_COLS", None)
        if gc:
            return dict(gc)
    except Exception:  # noqa: BLE001 — column listings are a nicety, not required
        pass
    return {}


def cmd_apply(args) -> int:
    apply_drops(args.position, args.drop, open_pr=args.pr)
    return 0


# --------------------------------------------------------------------------- #
# Stage-2 / Stage-3 orchestration (substage / substage-report / confirm /
# confirm-report) — thin wiring over feature_selection_stage2.
# --------------------------------------------------------------------------- #
def _load_stage1(in_dir: str, position: str) -> dict:
    path = Path(in_dir) / f"{position.lower()}.json"
    if not path.is_file():
        raise SystemExit(
            f"[fs] Stage-1 report not found: {path}. Run Stage 1 first "
            "(`feature_selection report ...`) or rebase on origin/main — the "
            "todo/feature_selection/{pos}.json reports are tracked there."
        )
    return json.loads(path.read_text())


def _exec_commands(commands: list[str]) -> None:
    """``--exec``: actually submit the Batch jobs. Fires REAL Spot $ — opt-in only;
    the default path prints the commands and touches no AWS.

    The generated commands carry a leading ``KEY=VAL`` env-prefix (so launch_ab's own
    resolve_spec sees the parametrized spec); split it back into ``env=`` and run the
    rest as argv, with ``python`` resolved to this interpreter."""
    print("\n[fs] --exec: submitting REAL Spot Batch jobs (this costs money)...")
    for cmd in commands:
        print(f"[fs] $ {cmd}")
        overrides, argv = fs2.split_env_prefixed_command(cmd)
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        rc = subprocess.run(argv, env={**os.environ, **overrides}, check=False).returncode
        if rc != 0:
            raise SystemExit(f"[fs] command failed (rc={rc}): {cmd}")


def _print_substage_plan(picks, skipped, *, image_sha) -> None:
    total = round(sum(fs2.estimate_cost(p) for p in picks), 2)
    print(
        f"Stage-2 sub-family screen plan — {len(picks)} sub-screen(s), "
        f"~${total:.2f} rough Spot $ (±2x; not a billing oracle).\n"
    )
    by_pos: dict[str, list] = {}
    for p in picks:
        by_pos.setdefault(p.position, []).append(p)
    skip_by_pos: dict[str, list] = {}
    for s in skipped:
        skip_by_pos.setdefault(s["position"], []).append(s)
    for pos in sorted(set(by_pos) | set(skip_by_pos)):
        sel = by_pos.get(pos, [])
        print(f"{pos} — {len(sel)} selected, {len(skip_by_pos.get(pos, []))} skipped:")
        for p in sel:
            mode = "stacked24" if p.stacked else f"eager{len(p.seeds)}"
            print(
                f"  zoom {p.family:<14} {p.n_variants:>2} var x {len(p.seeds):>2} seed "
                f"= {p.cells:>4} cells  ~${fs2.estimate_cost(p):>5.2f}  [{mode}]  ({p.reason})"
            )
        for s in skip_by_pos.get(pos, []):
            print(f"  skip {s['family']:<14} ({s['reason']})")
        print()
    print("Recommended flow (prints only — NEVER auto-fires Batch):")
    if not picks:
        print("  (nothing selected — adjust --only-family / --skip-family / --max-families)")
        return
    print("  1. SMOKE one real cell first — the riskiest arm at 1 seed. Degenerate arms")
    print("     only crash live (--list / unit tests validate grid construction, #1187->#1212):")
    print(f"       {fs2.smoke_command(picks[0], image_sha=image_sha)}")
    print("     Confirm it lands a non-crash result in S3, THEN launch each sub-screen:")
    for p in picks:
        print(f"       {fs2.subscreen_launch_command(p, image_sha=image_sha)}")
    rep = " ".join(sorted({p.position for p in picks}))
    print("  2. After they finish, build the consolidated Stage-2 report:")
    print(f"       python -m src.tuning.feature_selection substage-report --positions {rep}")


def cmd_substage(args) -> int:
    positions = [p.upper() for p in args.positions]
    stamp = fs2._utc_stamp()
    all_picks: list = []
    all_skipped: list = []
    for pos in positions:
        payload = _load_stage1(args.in_dir, pos)
        picks, skipped = fs2.select_subscreens(
            pos,
            payload,
            stacked_n=args.stacked_n,
            kdst_n=args.kdst_n,
            only=args.only_family,
            skip=args.skip_family,
            max_families=args.max_families,
        )
        fs2.assign_run_ids(picks, image_sha=args.image_sha, stamp=stamp)
        all_picks.extend(picks)
        all_skipped.extend(skipped)
    _print_substage_plan(all_picks, all_skipped, image_sha=args.image_sha)
    out_plan = args.out_plan or os.path.join(fs2.STAGE2_DIR, fs2.PLAN_FILE)
    fs2.write_plan(all_picks, all_skipped, out_path=out_plan, image_sha=args.image_sha, stamp=stamp)
    print(f"\n[fs] plan -> {out_plan}  ({len(all_picks)} sub-screens recorded)")
    if args.execute:
        _exec_commands(
            [fs2.subscreen_launch_command(p, image_sha=args.image_sha) for p in all_picks]
        )
    return 0


def cmd_substage_report(args) -> int:
    plan_path = args.plan or os.path.join(fs2.STAGE2_DIR, fs2.PLAN_FILE)
    plan = fs2.load_plan(plan_path)
    want = {p.upper() for p in args.positions} if args.positions else None
    by_pos: dict[str, list] = {}
    for pick in plan["picks"]:
        if want and pick["position"] not in want:
            continue
        by_pos.setdefault(pick["position"], []).append(pick)
    if not by_pos:
        raise SystemExit(f"[fs] no matching picks in {plan_path} (positions={args.positions})")
    written = []
    for pos, picks in by_pos.items():
        family_reports = []
        for pick in picks:
            spec, group_cols, row_drops = fs2.build_subscreen_spec(
                pos, pick["family"], pick["seeds"]
            )
            effects = fs2.collect_effects(
                spec, pos, list(group_cols), row_drops, pick["run_id"], s3_prefix=args.s3_prefix
            )
            family_reports.append(
                {
                    "family": pick["family"],
                    "effects": effects,
                    "group_cols": group_cols,
                    "run_id": pick["run_id"],
                    "seeds": pick["seeds"],
                }
            )
        md, js = fs2.write_stage2_report(pos, family_reports, out_dir=args.out_dir)
        written.append(md)
        print(f"[fs] {pos}: wrote {md} + {js}")
    print(f"[fs] {len(written)} Stage-2 report(s) under {args.out_dir}/")
    return 0


def cmd_confirm(args) -> int:
    pos = args.position.upper()
    if args.from_stage2:
        path = Path(args.stage2_dir) / f"{pos.lower()}.json"
        if not path.is_file():
            raise SystemExit(
                f"[fs] no Stage-2 report at {path}; run substage-report first or pass --drop."
            )
        drop_cols = json.loads(path.read_text()).get("combined_suggested_drop_cols", [])
    else:
        drop_cols = args.drop or []
    if not drop_cols:
        raise SystemExit(
            "[fs] nothing to confirm: pass --drop col ... or --from-stage2 (with a "
            "non-empty combined cut)."
        )
    stamp = fs2._utc_stamp()
    sha7 = (args.image_sha or "local")[:7]
    run_id = args.run_id or f"confirm-{pos.lower()}-{stamp}-{sha7}"
    stacked, seeds = fs2.confirm_regime(pos, eager=args.eager)
    cmd = fs2.confirm_launch_command(
        pos, drop_cols, eager=args.eager, image_sha=args.image_sha, run_id=run_id
    )
    smoke = fs2.confirm_smoke_command(pos, drop_cols, image_sha=args.image_sha, run_id=run_id)
    mode = "stacked24" if stacked else f"eager{len(seeds)}"
    print(
        f"Stage-3 confirm — {pos}: drop {len(drop_cols)} columns TOGETHER on the "
        f"PRODUCTION PCA-Ridge config ({mode}, {2 * len(seeds)} cells)."
    )
    print(f"  columns: {', '.join(drop_cols)}\n")
    print("Recommended flow (prints only — NEVER auto-fires Batch):")
    print("  1. SMOKE the drop arm first (a drop below ridge_pca_components crashes PCA live):")
    print(f"       {smoke}")
    print("  2. Run the confirm:")
    print(f"       {cmd}")
    print("  3. Report it (production PCA-Ridge, per-model MAE+RMSE):")
    print(f"       python -m src.tuning.feature_selection confirm-report --position {pos}")
    fs2.write_confirm_plan(
        pos,
        drop_cols,
        run_id,
        seeds,
        stacked,
        eager=args.eager,
        out_dir=args.stage2_dir,
        image_sha=args.image_sha,
    )
    if args.execute:
        _exec_commands([cmd])
    return 0


def cmd_confirm_report(args) -> int:
    pos = args.position.upper()
    entry = fs2.load_confirm_plan(pos, out_dir=args.stage2_dir) or {}
    run_id = args.run_id or entry.get("run_id")
    if args.drop:
        drop_cols = args.drop
    elif args.from_stage2:
        data = json.loads((Path(args.stage2_dir) / f"{pos.lower()}.json").read_text())
        drop_cols = data.get("combined_suggested_drop_cols", [])
    else:
        drop_cols = entry.get("drop_cols")
    seeds = args.seeds or entry.get("seeds")
    if not (run_id and drop_cols and seeds):
        raise SystemExit(
            "[fs] confirm-report needs a run-id + drop-set + seeds. Run `confirm` first "
            "(it records confirm_plan.json) or pass --run-id --drop ... --seeds ..."
        )
    spec, group_names, row_drops = fs2.build_confirm_spec(pos, drop_cols, seeds)
    effects = fs2.collect_effects(
        spec, pos, group_names, row_drops, run_id, s3_prefix=args.s3_prefix
    )
    md, js = fs2.write_confirm_report(pos, drop_cols, run_id, seeds, effects, out_dir=args.out_dir)
    print(f"[fs] {pos}: wrote {md} + {js}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("plan", help="Print the staged launch plan + cell counts")
    pl.add_argument(
        "--positions", nargs="+", default=list(SKILL_POSITIONS) + list(SPECIAL_POSITIONS)
    )
    pl.add_argument(
        "--stacked",
        action="store_true",
        default=True,
        help="Assume stacked seeds for skill (default)",
    )
    pl.add_argument("--no-stacked", dest="stacked", action="store_false")
    pl.set_defaults(func=cmd_plan)

    rp = sub.add_parser("report", help="Collect a run from S3 and write per-position reports")
    rp.add_argument("--spec", required=True, help="Dotted screen spec module")
    rp.add_argument("--run-id", required=True)
    rp.add_argument("--positions", nargs="+")
    rp.add_argument("--seeds", type=int, nargs="+")
    rp.add_argument("--family", help="(sub-screen only) the family that was zoomed")
    rp.add_argument("--s3-prefix", default="ab_runs")
    rp.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    rp.set_defaults(func=cmd_report)

    ap = sub.add_parser("apply", help="Apply a chosen drop-set to a position config (draft PR)")
    ap.add_argument("--position", required=True)
    ap.add_argument("--drop", nargs="+", required=True, help="Column names to drop")
    ap.add_argument("--pr", action="store_true", help="Open a DRAFT PR (else just edit + verify)")
    ap.set_defaults(func=cmd_apply)

    # --- Stage 2: sub-family zoom orchestration ---
    ss = sub.add_parser(
        "substage",
        help="Stage 2: select + print the sub-family screen launch commands (no submit)",
    )
    ss.add_argument("--positions", nargs="+", required=True)
    ss.add_argument("--in-dir", default=fs2.STAGE1_DIR, help="Where the Stage-1 {pos}.json live")
    ss.add_argument("--out-plan", default=None, help=f"(default: {fs2.STAGE2_DIR}/{fs2.PLAN_FILE})")
    ss.add_argument(
        "--image-sha", default=None, help="ff-training image tag for the printed commands"
    )
    ss.add_argument("--only-family", nargs="+", default=None, help="Restrict to these families")
    ss.add_argument("--skip-family", nargs="+", default=None, help="Exclude these families")
    ss.add_argument("--max-families", type=int, default=None, help="Cap sub-screens per position")
    ss.add_argument("--stacked-n", type=int, default=fs2.DEFAULT_STACKED_N)
    ss.add_argument("--kdst-n", type=int, default=fs2.DEFAULT_KDST_N)
    ss.add_argument(
        "--exec",
        dest="execute",
        action="store_true",
        help="Actually submit the Batch jobs (fires REAL Spot $; default just prints)",
    )
    ss.set_defaults(func=cmd_substage)

    sr = sub.add_parser(
        "substage-report",
        help="Stage 2: collect the sub-screen runs into a consolidated per-position report",
    )
    sr.add_argument("--positions", nargs="+", help="(default: all positions in the plan)")
    sr.add_argument("--plan", default=None, help=f"(default: {fs2.STAGE2_DIR}/{fs2.PLAN_FILE})")
    sr.add_argument("--out-dir", default=fs2.STAGE2_DIR)
    sr.add_argument("--s3-prefix", default="ab_runs")
    sr.set_defaults(func=cmd_substage_report)

    # --- Stage 3: confirm a chosen drop-set on the production PCA-Ridge config ---
    cf = sub.add_parser(
        "confirm",
        help="Stage 3: print the combined drop-set confirm on production PCA-Ridge (no submit)",
    )
    cf.add_argument("--position", required=True)
    cf.add_argument("--drop", nargs="+", default=None, help="Column names to drop together")
    cf.add_argument(
        "--from-stage2",
        action="store_true",
        help="Read the combined drop-set from the Stage-2 report instead of --drop",
    )
    cf.add_argument(
        "--eager",
        action="store_true",
        help="Skill: eager (faithful attention) instead of the default stacked-24",
    )
    cf.add_argument("--stage2-dir", default=fs2.STAGE2_DIR)
    cf.add_argument("--image-sha", default=None)
    cf.add_argument("--run-id", default=None)
    cf.add_argument(
        "--exec",
        dest="execute",
        action="store_true",
        help="Actually submit the confirm Batch job (fires REAL Spot $; default just prints)",
    )
    cf.set_defaults(func=cmd_confirm)

    cr = sub.add_parser(
        "confirm-report",
        help="Stage 3: report a confirm run (production PCA-Ridge per-model MAE+RMSE)",
    )
    cr.add_argument("--position", required=True)
    cr.add_argument("--run-id", default=None, help="(default: the recorded confirm_plan.json run)")
    cr.add_argument("--drop", nargs="+", default=None)
    cr.add_argument("--from-stage2", action="store_true")
    cr.add_argument("--seeds", type=int, nargs="+", default=None)
    cr.add_argument("--stage2-dir", default=fs2.STAGE2_DIR)
    cr.add_argument("--out-dir", default=fs2.STAGE2_DIR)
    cr.add_argument("--s3-prefix", default="ab_runs")
    cr.set_defaults(func=cmd_confirm_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
