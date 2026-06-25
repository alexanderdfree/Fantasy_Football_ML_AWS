"""Canonical per-position **feature manifest** + a merge-time drift detector.

Why this exists
---------------
The repo already detects *file-level* change on merge and adjusts downstream
consumers: ``src/scripts/scope_positions.py`` maps a changed file to the
positions that retrain, ``.github/workflows/refresh-splits.yml`` rebuilds
``data/splits`` (with a whitelist verify gate), and a handful of contract tests
pin individual invariants. What was missing is a single *feature-level* view —
nothing computed "what features / targets / attention inputs does each position
declare" and diffed it across a merge to say *"feature X was added/removed →
here are the consumers that must move with it."*

This module derives that manifest straight off each position's authoritative
``POSITION_CONFIG`` (the :class:`~src.shared.position_config.PositionConfig`
dataclass — there are no UPPERCASE mirror constants), and a committed snapshot
(``tests/shared/feature_manifest.snapshot.json``) records the last-acknowledged
state. ``tests/shared/test_feature_manifest.py`` fails loud when the live
manifest drifts from the snapshot, so a feature change cannot land without an
explicit acknowledgement (regenerating the snapshot) plus a reminder of the
downstream consumers to update.

CLI
---
- ``python -m src.scripts.feature_manifest``            validate live vs snapshot (exit 1 on drift)
- ``python -m src.scripts.feature_manifest --write``    regenerate the committed snapshot
- ``python -m src.scripts.feature_manifest --diff REF`` diff the live manifest vs another ref's snapshot

Design notes
------------
- Lives in ``src/scripts/`` so editing it never trips ``scope_positions`` (no
  retrain) nor the pre-PR B2 benchmark gate.
- The manifest tracks only the **feature / target / model-shape contract** — the
  lists with hard downstream coupling — and *deliberately omits* pure training
  hyperparameters (lr, epochs, scheduler, LightGBM/TabPFN knobs). That split is
  enforced field-by-field by ``MANIFEST_FIELDS`` / ``IGNORED_FIELDS`` below, so
  retuning a Huber delta does not churn the snapshot but adding a feature does.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.features.engineer import flatten_include_features
from src.shared.position_config import PositionConfig
from src.shared.registry import (
    ALL_POSITIONS,
    _flat_attn_kwargs_static,
    _nested_attn_kwargs_static,
)

# Repo-root-relative default location of the committed snapshot. Under
# ``tests/`` so it never triggers a retrain (scope_positions strips tests/) and
# never fires the deploy workflow (unlike docs/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = _REPO_ROOT / "tests" / "shared" / "feature_manifest.snapshot.json"
_SNAPSHOT_REL = "tests/shared/feature_manifest.snapshot.json"

# ---------------------------------------------------------------------------
# Field classification — every PositionConfig dataclass field must be in exactly
# one of these two sets. ``test_feature_manifest.py`` asserts the partition is
# total + disjoint, so a NEW config field fails the test until it is classified.
# That is the guard against the "new attn_* knob added to config but never
# threaded into serving" trap (#121 / the 2026-06-15 staleness NaN): a new
# ``attn_*`` field lands in neither set, the test fails, and the author must
# decide MANIFEST (capture it here + thread arch knobs through
# registry._flat/_nested_attn_kwargs_static) vs IGNORED (pure hyperparameter).
# ---------------------------------------------------------------------------

# Fields that define the feature / target / model-shape contract. A change to
# any of these is a "feature change" the manifest records and the snapshot diff
# surfaces. Arch-shaping knobs here MUST also be forwarded to serving by
# registry._flat_attn_kwargs_static / _nested_attn_kwargs_static.
MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        # identity + feature whitelist
        "name",
        "targets",
        "specific_features",
        "include_features",
        "contextual_features",
        "all_features",
        "drop_features",
        # loss-head shape (which targets, which family; tuned *values* excluded)
        "head_losses",
        "loss_weights",
        "huber_deltas",
        "poisson_targets",
        "gated_targets",
        "nn_non_negative_targets",
        # base-NN architecture (parameter shapes)
        "nn_backbone_layers",
        "nn_head_hidden",
        "nn_head_hidden_overrides",
        "nn_dropout",
        # attention-NN architecture
        "train_attention_nn",
        "attn_d_model",
        "attn_n_heads",
        "attn_encoder_hidden_dim",
        "attn_max_seq_len",
        "attn_positional_encoding",
        "attn_dropout",
        "attn_project_kv",
        "attn_gated_fusion",
        "attn_gated",
        "attn_gate_hidden",
        "attn_no_history_embedding",
        "attn_condition_queries_on_static",
        "attn_history_stats",
        "attn_static_features",
        # opposing-side attention branch
        "opp_attn_history_stats",
        "opp_attn_max_seq_len",
        "opp_attn_kind",
        # nested kick attention (K)
        "attn_max_games",
        "attn_kick_dim",
        "attn_max_kicks_per_game",
        "attn_kick_stats",
        # which served model variants exist + their head shapes
        "train_lightgbm",
        "td_model_type",
        "two_stage_targets",
        "ordinal_targets",
        "gated_ordinal_targets",
        # K serving aggregator
        "target_signs",
    }
)

# Fields the manifest deliberately ignores: pure training/tuning hyperparameters
# and orchestration metadata that do not change which features/targets the model
# consumes nor its parameter shapes. Retuning these must NOT churn the snapshot.
IGNORED_FIELDS: frozenset[str] = frozenset(
    {
        # training population
        "min_games_per_season",
        "min_games",
        "seasons",
        # ridge / elasticnet search
        "ridge_alpha_grids",
        "ridge_pca_components",
        "ridge_cv_folds",
        "ridge_refine_points",
        "cv_split_column",
        "train_elasticnet",
        "enet_l1_ratios",
        # base-NN optimizer / schedule
        "nn_lr",
        "nn_weight_decay",
        "nn_epochs",
        "nn_batch_size",
        "nn_patience",
        "nn_use_amp",
        "scheduler_type",
        "cosine_t0",
        "cosine_t_mult",
        "cosine_eta_min",
        "onecycle_max_lr",
        "onecycle_pct_start",
        # attention optimizer / schedule + loss scaling (not a shape)
        "attn_lr",
        "attn_weight_decay",
        "attn_batch_size",
        "attn_patience",
        "attn_cosine_eta_min",
        "attn_onecycle_max_lr",
        "attn_scheduler_type",
        "attn_cosine_t0",
        "attn_cosine_t_mult",
        "attn_onecycle_pct_start",
        "attn_gate_weight",
        # LightGBM hyperparameters
        "lgbm_n_estimators",
        "lgbm_learning_rate",
        "lgbm_num_leaves",
        "lgbm_max_depth",
        "lgbm_subsample",
        "lgbm_colsample_bytree",
        "lgbm_reg_lambda",
        "lgbm_reg_alpha",
        "lgbm_min_child_samples",
        "lgbm_min_split_gain",
        "lgbm_objective",
        # TabPFN (benchmark-only, never served)
        "train_tabpfn",
        "tabpfn_n_estimators",
        "tabpfn_pca_components",
        "tabpfn_ignore_pretraining_limits",
        "tabpfn_softmax_temperature",
        "tabpfn_auto_scale_n_estimators",
        "tabpfn_inference_config",
        # orchestration metadata
        "accepts_dataframes",
        "cpu_only",
    }
)


def positionconfig_fields() -> set[str]:
    """All ``PositionConfig`` dataclass field names (the live schema)."""
    return {f.name for f in dataclasses.fields(PositionConfig)}


def _position_config(pos: str) -> PositionConfig:
    """Authoritative ``POSITION_CONFIG`` for a position (no data/torch import)."""
    return importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG


def _feature_columns(pc: PositionConfig) -> list[str]:
    """The flat, ordered feature whitelist the model actually sees.

    Skill positions (QB/RB/WR/TE) carry the ``include_features`` category dict
    and flatten it via the same helper production uses (so ``n_features`` and
    ordering match ``features.get_feature_columns``); K/DST carry the flat
    ``all_features`` list directly.
    """
    if pc.include_features:
        return list(flatten_include_features(pc.include_features))
    return list(pc.all_features)


def _served_attn_kwargs_keys(pos: str, pc: PositionConfig) -> list[str]:
    """Sorted keys of the attention served-kwargs dict app.py reconstructs.

    Captured so any change to the serving contract surface (a knob added/removed
    from registry._flat/_nested_attn_kwargs_static) shows up in the snapshot
    diff alongside the config change that caused it.
    """
    kwargs = _nested_attn_kwargs_static(pc) if pos == "K" else _flat_attn_kwargs_static(pc)
    return sorted(kwargs)


def build_position_manifest(pos: str) -> dict[str, Any]:
    """Derive the canonical feature manifest for one position."""
    pc = _position_config(pos)
    features = _feature_columns(pc)
    return {
        "name": pc.name,
        "targets": list(pc.targets),
        "n_features": len(features),
        "features": features,
        "specific_features": list(pc.specific_features),
        "contextual_features": list(pc.contextual_features),
        "drop_features": sorted(pc.drop_features),
        "attn": {
            "structure": "nested" if (pc.attn_kick_stats or pc.attn_max_games) else "flat",
            "train_attention_nn": pc.train_attention_nn,
            "static_features": list(pc.attn_static_features),
            "history_stats": list(pc.attn_history_stats),
            "opp_history_stats": list(pc.opp_attn_history_stats),
            "opp_kind": pc.opp_attn_kind,
            "kick_stats": list(pc.attn_kick_stats),
            "max_seq_len": pc.attn_max_seq_len,
            "max_games": pc.attn_max_games,
            "kick_dim": pc.attn_kick_dim,
            "max_kicks_per_game": pc.attn_max_kicks_per_game,
            "d_model": pc.attn_d_model,
            "n_heads": pc.attn_n_heads,
            "encoder_hidden_dim": pc.attn_encoder_hidden_dim,
            "positional_encoding": pc.attn_positional_encoding,
            "project_kv": pc.attn_project_kv,
            "gated_fusion": pc.attn_gated_fusion,
            "gated": pc.attn_gated,
            "gate_hidden": pc.attn_gate_hidden,
            "no_history_embedding": pc.attn_no_history_embedding,
            "condition_queries_on_static": pc.attn_condition_queries_on_static,
            "served_kwargs_keys": _served_attn_kwargs_keys(pos, pc),
        },
        "nn": {
            "backbone_layers": list(pc.nn_backbone_layers),
            "head_hidden": pc.nn_head_hidden,
            "head_hidden_overrides": dict(sorted(pc.nn_head_hidden_overrides.items())),
            "dropout": pc.nn_dropout,
        },
        "loss_heads": {
            # target -> family (mse / huber / poisson_nll / ...). The *family* is
            # structural (changes the head); tuned delta/weight VALUES are owned
            # by tests/test_invariants.py, so only the KEY sets are captured here.
            "head_losses": {t: pc.head_losses.get(t) for t in pc.targets},
            "huber_delta_targets": sorted(pc.huber_deltas),
            "loss_weight_targets": sorted(pc.loss_weights),
            "poisson_targets": list(pc.poisson_targets),
            "gated_targets": list(pc.gated_targets),
            "non_negative_targets": sorted(pc.nn_non_negative_targets),
        },
        "td_model": {
            "type": pc.td_model_type,
            "two_stage_targets": sorted(pc.two_stage_targets),
            "ordinal_targets": sorted(pc.ordinal_targets),
            "gated_ordinal_targets": sorted(pc.gated_ordinal_targets),
        },
        "target_signs_keys": sorted(pc.target_signs) if pc.target_signs else [],
        "lightgbm_served": pc.train_lightgbm,
    }


def build_manifest() -> dict[str, Any]:
    """Derive the full manifest for all six positions, in canonical order."""
    return {pos: build_position_manifest(pos) for pos in ALL_POSITIONS}


def dumps(manifest: dict[str, Any]) -> str:
    """Canonical JSON serialization (stable, diff-friendly, trailing newline)."""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any] | None:
    """Load the committed snapshot, or ``None`` if it does not exist yet."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Diff + report
# ---------------------------------------------------------------------------


def _diff_value(path: str, old: Any, new: Any, out: list[str]) -> None:
    """Recursively diff two JSON-able values, appending human lines to ``out``."""
    if isinstance(old, dict) or isinstance(new, dict):
        old_d = old if isinstance(old, dict) else {}
        new_d = new if isinstance(new, dict) else {}
        for key in sorted(set(old_d) | set(new_d)):
            _diff_value(f"{path}.{key}" if path else str(key), old_d.get(key), new_d.get(key), out)
    elif isinstance(old, list) or isinstance(new, list):
        old_l = old if isinstance(old, list) else []
        new_l = new if isinstance(new, list) else []
        removed = [x for x in old_l if x not in new_l]
        added = [x for x in new_l if x not in old_l]
        if removed:
            out.append(f"  - {path}: removed {removed}")
        if added:
            out.append(f"  + {path}: added {added}")
    elif old != new:
        out.append(f"  ~ {path}: {old!r} -> {new!r}")


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Per-position structured diff between two manifests."""
    lines: list[str] = []
    for pos in sorted(set(old) | set(new)):
        pos_lines: list[str] = []
        _diff_value("", old.get(pos, {}), new.get(pos, {}), pos_lines)
        if pos_lines:
            lines.append(f"[{pos}]")
            lines.extend(pos_lines)
    return lines


# The "adjust downstream consumers accordingly" checklist printed on drift.
CONSUMER_CHECKLIST = """\
Downstream consumers to verify for the change(s) above:
  • features add/remove   -> engineer.build_features OR {pos}/features.add_specific_features must
                             produce it; data/splits (refresh-splits verify gate) + test fixtures.
  • feed into attention   -> add to attn.static_features (non-temporal) OR attn.history_stats; the
                             temporal-leak rule is enforced by tests/test_attn_static_columns.py.
  • new attn_*/nn_* knob   -> thread through registry._flat/_nested_attn_kwargs_static (else serving
                             rebuilds a mismatched state_dict and NaNs after the next model swap).
  • target add/remove     -> head_losses + (huber heads) huber_deltas/loss_weights +
                             nn_non_negative_targets; aggregate_targets + serving comparison tab.
After updating the consumers, run:  python -m src.scripts.feature_manifest --write"""


def _git_show_snapshot(ref: str) -> dict[str, Any]:
    raw = subprocess.run(
        ["git", "show", f"{ref}:{_SNAPSHOT_REL}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the committed snapshot")
    parser.add_argument(
        "--diff", metavar="REF", help="diff live manifest vs another git ref's snapshot"
    )
    parser.add_argument("--manifest-path", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    live = build_manifest()

    if args.write:
        args.manifest_path.write_text(dumps(live))
        print(f"wrote {len(live)} positions -> {args.manifest_path}")
        return 0

    if args.diff:
        old = _git_show_snapshot(args.diff)
        lines = diff_manifests(old, live)
        if not lines:
            print(f"no feature-manifest changes vs {args.diff}")
        else:
            print(f"feature-manifest changes vs {args.diff}:")
            print("\n".join(lines))
        return 0

    # default: validate live vs committed snapshot
    snapshot = load_snapshot(args.manifest_path)
    if snapshot is None:
        print(
            f"no snapshot at {args.manifest_path}; create it with "
            f"`python -m src.scripts.feature_manifest --write`",
            file=sys.stderr,
        )
        return 1
    if snapshot == live:
        print(f"feature manifest is in sync ({len(live)} positions)")
        return 0
    print(
        "FEATURE MANIFEST DRIFT — the live config no longer matches the snapshot:\n",
        file=sys.stderr,
    )
    print("\n".join(diff_manifests(snapshot, live)), file=sys.stderr)
    print("\n" + CONSUMER_CHECKLIST, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
