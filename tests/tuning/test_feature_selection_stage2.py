"""Unit tests for the Stage-2/3 feature-selection orchestration (CI-safe).

No training and no AWS (the screens run on the GPU Batch fleet; local training
SIGSEGVs on the macOS libomp triple-load). Coverage is the pure orchestration logic:

  * SELECT — the Comprehensive selection rule (drop-candidate/large families zoomed,
    atomic + clean all-signal KEEP skipped; skill stacked-24, K/DST eager-8), plus the
    --only/--skip/--max-families overrides,
  * LAUNCH — the exact launch_ab command + smoke command strings (explicit seeds,
    stacked flag, --max-cells = grid, the K/DST eager attempt-timeout),
  * CONFIRM — build_confirm_variants keeps PRODUCTION PCA-Ridge (disable_pca=False) and
    the confirm command's stacked/eager regimes,
  * REPORT — the consolidated per-position render/merge + combined drop-set, and that
    the spec builders construct collection Specs WITHOUT importing the env-pinned spec,
  * the default substage/confirm CLI paths touch NO AWS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tuning import feature_groups as fg
from src.tuning import feature_selection as fs
from src.tuning import feature_selection_stage2 as fs2

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_real(position: str) -> dict:
    return json.loads(
        (_REPO_ROOT / "todo" / "feature_selection" / f"{position.lower()}.json").read_text()
    )


def _effects(families, mae_by_family, models=("Ridge",)):
    """A Stage-1-shaped effects dict: one mean_effect per family (rmse mirrors mae)."""
    out: dict = {}
    for m in models:
        out[m] = {"mae": {}, "rmse": {}}
        for f in families:
            v = mae_by_family.get(f, 0.10)
            out[m]["mae"][f] = {"mean_effect": v, "std_effect": 0.0, "n_seeds": 24}
            out[m]["rmse"][f] = {"mean_effect": v, "std_effect": 0.0, "n_seeds": 24}
    return out


def _payload(position, *, effects, suggested_cut):
    return {
        "position": position,
        "spec": "src.tuning.ab_feature_screen_extended",
        "seeds": [42],
        "noise_floor": 0.02,
        "effects": effects,
        "suggested_cut": suggested_cut,
        "static_audit_flags": {},
    }


def _pick(position, family, *, run_id="rid"):
    picks, _ = fs2.select_subscreens(position, _load_real(position))
    p = next(p for p in picks if p.family == family)
    p.run_id = run_id
    return p


# --------------------------------------------------------------------------- #
# SELECT — the Comprehensive rule
# --------------------------------------------------------------------------- #
def test_select_comprehensive_rule_synthetic():
    families = [
        "rolling",
        "prior_season",
        "trend",
        "share",
        "matchup",
        "defense",
        "contextual",
        "weather_vegas",
        "specific",
    ]
    mae = {f: 0.10 for f in families}  # all KEEP by default
    mae["trend"] = -0.07  # (trend is always-zoom anyway)
    mae["share"] = -0.05  # drop-candidate, NOT always-zoom -> selected
    payload = _payload("RB", effects=_effects(families, mae), suggested_cut=["trend", "share"])

    picks, skipped = fs2.select_subscreens("RB", payload)
    names = {p.family for p in picks}
    skipped_names = {s["family"] for s in skipped}

    # large/heterogeneous families are zoomed regardless of a KEEP verdict
    assert {"rolling", "prior_season", "specific", "trend"} <= names
    # a drop-candidate (in suggested_cut) that is NOT always-zoom is still selected
    assert "share" in names
    # clean all-signal KEEP families (not always-zoom, not drop-cand) are skipped
    assert "contextual" in skipped_names
    assert "weather_vegas" in skipped_names
    # every skill pick is stacked 24-seed [42..65]
    for p in picks:
        assert p.stacked and p.seeds == list(range(42, 66))
        assert p.cells == p.n_variants * 24


def test_select_real_rb_skill_structural():
    picks, skipped = fs2.select_subscreens("RB", _load_real("RB"))
    names = {p.family for p in picks}
    # always-zoom large families + the cross-position drop-candidate are selected
    assert {"rolling", "prior_season", "specific", "trend"} <= names
    for p in picks:
        assert p.stacked and len(p.seeds) == fs2.DEFAULT_STACKED_N
        assert p.n_subgroups >= 2 and p.cells == p.n_variants * len(p.seeds)


def test_select_real_k_eager_and_atomic_skip():
    picks, skipped = fs2.select_subscreens("K", _load_real("K"))
    skipped_names = {s["family"] for s in skipped}
    # fg_distance is a single column (avg_fg_distance_L3) -> atomic, cannot decompose
    assert "fg_distance" in skipped_names
    for p in picks:
        assert not p.stacked and len(p.seeds) == fs2.DEFAULT_KDST_N
        assert p.n_subgroups >= 2


def test_select_overrides_only_skip_max():
    payload = _load_real("RB")
    only = fs2.select_subscreens("RB", payload, only=["trend"])[0]
    assert {p.family for p in only} == {"trend"}

    skipped_out = fs2.select_subscreens("RB", payload, skip=["trend", "rolling"])[0]
    assert {"trend", "rolling"}.isdisjoint({p.family for p in skipped_out})

    capped, spilled = fs2.select_subscreens("RB", payload, max_families=2)
    assert len(capped) == 2
    assert any("max-families" in s["reason"] for s in spilled)


# --------------------------------------------------------------------------- #
# LAUNCH — command + smoke strings
# --------------------------------------------------------------------------- #
def test_subscreen_command_skill_stacked():
    p = _pick("RB", "trend", run_id="RID")
    cmd = fs2.subscreen_launch_command(p, image_sha="abc1234def")
    # the env is set BOTH locally (prefix, for launch_ab's own resolve_spec) and via
    # --env (for the Batch container) — see subscreen_launch_command's docstring.
    assert cmd.startswith("FF_SUBSCREEN_POSITION=RB FF_SUBSCREEN_FAMILY=trend python -m")
    assert "--spec src.tuning.ab_feature_subscreen" in cmd
    assert "--env FF_SUBSCREEN_POSITION=RB" in cmd and "--env FF_SUBSCREEN_FAMILY=trend" in cmd
    assert "--stacked-seeds" in cmd
    assert "--attempt-timeout" not in cmd  # stacked jobs are short
    assert f"--max-cells {p.cells}" in cmd
    assert "--run-id RID" in cmd and "--image-sha abc1234def" in cmd
    assert cmd.count("--seeds") == 1 and " 42 43 44" in cmd


def test_subscreen_command_kdst_eager_has_attempt_timeout():
    p = _pick("K", "game_context", run_id="RID")
    cmd = fs2.subscreen_launch_command(p)
    assert cmd.startswith("FF_SUBSCREEN_POSITION=K FF_SUBSCREEN_FAMILY=game_context python -m")
    assert "--stacked-seeds" not in cmd
    assert f"--attempt-timeout {fs2.EAGER_ATTEMPT_TIMEOUT_S}" in cmd
    assert "--image-sha" not in cmd  # omitted when no sha given


def test_smoke_command_one_risky_arm():
    p = _pick("RB", "trend", run_id="RID")
    cmd = fs2.smoke_command(p)
    assert f"--only {p.riskiest_variant}" in cmd
    assert f"--seeds {p.seeds[0]}" in cmd and " --seeds " in cmd
    assert "--max-cells 2" in cmd and "--run-id RID-smoke" in cmd
    assert "--stacked-seeds" not in cmd  # 1-seed smoke is eager


def test_estimate_cost_positive_and_mode_dependent():
    rb = _pick("RB", "trend", run_id="r")  # stacked
    k = _pick("K", "game_context", run_id="r")  # eager
    assert fs2.estimate_cost(rb) > 0 and fs2.estimate_cost(k) > 0


def test_split_env_prefixed_command_for_exec():
    # the leading KEY=VAL env-prefix is split off (so --exec can run the command
    # via subprocess argv); the --env flags stay in argv for launch_ab/the container.
    cmd = fs2.subscreen_launch_command(_pick("RB", "trend", run_id="RID"))
    env, argv = fs2.split_env_prefixed_command(cmd)
    assert env == {"FF_SUBSCREEN_POSITION": "RB", "FF_SUBSCREEN_FAMILY": "trend"}
    assert argv[:3] == ["python", "-m", "src.tuning.launch_ab"]
    assert "--env" in argv and "FF_SUBSCREEN_FAMILY=trend" in argv

    cenv, cargv = fs2.split_env_prefixed_command(
        fs2.confirm_launch_command("K", ["a", "b"], run_id="r1")
    )
    assert cenv == {"FF_CONFIRM_POSITION": "K", "FF_CONFIRM_DROP_COLS": "a,b"}
    assert cargv[0] == "python"


# --------------------------------------------------------------------------- #
# CONFIRM — production PCA-Ridge variant builder + command regimes
# --------------------------------------------------------------------------- #
def test_build_confirm_variants_keeps_production_pca():
    variants, row_drops = fg.build_confirm_variants(frozenset({"a", "b"}))
    assert [v.name for v in variants] == ["baseline", "drop_confirmed"]
    assert variants[0].cfg_mutator is None  # true production config, untouched
    drop = variants[1]
    assert drop.expect_ridge_identical is False  # a real drop MUST move Ridge (#1172)
    cfg = {"get_feature_columns_fn": lambda: ["a", "b", "c"], "ridge_pca_components": 80}
    drop.cfg_mutator(cfg)
    assert cfg["get_feature_columns_fn"]() == ["c"]  # a, b dropped together
    assert cfg["ridge_pca_components"] == 80  # PCA PRESERVED (production-faithful)
    assert row_drops == {"drop_confirmed": frozenset({"confirmed_drop"})}


def test_build_confirm_variants_empty_raises():
    with pytest.raises(ValueError):
        fg.build_confirm_variants(frozenset())


def test_confirm_command_regimes():
    skill = fs2.confirm_launch_command("RB", ["b", "a"], run_id="r1")
    assert skill.startswith("FF_CONFIRM_POSITION=RB FF_CONFIRM_DROP_COLS=a,b python -m")
    assert "--spec src.tuning.ab_feature_confirm" in skill
    assert "--env FF_CONFIRM_DROP_COLS=a,b" in skill  # sorted, comma-joined
    assert "--stacked-seeds" in skill and "--max-cells 48" in skill  # 2 variants x 24 seeds

    skill_eager = fs2.confirm_launch_command("RB", ["a", "b"], eager=True, run_id="r1")
    assert "--stacked-seeds" not in skill_eager and "--max-cells 16" in skill_eager  # 2 x 8

    k = fs2.confirm_launch_command("K", ["a", "b"], run_id="r1")
    assert "--stacked-seeds" not in k and "--max-cells 16" in k  # K can't stack


def test_confirm_regime_resolution():
    assert fs2.confirm_regime("RB") == (True, list(range(42, 66)))
    assert fs2.confirm_regime("RB", eager=True) == (False, list(range(42, 50)))
    assert fs2.confirm_regime("K") == (False, list(range(42, 50)))


# --------------------------------------------------------------------------- #
# Spec construction — direct, no env-pinned import (module-cache safety)
# --------------------------------------------------------------------------- #
def test_build_subscreen_spec_no_env_pin():
    spec_a, gc_a, rd_a = fs2.build_subscreen_spec("RB", "matchup", [42, 43])
    spec_b, gc_b, rd_b = fs2.build_subscreen_spec("RB", "trend", [42, 43])
    # two different families resolved in ONE process give DIFFERENT sub-groups (the
    # env-import gotcha would pin both to the first family).
    assert set(gc_a) != set(gc_b)
    assert spec_a.dotted == fs2.SUBSCREEN_SPEC and spec_a.positions == ["RB"]
    assert spec_a.seeds == [42, 43] and spec_a.baseline == "baseline"
    assert "baseline" in spec_a.variants and len(spec_a.variants) == 1 + len(
        fg.build_drop_variants(gc_a)[1]
    )


def test_build_confirm_spec():
    spec, names, rd = fs2.build_confirm_spec("RB", ["a", "b"], [42, 43, 44])
    assert spec.dotted == fs2.CONFIRM_SPEC and spec.seeds == [42, 43, 44]
    assert list(spec.variants) == ["baseline", "drop_confirmed"]
    assert names == ["confirmed_drop"] and rd == {"drop_confirmed": frozenset({"confirmed_drop"})}


def test_collect_effects_reuses_collector(monkeypatch):
    spec, gc, rd = fs2.build_subscreen_spec("RB", "matchup", [42, 43])
    cells = [
        {
            "ok": True,
            "position": "RB",
            "variant": v,
            "seed": s,
            "metrics": {"Ridge": {"mae": 4.5, "rmse": 6.5}},
        }
        for v in spec.variants
        for s in (42, 43)
    ]
    monkeypatch.setattr("src.tuning.launch_ab.collect_results", lambda *a, **k: cells)
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: object())
    eff = fs2.collect_effects(spec, "RB", list(gc), rd, "run1")
    assert "Ridge" in eff and set(eff["Ridge"]) == {"mae", "rmse"}


# --------------------------------------------------------------------------- #
# REPORT — consolidated render/merge
# --------------------------------------------------------------------------- #
def _family_report(family, group_cols, *, drop_grp):
    names = list(group_cols)
    eff: dict = {"Ridge": {"mae": {}, "rmse": {}}, "LightGBM": {"mae": {}, "rmse": {}}}
    for g in names:
        v = -0.05 if g == drop_grp else 0.10
        for m in eff:
            eff[m]["mae"][g] = {"mean_effect": v, "std_effect": 0.0, "n_seeds": 3}
            eff[m]["rmse"][g] = {"mean_effect": v, "std_effect": 0.0, "n_seeds": 3}
    return {
        "family": family,
        "effects": eff,
        "group_cols": group_cols,
        "run_id": "rid",
        "seeds": [42, 43, 44],
    }


def test_combined_drop_cols_union_of_subcuts():
    frs = [
        _family_report(
            "share", {"x": frozenset({"x1", "x2"}), "y": frozenset({"y1"})}, drop_grp="x"
        ),
        _family_report("matchup", {"z": frozenset({"z1"}), "w": frozenset({"w1"})}, drop_grp="z"),
    ]
    assert fs2.combined_drop_cols(frs) == ["x1", "x2", "z1"]


def test_render_and_write_stage2_report(tmp_path):
    frs = [
        _family_report(
            "share", {"x": frozenset({"x1", "x2"}), "y": frozenset({"y1"})}, drop_grp="x"
        ),
        _family_report("matchup", {"z": frozenset({"z1"}), "w": frozenset({"w1"})}, drop_grp="z"),
    ]
    md = fs2.render_stage2_md("RB", frs)
    assert "Stage-2" in md
    assert "## `share` sub-groups" in md and "## `matchup` sub-groups" in md
    assert "Ridge MAE" in md and "LightGBM RMSE" in md
    assert "confirm --position RB --from-stage2" in md
    assert "apply --position RB --drop x1 x2 z1" in md

    md_path, json_path = fs2.write_stage2_report("RB", frs, out_dir=str(tmp_path))
    assert Path(md_path).is_file()
    payload = json.loads(Path(json_path).read_text())
    assert payload["stage"] == 2 and payload["position"] == "RB"
    assert payload["combined_suggested_drop_cols"] == ["x1", "x2", "z1"]
    assert set(payload["families"]) == {"share", "matchup"}
    assert payload["families"]["share"]["suggested_drop_cols"] == ["x1", "x2"]


def test_render_confirm_report_verdicts():
    eff = {
        "Ridge": {
            "mae": {"confirmed_drop": {"mean_effect": -0.01, "std_effect": 0.0, "n_seeds": 24}},
            "rmse": {"confirmed_drop": {"mean_effect": -0.02, "std_effect": 0.0, "n_seeds": 24}},
        },
    }
    md = fs2.render_confirm_md("RB", ["a", "b"], "rid", [42, 43], eff)
    assert "Stage-3 confirm" in md and "PCA-Ridge ON" in md
    assert "Confirmed" in md  # neutral on production -> safe to drop


# --------------------------------------------------------------------------- #
# CLI wiring + no-AWS on the default paths
# --------------------------------------------------------------------------- #
def test_parser_dispatches_stage2_subcommands():
    p = fs._build_parser()
    assert p.parse_args(["substage", "--positions", "RB"]).func is fs.cmd_substage
    assert p.parse_args(["substage-report"]).func is fs.cmd_substage_report
    assert p.parse_args(["confirm", "--position", "RB", "--drop", "x"]).func is fs.cmd_confirm
    assert p.parse_args(["confirm-report", "--position", "RB"]).func is fs.cmd_confirm_report


def test_substage_default_path_touches_no_aws(monkeypatch, tmp_path):
    import boto3

    def _boom(*a, **k):
        raise AssertionError("the default substage path must not touch AWS")

    monkeypatch.setattr(boto3, "client", _boom)
    out_plan = tmp_path / "plan.json"
    rc = fs.main(["substage", "--positions", "RB", "K", "--out-plan", str(out_plan)])
    assert rc == 0 and out_plan.is_file()
    plan = json.loads(out_plan.read_text())
    assert plan["stage"] == 2 and plan["picks"]
    # run-ids were assigned + recorded for the report to collect later
    assert all(p["run_id"] for p in plan["picks"])


def test_confirm_default_path_touches_no_aws(monkeypatch, tmp_path):
    import boto3

    def _boom(*a, **k):
        raise AssertionError("the default confirm path must not touch AWS")

    monkeypatch.setattr(boto3, "client", _boom)
    rc = fs.main(
        ["confirm", "--position", "RB", "--drop", "trend_targets", "--stage2-dir", str(tmp_path)]
    )
    assert rc == 0
    recorded = json.loads((tmp_path / fs2.CONFIRM_PLAN_FILE).read_text())
    assert recorded["RB"]["drop_cols"] == ["trend_targets"]
    assert recorded["RB"]["stacked"] is True
