"""Unit tests for the feature-selection layer (feature_groups + the driver).

No training (the screens run on the GPU Batch fleet; local training SIGSEGVs on
the macOS libomp triple-load). Coverage is the pure logic these modules add:

  * grouping: skill family unions, the cross-position screenable guard (ewma is
    QB-only), the K/DST exhaustive partitions, within-family sub-groups,
  * designs: Plackett-Burman (<=11 groups) vs leave-one-out (>11),
  * the column-drop mutator filters BOTH model paths (the #1172 shape),
  * the metric-agnostic main-effects estimator recovers a planted MAE *and* RMSE
    effect and leaves the orthogonal groups at zero,
  * the driver: design reconstruction, per-model/per-metric effects, the
    suggested-cut rule, report rendering (MAE AND RMSE columns), the apply
    drop-block shapes, and the launch_ab --env passthrough.
"""

from __future__ import annotations

import pytest

from src.tuning import ab_feature_screen as CORE
from src.tuning import feature_groups as fg
from src.tuning import feature_selection as fs

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# feature_groups: skill families + the cross-position screenable guard
# --------------------------------------------------------------------------- #
def test_core_families_match_validated_screen():
    """feature_groups.CORE_FAMILIES must stay in sync with the validated core-8
    screen's SCREENED_FAMILIES (a divergence would silently change the design)."""
    assert tuple(fg.CORE_FAMILIES) == tuple(CORE.SCREENED_FAMILIES)


def test_skill_family_columns_union_and_drop_empty():
    cols = fg.skill_family_columns(["rolling", "ewma"], ["RB"])
    assert cols["rolling"]  # RB populates rolling
    assert cols["ewma"] == frozenset()  # RB has no ewma
    pruned = fg.skill_family_columns(["rolling", "ewma"], ["RB"], drop_empty=True)
    assert "ewma" not in pruned and "rolling" in pruned


def test_screenable_excludes_qb_only_ewma():
    """ewma is populated only for QB, so a multi-skill screen must exclude it
    (an empty drop on RB/WR/TE would false-trip the Ridge sentinel)."""
    screenable = fg.screenable_skill_families(fg.EXTENDED_FAMILIES, fg.SKILL_POSITIONS)
    assert "specific" in screenable  # populated everywhere -> screenable
    assert "ewma" not in screenable
    # but QB alone CAN screen its ewma
    assert "ewma" in fg.screenable_skill_families(["ewma"], ["QB"])


# --------------------------------------------------------------------------- #
# feature_groups: K/DST exhaustive partitions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("position", ["K", "DST"])
def test_special_partition_is_exhaustive(position):
    import importlib

    groups = fg.special_family_columns(position)
    covered = set().union(*groups.values())
    all_feats = set(
        importlib.import_module(f"src.{position.lower()}.config").POSITION_CONFIG.all_features
    )
    assert covered == all_feats  # every column in exactly one group, no extras
    # groups are disjoint
    seen: set[str] = set()
    for members in groups.values():
        assert not (seen & members)
        seen |= members


def test_special_partition_raises_on_unknown_column(monkeypatch):
    bad = dict(fg._K_GROUPS)
    bad["bogus"] = ("not_a_real_kicker_column",)
    monkeypatch.setitem(fg._SPECIAL_GROUPS, "K", bad)
    with pytest.raises(ValueError, match="not exhaustive"):
        fg.special_family_columns("K")


# --------------------------------------------------------------------------- #
# feature_groups: sub-family decomposition
# --------------------------------------------------------------------------- #
def test_subfamily_rolling_groups_by_stat_root():
    groups = fg.subfamily_groups("RB", "rolling")
    assert "rushing_yards" in groups and "carries" in groups
    # every rolling_*_rushing_yards_L* column lands under the rushing_yards root
    assert all("rushing_yards" in c for c in groups["rushing_yards"])
    assert all(c.startswith("rolling_") for cols in groups.values() for c in cols)


def test_subfamily_small_family_is_one_group_per_column():
    groups = fg.subfamily_groups("RB", "matchup")
    # matchup is small -> each column is its own sub-group
    assert all(len(cols) == 1 for cols in groups.values())


# --------------------------------------------------------------------------- #
# feature_groups: designs (PB vs LOO)
# --------------------------------------------------------------------------- #
def test_design_pb_for_small_group_count():
    rows = fg.design_for_groups(["a", "b", "c"])
    assert len(rows) == 12  # the 12-run PB design
    assert [n for n, _ in rows] == [f"pb{i:02d}" for i in range(1, 13)]
    # every group is dropped in some rows and kept in others (balanced contrast)
    for grp in ("a", "b", "c"):
        dropped = [n for n, d in rows if grp in d]
        assert 0 < len(dropped) < 12


def test_design_loo_for_large_group_count():
    names = [f"g{i}" for i in range(13)]  # > 11 -> leave-one-out
    rows = fg.design_for_groups(names)
    assert len(rows) == 13
    assert all(len(d) == 1 for _, d in rows)
    assert {next(iter(d)) for _, d in rows} == set(names)


@pytest.mark.parametrize("n", [1, 2])
def test_design_small_n_uses_loo_no_empty_drop(n):
    """PB's tiny designs contain an all-kept row that drops nothing — a
    baseline-identical variant that would false-trip the Ridge sentinel. n<=2
    must fall back to leave-one-out so every row drops exactly one group."""
    names = [f"g{i}" for i in range(n)]
    rows = fg.design_for_groups(names)
    assert len(rows) == n
    assert all(len(d) == 1 for _, d in rows)  # every row drops one group, none empty


@pytest.mark.parametrize(
    "group_cols",
    [
        {"solo": frozenset({"c1"})},  # 1 group
        {"a": frozenset({"c1"}), "b": frozenset({"c2"})},  # 2 groups
    ],
)
def test_build_drop_variants_never_emits_empty_drop(group_cols):
    """Every non-baseline variant must drop >=1 real column (else Ridge Δ=0 trips
    the expect_ridge_identical=False sentinel)."""
    variants, row_drops = fg.build_drop_variants(group_cols)
    assert len(variants) == 1 + len(group_cols)  # baseline + one per group
    for v in variants[1:]:
        cfg = {"get_feature_columns_fn": lambda: ["c1", "c2", "other"]}
        v.cfg_mutator(cfg)
        assert cfg["get_feature_columns_fn"]() != ["c1", "c2", "other"]  # something dropped


# --------------------------------------------------------------------------- #
# feature_groups: the drop mutator filters BOTH model paths (#1172 shape)
# --------------------------------------------------------------------------- #
def test_drop_columns_mutator_filters_both_paths():
    cfg = {
        "get_feature_columns_fn": lambda: ["a", "b", "c"],
        "attn_static_features": ["a", "b"],
    }
    fg.drop_columns_mutator(frozenset({"a"}))(cfg)
    assert cfg["get_feature_columns_fn"]() == ["b", "c"]
    assert cfg["attn_static_features"] == ["b"]


def test_drop_columns_mutator_no_attn_static_key_is_safe():
    cfg = {"get_feature_columns_fn": lambda: ["a", "b"]}  # K/DST may omit it
    fg.drop_columns_mutator(frozenset({"a"}))(cfg)
    assert cfg["get_feature_columns_fn"]() == ["b"]


def test_build_drop_variants_shape_and_sentinel():
    variants, row_drops = fg.build_drop_variants(
        {"x": frozenset({"x1", "x2"}), "y": frozenset({"y1"})}
    )
    assert variants[0].name == "baseline"
    # baseline now carries a drop-nothing PCA-off mutator (shares the screen
    # config), so it is NOT is_baseline_shape — but it must drop no columns.
    assert variants[0].cfg_mutator is not None
    cfg = {"get_feature_columns_fn": lambda: ["x1", "x2", "y1"]}
    variants[0].cfg_mutator(cfg)
    assert cfg["get_feature_columns_fn"]() == ["x1", "x2", "y1"]  # baseline drops nothing
    assert cfg["ridge_pca_components"] is None  # but disables PCA
    for v in variants[1:]:
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is False  # a real drop MUST move Ridge
    assert set(row_drops) == {v.name for v in variants[1:]}


def test_build_drop_variants_prunes_empty_group():
    """An empty group is pruned before designing — a no-op drop would false-trip
    the Ridge sentinel."""
    _, row_drops = fg.build_drop_variants({"x": frozenset({"x1"}), "empty": frozenset()})
    assert all("empty" not in dropped for dropped in row_drops.values())


def test_full_feature_set_skips_all_drop_arm():
    """When the screened groups ARE the whole feature set, the all-drop PB arm
    leaves 0 features and crashes StandardScaler — it must be skipped."""
    cols = {"a": frozenset({"a1"}), "b": frozenset({"b1"}), "c": frozenset({"c1"})}
    _, full = fg.build_drop_variants(cols, full_feature_set=True)
    assert all(d != frozenset({"a", "b", "c"}) for d in full.values())  # no all-drop arm
    # sub-screen (other families remain) keeps the whole-family-drop arm
    _, sub = fg.build_drop_variants(cols, full_feature_set=False)
    assert any(d == frozenset({"a", "b", "c"}) for d in sub.values())


def test_drop_mutator_disables_pca():
    cfg = {"get_feature_columns_fn": lambda: ["a", "b"], "ridge_pca_components": 80}
    fg.drop_columns_mutator(frozenset({"a"}))(cfg)
    assert cfg["ridge_pca_components"] is None  # screen runs Ridge on raw features
    # opt-out keeps PCA for callers that want pure column-dropping
    cfg2 = {"get_feature_columns_fn": lambda: ["a", "b"], "ridge_pca_components": 80}
    fg.drop_columns_mutator(frozenset({"a"}), disable_pca=False)(cfg2)
    assert cfg2["ridge_pca_components"] == 80


# --------------------------------------------------------------------------- #
# feature_groups: metric-agnostic main effects (MAE and RMSE)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric_offset", [0.10, -0.07])
def test_main_effects_recovers_single_planted_effect(metric_offset):
    names = ["a", "b", "c"]
    rows = fg.design_for_groups(names)
    row_drops = dict(rows)
    target = "a"
    variant_seed_value = {
        name: {
            42: 1.0 + (metric_offset if target in drop else 0.0),
            43: 1.0 + (metric_offset if target in drop else 0.0),
        }
        for name, drop in rows
    }
    eff = fg.main_effects(variant_seed_value, row_drops, names)
    assert eff[target]["mean_effect"] == pytest.approx(metric_offset)
    assert eff[target]["std_effect"] == pytest.approx(0.0)
    assert eff[target]["n_seeds"] == 2
    for other in ("b", "c"):
        assert eff[other]["mean_effect"] == pytest.approx(0.0, abs=1e-9)


def test_extract_variant_seed_metric_skips_bad_cells():
    results = [
        {
            "ok": True,
            "position": "RB",
            "variant": "v1",
            "seed": 42,
            "metrics": {"Ridge": {"mae": 4.5, "rmse": 6.5}},
        },
        {
            "ok": True,
            "position": "RB",
            "variant": "v1",
            "seed": 43,
            "metrics": {"Ridge": {"mae": float("nan"), "rmse": 6.6}},
        },  # NaN mae skipped
        {"ok": False, "position": "RB", "variant": "v1", "seed": 7, "metrics": {}},  # not ok
        {
            "ok": True,
            "position": "WR",
            "variant": "v1",
            "seed": 42,
            "metrics": {"Ridge": {"mae": 9.9, "rmse": 1.0}},
        },  # other position
    ]
    mae = fg.extract_variant_seed_metric(results, "RB", "Ridge", "mae")
    assert mae == {"v1": {42: 4.5}}
    rmse = fg.extract_variant_seed_metric(results, "RB", "Ridge", "rmse")
    assert rmse == {"v1": {42: 6.5, 43: 6.6}}


# --------------------------------------------------------------------------- #
# driver: design reconstruction (new specs + the validated core)
# --------------------------------------------------------------------------- #
def test_spec_design_new_spec_uses_row_drops():
    names, row_drops = fs.spec_design("src.tuning.ab_feature_screen_k")
    assert len(names) == 6  # the six K groups
    assert len(row_drops) == 11  # 12 PB rows minus the skipped all-drop arm (pb12)
    assert all(isinstance(v, frozenset) for v in row_drops.values())


def test_spec_design_core_screen_reconstructs_pb():
    names, row_drops = fs.spec_design("src.tuning.ab_feature_screen")
    assert tuple(names) == tuple(CORE.SCREENED_FAMILIES)
    assert len(row_drops) == 12
    # reconstruction matches what the core screen's own estimator would use
    assert all(d <= set(names) for d in row_drops.values())


# --------------------------------------------------------------------------- #
# driver: per-model/per-metric effects + suggested cut + report
# --------------------------------------------------------------------------- #
def _synthetic_results(position, names, row_drops, *, hurt="a", help_="b"):
    """Dropping `hurt` raises MAE/RMSE; dropping `help_` lowers it; else flat.

    Effects are ADDITIVE (both contributions apply when both groups are dropped),
    so the orthogonal PB contrast isolates each group's effect cleanly.
    """
    out = []
    for v, drop in row_drops.items():
        d = (0.20 if hurt in drop else 0.0) + (-0.05 if help_ in drop else 0.0)
        for seed in (42, 123, 7):
            out.append(
                {
                    "ok": True,
                    "position": position,
                    "variant": v,
                    "seed": seed,
                    "metrics": {
                        "Ridge": {"mae": 4.5 + d, "rmse": 6.5 + d, "bias": 0.0, "n": 100},
                        "LightGBM": {"mae": 4.4 + d, "rmse": 6.4 + d, "bias": 0.0, "n": 100},
                    },
                }
            )
    return out


def test_position_effects_has_both_metrics_per_model():
    names = ["a", "b", "c"]
    row_drops = dict(fg.design_for_groups(names))
    results = _synthetic_results("RB", names, row_drops)
    eff = fs.position_effects(results, "RB", names, row_drops)
    assert set(eff) == {"Ridge", "LightGBM"}
    for model in eff:
        assert set(eff[model]) == {"mae", "rmse"}
        assert eff[model]["mae"]["a"]["mean_effect"] == pytest.approx(0.20)
        assert eff[model]["mae"]["b"]["mean_effect"] == pytest.approx(-0.05)
        assert eff[model]["rmse"]["a"]["mean_effect"] == pytest.approx(0.20)


def test_suggested_cut_requires_neutral_for_all_models():
    # group "a" hurts both -> not suggested; "b"/"c" neutral-or-helpful -> suggested
    names = ["a", "b", "c"]
    row_drops = dict(fg.design_for_groups(names))
    results = _synthetic_results("RB", names, row_drops)
    eff = fs.position_effects(results, "RB", names, row_drops)
    cut = fs.suggested_cut(eff, names)
    assert "a" not in cut
    assert "b" in cut and "c" in cut


def test_suggested_cut_excludes_group_that_helps_one_model_hurts_another():
    eff = {
        "Ridge": {
            "mae": {"g": {"mean_effect": -0.10, "std_effect": 0.0, "n_seeds": 3}},
            "rmse": {},
        },
        "LightGBM": {
            "mae": {"g": {"mean_effect": 0.30, "std_effect": 0.0, "n_seeds": 3}},
            "rmse": {},
        },
    }
    assert fs.suggested_cut(eff, ["g"]) == []  # MIXED -> not suggested


def test_render_report_has_mae_and_rmse_and_sections():
    names = ["a", "b", "c"]
    row_drops = dict(fg.design_for_groups(names))
    results = _synthetic_results("RB", names, row_drops)
    eff = fs.position_effects(results, "RB", names, row_drops)
    md = fs.render_report_md("RB", "spec", "run1", [42, 123, 7], eff, names)
    assert "Ridge MAE" in md and "Ridge RMSE" in md
    assert "LightGBM MAE" in md and "LightGBM RMSE" in md
    assert "verdict" in md
    assert "Suggested conservative cut" in md
    assert "Neutral overall ≠ useless" in md


def test_verdict_classifies_keep_drop_mixed():
    keep = {"M": {"mae": {"g": {"mean_effect": 0.30}}}}
    drop = {"M": {"mae": {"g": {"mean_effect": -0.10}}}}
    mixed = {
        "A": {"mae": {"g": {"mean_effect": 0.30}}},
        "B": {"mae": {"g": {"mean_effect": -0.10}}},
    }
    assert fs._verdict(keep, "g", noise=fs.NOISE_FLOOR) == "KEEP"
    assert fs._verdict(drop, "g", noise=fs.NOISE_FLOOR) == "DROP-CAND"
    assert fs._verdict(mixed, "g", noise=fs.NOISE_FLOOR) == "MIXED"


# --------------------------------------------------------------------------- #
# driver: apply drop-block shapes + idempotent strip
# --------------------------------------------------------------------------- #
def test_drop_block_skill_shape():
    block = fs._drop_block("RB", ["trend_targets", "target_share_L3"])
    assert "POSITION_CONFIG.include_features" in block
    assert "POSITION_CONFIG.attn_static_features" in block
    assert '"trend_targets",' in block and '"target_share_L3",' in block
    assert "all_features" not in block  # skill path uses include_features


def test_drop_block_special_shape():
    block = fs._drop_block("K", ["game_wind"])
    assert "POSITION_CONFIG.all_features" in block
    assert "POSITION_CONFIG.contextual_features" in block
    assert "POSITION_CONFIG.attn_static_features" in block
    assert "include_features" not in block  # K/DST use the flat lists


def test_drop_block_regex_strips_for_idempotency():
    base = "x = 1\n"
    once = base + fs._drop_block("RB", ["a"])
    stripped = fs._DROP_BLOCK_RE.sub("\n", once)
    assert "feature_selection drops" not in stripped
    # re-applying yields exactly one block
    twice = stripped.rstrip("\n") + "\n" + fs._drop_block("RB", ["b"])
    assert twice.count(">>> feature_selection drops") == 1


# --------------------------------------------------------------------------- #
# launch_ab --env passthrough
# --------------------------------------------------------------------------- #
def test_parse_env_pairs():
    from src.tuning.launch_ab import _parse_env_pairs

    assert _parse_env_pairs(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    assert _parse_env_pairs(None) == {}
    with pytest.raises(SystemExit):
        _parse_env_pairs(["NOEQUALS"])


def test_submit_ab_job_forwards_extra_env_and_guards_collisions():
    from src.tuning.launch_ab import submit_ab_job

    class _FakeBatch:
        def __init__(self):
            self.kwargs = None

        def submit_job(self, **kw):
            self.kwargs = kw
            return {"jobId": "job-123"}

    fake = _FakeBatch()
    pos, job_id = submit_ab_job(
        "RB",
        spec_dotted="src.tuning.ab_feature_subscreen",
        run_id="r1",
        s3_prefix="ab_runs",
        job_definition="ff-ab-job:1",
        image_sha="abc123",
        seeds=None,
        only=None,
        extra_env={"FF_SUBSCREEN_FAMILY": "rolling"},
        batch_client=fake,
    )
    assert job_id == "job-123"
    env = {e["name"]: e["value"] for e in fake.kwargs["containerOverrides"]["environment"]}
    assert env["FF_SUBSCREEN_FAMILY"] == "rolling"

    with pytest.raises(ValueError, match="collides"):
        submit_ab_job(
            "RB",
            spec_dotted="s",
            run_id="r1",
            s3_prefix="ab_runs",
            job_definition="ff-ab-job:1",
            image_sha="abc",
            seeds=None,
            only=None,
            extra_env={"S3_BUCKET": "evil"},  # managed var -> must raise
            batch_client=fake,
        )
