"""Unit tests for issue #720 attention-knob experiment tooling."""

from __future__ import annotations

import optuna
import pytest

from src.tuning import attn_knob_experiments as ake

pytestmark = pytest.mark.unit


def test_attention_knob_inventory_is_the_issue_720_eight():
    assert ake.DEFAULT_SEEDS == (42, 43, 44)
    assert ake.DEFAULT_N_JOBS == 2
    assert ake.KNOB_NAMES == (
        "attn_d_model",
        "attn_n_heads",
        "attn_encoder_hidden_dim",
        "attn_dropout",
        "attn_lr",
        "attn_batch_size",
        "attn_weight_decay",
        "attn_patience",
    )
    assert len(ake.ATTN_KNOBS) == 8


def test_sample_attention_overrides_ranges():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    overrides = ake._sample_attention_overrides(trial)

    assert set(overrides) == set(ake.KNOB_NAMES)
    assert overrides["attn_d_model"] in (16, 24, 32, 48, 64)
    assert overrides["attn_n_heads"] in (1, 2, 4)
    assert overrides["attn_d_model"] % overrides["attn_n_heads"] == 0
    assert overrides["attn_encoder_hidden_dim"] in (0, 16, 32, 64)
    assert 0.0 <= overrides["attn_dropout"] <= 0.3
    assert 1e-4 <= overrides["attn_lr"] <= 1e-2
    assert overrides["attn_batch_size"] in (128, 256, 512, 1024)
    assert 1e-5 <= overrides["attn_weight_decay"] <= 1e-3
    assert 15 <= overrides["attn_patience"] <= 30


def test_make_cfg_disables_unneeded_sibling_models_and_deep_copies():
    base = {
        "train_ridge": True,
        "train_base_nn": True,
        "train_lightgbm": True,
        "train_elasticnet": True,
        "huber_deltas": {"rushing_yards": 15.0},
    }
    cfg = ake._make_cfg(base, {"attn_lr": 0.001}, ridge_sentinel=False)

    assert cfg["attn_lr"] == 0.001
    assert cfg["train_ridge"] is False
    assert cfg["train_base_nn"] is False
    assert cfg["train_lightgbm"] is False
    assert cfg["train_elasticnet"] is False
    assert base["train_ridge"] is True
    cfg["huber_deltas"]["rushing_yards"] = 1.0
    assert base["huber_deltas"]["rushing_yards"] == 15.0


def test_make_cfg_can_keep_ridge_sentinel():
    cfg = ake._make_cfg({}, {}, ridge_sentinel=True)
    assert cfg["train_ridge"] is True
    assert cfg["train_base_nn"] is False


def test_plackett_burman_design_shape_balance_and_orthogonality():
    design = ake.plackett_burman_design(8)
    assert len(design) == 12
    assert all(len(row) == 8 for row in design)

    for col in range(8):
        values = [row[col] for row in design]
        assert sorted(set(values)) == [-1, 1]
        assert sum(values) == 0

    for left in range(8):
        for right in range(left + 1, 8):
            assert sum(row[left] * row[right] for row in design) == 0


def test_plackett_burman_rejects_unsupported_factor_count():
    with pytest.raises(ValueError, match="1..11"):
        ake.plackett_burman_design(12)


def test_doe_overrides_select_low_and_high_levels():
    overrides = ake.doe_overrides([1, -1, 1, -1, 1, -1, 1, -1])

    assert overrides["attn_d_model"] == 64
    assert overrides["attn_n_heads"] == 1
    assert overrides["attn_encoder_hidden_dim"] == 64
    assert overrides["attn_dropout"] == 0.0
    assert overrides["attn_lr"] == 1e-2
    assert overrides["attn_batch_size"] == 128
    assert overrides["attn_weight_decay"] == 1e-3
    assert overrides["attn_patience"] == 15


def test_doe_overrides_rejects_wrong_sign_count():
    with pytest.raises(ValueError, match="expected 8 signs"):
        ake.doe_overrides([1, -1])


def test_estimate_doe_effects_known_main_effect():
    rows = []
    for run_idx, signs in enumerate(ake.plackett_burman_design(8), start=1):
        signs_by_name = dict(zip(ake.KNOB_NAMES, signs, strict=True))
        # Only attn_lr has a true effect: high rows are four points worse than
        # low rows, so the high-minus-low estimate should be exactly +4.
        mae = 10.0 + (2.0 if signs_by_name["attn_lr"] > 0 else -2.0)
        rows.append(
            {
                "seed": 42,
                "run_idx": run_idx,
                "signs": signs_by_name,
                "attn_test_mae": mae,
                "ridge_mae": 4.2,
            }
        )

    effects = ake.estimate_doe_effects(rows)
    assert effects["attn_lr"]["mean_effect"] == pytest.approx(4.0)
    for name in set(ake.KNOB_NAMES) - {"attn_lr"}:
        assert effects[name]["mean_effect"] == pytest.approx(0.0)


def test_ridge_sentinel_flags_attention_only_data_drift():
    rows = [
        {"seed": 42, "ridge_mae": 4.2},
        {"seed": 42, "ridge_mae": 4.2},
        {"seed": 43, "ridge_mae": 4.5},
        {"seed": 43, "ridge_mae": 4.7},
    ]
    assert ake.ridge_sentinel_ok(rows[:2]) is True
    assert ake.ridge_sentinel_ok(rows) is False


def test_dry_run_prints_design_without_training(capsys):
    ake.main(["doe", "--dry-run", "--seeds", "42"])
    out = capsys.readouterr().out
    assert "planned pipeline runs: 12" in out
    assert "plackett_burman_12" in out


def test_doe_cli_threads_ridge_sentinel_flag(monkeypatch):
    seen: list[tuple[bool, int]] = []

    def fake_run_doe(position, seeds, *, ridge_sentinel, n_jobs):
        assert position == "RB"
        assert seeds == [42]
        seen.append((ridge_sentinel, n_jobs))
        return {"ok": True}

    monkeypatch.setattr(ake, "run_doe", fake_run_doe)

    ake.main(["doe", "--seeds", "42", "--no-history"])
    ake.main(["doe", "--seeds", "42", "--ridge-sentinel", "--n-jobs", "3", "--no-history"])

    assert seen == [(False, 2), (True, 3)]


def test_fanova_cli_threads_n_jobs(monkeypatch):
    seen: list[tuple[int, int, bool, int]] = []

    def fake_run_fanova(
        position,
        seeds,
        *,
        n_trials,
        sampler_seed,
        ridge_sentinel,
        n_jobs,
    ):
        assert position == "RB"
        assert seeds == [42]
        seen.append((n_trials, sampler_seed, ridge_sentinel, n_jobs))
        return {"ok": True}

    monkeypatch.setattr(ake, "run_fanova", fake_run_fanova)

    ake.main(
        [
            "fanova",
            "--seeds",
            "42",
            "--n-trials",
            "7",
            "--sampler-seed",
            "99",
            "--ridge-sentinel",
            "--n-jobs",
            "3",
            "--no-history",
        ]
    )

    assert seen == [(7, 99, True, 3)]
