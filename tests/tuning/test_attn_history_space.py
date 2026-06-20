"""Tests for the attention game-history-branch search space + tune_nn --scope history.

Two layers:
  * ``src/tuning/attn_history_space.py`` pure helpers — bundle resolution, the
    drift contract (all optional bundles == production ``attn_history_stats``),
    core-always-on, and the windowed-column stop-rule guard.
  * ``tune_nn`` history-scope wiring — sampling produces a valid history-scope
    override set (no nn_* backbone keys), validation accepts it and rejects
    full-scope keys, and the FrozenTrial round-trip resolves the token bundles.
"""

from unittest.mock import MagicMock

import optuna
import pytest

from src.shared.registry import get_config
from src.tuning import attn_history_space as ahs
from src.tuning import tune_nn

pytestmark = pytest.mark.unit

_POSITIONS = ["QB", "RB", "WR", "TE"]


# ---------------------------------------------------------------------------
# attn_history_space pure helpers
# ---------------------------------------------------------------------------


def test_supported_positions_are_flat_history_set():
    assert set(ahs.supported_positions()) == set(_POSITIONS)
    assert ahs.is_supported("rb") and ahs.is_supported("RB")
    assert not ahs.is_supported("K")
    assert not ahs.is_supported("DST")


def test_seq_len_choices_include_production_and_are_positive_ints():
    assert 17 in ahs.SEQ_LEN_CHOICES  # production default
    assert all(
        isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in ahs.SEQ_LEN_CHOICES
    )
    assert sorted(set(ahs.SEQ_LEN_CHOICES)) == ahs.SEQ_LEN_CHOICES


@pytest.mark.parametrize("pos", _POSITIONS)
def test_all_optional_bundles_reproduce_production(pos):
    """DRIFT CONTRACT: core + every optional bundle must equal the position's
    production ``attn_history_stats``. If a config token set changes without a
    matching bundle-map update, this fails loudly (set-equality catches any
    add/remove; the list-equality below also pins the order)."""
    production = list(get_config(pos)["attn_history_stats"])
    resolved = ahs.production_history_stats(pos)
    assert set(resolved) == set(production), (
        f"{pos} bundle map drifted from src/{pos.lower()}/config.py attn_history_stats; "
        f"only-in-bundles={set(resolved) - set(production)} "
        f"only-in-config={set(production) - set(resolved)}"
    )
    # Contiguous bundles in production order => all-on resolves verbatim.
    assert resolved == production


@pytest.mark.parametrize("pos", _POSITIONS)
def test_core_always_included_and_is_minimal(pos):
    core = ahs.core_stats(pos)
    assert core, "core bundle must be non-empty"
    # Empty optional selection => exactly the core tokens.
    assert ahs.resolve_history_stats(pos, []) == core
    # Core is a subset of every resolution.
    full = ahs.resolve_history_stats(pos, ahs.optional_bundles(pos))
    assert set(core) <= set(full)
    # "core" is not itself a toggleable optional bundle.
    assert ahs.CORE_BUNDLE not in ahs.optional_bundles(pos)


@pytest.mark.parametrize("pos", _POSITIONS)
def test_resolve_dedupes_and_rejects_unknown_bundle(pos):
    opt = ahs.optional_bundles(pos)
    # A subset resolves to core + that bundle, no duplicates.
    one = ahs.resolve_history_stats(pos, opt[:1])
    assert len(one) == len(set(one))
    with pytest.raises(KeyError, match="unknown"):
        ahs.resolve_history_stats(pos, ["does_not_exist"])


@pytest.mark.parametrize("pos", _POSITIONS)
def test_assert_raw_per_game_accepts_production(pos):
    # No production token should look windowed/expanding/rolling.
    ahs.assert_raw_per_game(ahs.production_history_stats(pos))


@pytest.mark.parametrize(
    "col",
    [
        "rushing_yards_l5",
        "targets_l3",
        "receptions_ewma",
        "snap_pct_raw_expanding_mean",
        "carries_roll3",
        "snap_pct_trend",
        "rushing_yards_career",
        "points_ytd",
        "targets_mean",
        "rush_yards_avg",
    ],
)
def test_assert_raw_per_game_rejects_windowed(col):
    with pytest.raises(ValueError, match="raw per-game"):
        ahs.assert_raw_per_game(["rushing_yards", col])


# ---------------------------------------------------------------------------
# tune_nn --scope history wiring
# ---------------------------------------------------------------------------


def _ask_history_overrides(pos: str, seed: int = 0) -> tuple[optuna.Trial, dict]:
    """Drive _sample_overrides in history scope (v2 isolation) with a real
    Optuna trial. The v2 history space samples only seq_len + token bundles
    (no d_model/n_heads), so it never prunes; the retry loop is just defensive."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    for _ in range(20):
        trial = study.ask()
        try:
            return trial, tune_nn._sample_overrides(trial, "history", pos)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    raise AssertionError("could not sample a history trial")


def test_history_sample_overrides_are_isolated_to_two_axes():
    """v2 isolation: history sampling returns EXACTLY the two game-history axes —
    attn_max_seq_len + attn_history_stats — and freezes the whole production
    recipe (no attn sizing, lr, batch, scheduler, or nn_* keys)."""
    trial, overrides = _ask_history_overrides("RB")
    assert set(overrides) == {"attn_max_seq_len", "attn_history_stats"}
    assert set(overrides) == tune_nn._HISTORY_REQUIRED_KEYS
    # No frozen-recipe key leaks into the override set.
    assert not any(k.startswith("nn_") for k in overrides)
    assert not any(
        k in overrides
        for k in ("attn_d_model", "attn_n_heads", "attn_lr", "attn_batch_size", "scheduler_type")
    )
    # The trial params carry only seq_len + the histbundle_* booleans (nothing
    # from the frozen recipe gets a suggest_* call).
    assert "attn_max_seq_len" in trial.params
    assert all(k == "attn_max_seq_len" or k.startswith("histbundle_") for k in trial.params)
    # Sequence length is one of the candidate values.
    assert overrides["attn_max_seq_len"] in ahs.SEQ_LEN_CHOICES
    # Token set is a subset of production that always contains core.
    stats = overrides["attn_history_stats"]
    assert set(ahs.core_stats("RB")) <= set(stats)
    assert set(stats) <= set(ahs.production_history_stats("RB"))
    # The resolved list is recorded as a user_attr for round-trip.
    assert trial.user_attrs["attn_history_stats"] == stats
    # And it validates cleanly under history scope.
    tune_nn._validate_overrides(overrides, "history")


@pytest.mark.parametrize(
    "frozen_key,value",
    [
        ("nn_dropout", 0.2),  # static-backbone key
        ("attn_d_model", 32),  # v1 REQUIRED this; v2 freezes it
        ("attn_lr", 1e-3),  # the v1 confound — frozen in v2
        ("scheduler_type", "onecycle"),
    ],
)
def test_history_validate_rejects_frozen_recipe_keys(frozen_key, value):
    """v2 isolation: any frozen-recipe key (sizing, lr, scheduler, nn_*) is an
    unknown key under history scope — even the ones v1 sampled/required."""
    _, overrides = _ask_history_overrides("WR")
    overrides[frozen_key] = value
    with pytest.raises(ValueError, match="unknown keys"):
        tune_nn._validate_overrides(overrides, "history")


def test_history_validate_requires_history_keys():
    _, overrides = _ask_history_overrides("QB")
    overrides.pop("attn_history_stats")
    with pytest.raises(ValueError, match="missing keys"):
        tune_nn._validate_overrides(overrides, "history")


def test_history_validate_rejects_windowed_token():
    _, overrides = _ask_history_overrides("TE")
    overrides["attn_history_stats"] = [*overrides["attn_history_stats"], "targets_l5"]
    with pytest.raises(ValueError, match="raw per-game"):
        tune_nn._validate_overrides(overrides, "history")


def test_history_trial_to_params_roundtrip_and_config_lines():
    """A history FrozenTrial: bundle booleans dropped, attn_history_stats taken
    from user_attrs, and the result renders paste-ready config lines."""
    trial, overrides = _ask_history_overrides("RB")
    frozen = MagicMock()
    # Mirror what Optuna stores: scalar params (incl. histbundle_* booleans +
    # attn_max_seq_len) and the resolved list as a user_attr.
    frozen.params = dict(trial.params)
    frozen.user_attrs = {"attn_history_stats": overrides["attn_history_stats"]}
    assert any(k.startswith("histbundle_") for k in frozen.params)

    p = tune_nn._trial_to_params(frozen, "history", "RB")
    assert not any(k.startswith("histbundle_") for k in p)  # booleans dropped
    assert p["attn_history_stats"] == overrides["attn_history_stats"]
    assert p["attn_max_seq_len"] == overrides["attn_max_seq_len"]
    assert not any(k.startswith("nn_") for k in p)
    # v2: the resolved params are exactly the two history axes — nothing else.
    assert set(p) == {"attn_max_seq_len", "attn_history_stats"}

    lines = tune_nn._format_config_lines("RB", p)
    assert "attn_max_seq_len=" in lines
    assert "attn_history_stats=" in lines


def test_history_overrides_build_attention_model_and_forward():
    """End-to-end (data-free) integration: a sampled token SUBSET sizes game_dim
    and the sampled sequence length sizes the positional embedding, and the
    attention model forwards to finite per-target outputs. Proves the two new
    history-branch axes flow through the real array width + model construction
    that a live ``--scope history`` run exercises, without splits/training."""
    import torch

    from src.shared.neural_net import build_multihead_net_with_history

    _, overrides = _ask_history_overrides("RB")
    seq_len = overrides["attn_max_seq_len"]
    game_dim = len(overrides["attn_history_stats"])
    # v2 history freezes the ENTIRE recipe (attn sizing + static backbone), so
    # the factory reads sizing from the production config; overlay only the two
    # searched axes (seq_len + tokens). Enable PE so seq_len sizes the embedding.
    cfg = {**get_config("RB"), **overrides, "attn_positional_encoding": True}
    targets = ["target_a", "target_b"]
    batch, static_dim = 3, 5
    x_static = torch.randn(batch, static_dim)
    x_history = torch.randn(batch, seq_len, game_dim)
    history_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    history_mask[:, 0] = True  # at least one real game per row

    model = build_multihead_net_with_history(
        cfg, static_dim=static_dim, game_dim=game_dim, targets=targets
    )
    model.eval()
    with torch.no_grad():
        preds = model(x_static, x_history, history_mask)
    assert set(preds) == set(targets)
    for pred in preds.values():
        assert pred.shape == (batch,)
        assert torch.isfinite(pred).all()


def test_full_scope_sampling_unchanged():
    """Default (full) scope still samples the static backbone — guards against
    the history refactor leaking into the production search space."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(20):
        trial = study.ask()
        try:
            overrides = tune_nn._sample_overrides(trial)  # default scope == full
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        break
    assert "nn_backbone_layers" in overrides
    assert "attn_max_seq_len" not in overrides
    assert "attn_history_stats" not in overrides
    tune_nn._validate_overrides(overrides)  # full-scope validation


def test_history_storage_namespace_is_separate():
    """History-scope studies must not share the scheduler_v2 namespace."""
    from src.tuning.tune_nn_storage import (
        HISTORY_SEARCH_SPACE_VERSION,
        SEARCH_SPACE_VERSION,
        resolve_search_space_version,
    )

    assert HISTORY_SEARCH_SPACE_VERSION != SEARCH_SPACE_VERSION
    eager = resolve_search_space_version("thread", root=HISTORY_SEARCH_SPACE_VERSION)
    assert eager == "history_v2"
    assert resolve_search_space_version("thread") == SEARCH_SPACE_VERSION
    # Execution-profile suffixes still apply to the history root.
    assert (
        resolve_search_space_version("mps", cuda_graph=True, root=HISTORY_SEARCH_SPACE_VERSION)
        == "history_v2_mps_graph"
    )
