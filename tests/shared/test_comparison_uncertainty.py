"""Paired, player-clustered intervals decide whether a comparison row has a winner."""

import numpy as np
import pandas as pd
import pytest

from src.shared.comparison_uncertainty import (
    METHOD,
    VERDICT_POLICY,
    group_gap_intervals,
    served_gap,
)

pytestmark = pytest.mark.unit

MODELS = ("ridge", "nn", "attn_nn", "lgbm")
EXPERTS = ("rotowire", "espn")
COLUMNS = {source: f"{source}_pred" for source in (*MODELS, *EXPERTS)}


def slate(errors: dict, players=60, weeks=5, seed=0):
    """Truth plus per-source errors; ``errors`` maps source -> callable(rng, n)."""
    rng = np.random.default_rng(seed)
    n = players * weeks
    frame = pd.DataFrame(
        {
            "player_id": np.repeat([f"p{i:03}" for i in range(players)], weeks),
            "actual": rng.gamma(2.0, 4.0, size=n),
        }
    )
    for source, error in errors.items():
        frame[COLUMNS[source]] = frame["actual"] + error(rng, n)
    return frame


def run(frame, **kwargs):
    return group_gap_intervals(frame, "actual", COLUMNS, MODELS, EXPERTS, **kwargs)


def test_seeded_intervals_are_reproducible_and_json_ready():
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    first, second = run(frame), run(frame)
    assert first == second
    assert first["status"] == "available" and first["players"] == 60 and first["n"] == 300
    assert set(first["mae"]["models"]) == set(MODELS)
    assert METHOD["replicates"] == 2000 and METHOD["confidence"] == 0.95


def test_best_of_four_selection_is_inside_the_interval():
    # Equal skill: the best of four noisy models usually looks better than the
    # experts by luck alone. Taking the minimum inside each replicate keeps that
    # post-hoc pick from reading as a win.
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    gaps = run(frame)
    assert gaps["winner"] == "tie"
    for metric in ("mae", "rmse"):
        low, high = gaps[metric]["ci"]
        assert low < 0 < high


def test_metric_disagreement_is_a_tie():
    # Models are exact on 90% of rows and badly wrong on the rest (better MAE);
    # experts are uniformly off by 3 (better RMSE). No winner is declared.
    def spiky(rng, n):
        return np.where(rng.random(n) < 0.9, 0.0, 20.0)

    errors = {model: spiky for model in MODELS}
    errors.update({expert: (lambda rng, n: np.full(n, 3.0)) for expert in EXPERTS})
    gaps = run(slate(errors, players=200))
    assert gaps["mae"]["verdict"] == "models"
    assert gaps["rmse"]["verdict"] == "experts"
    assert gaps["winner"] == "tie"


def test_rows_missing_any_source_are_dropped_for_every_source():
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    frame.loc[:4, COLUMNS["espn"]] = np.nan
    frame.loc[5, COLUMNS["ridge"]] = np.inf
    assert run(frame)["n"] == 294


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda f: f.drop(columns=["player_id"]), "no_common_rows"),
        (lambda f: f.iloc[:0], "no_common_rows"),
        (lambda f: f[f.player_id.eq("p000")], "too_few_players"),
    ],
)
def test_unsupported_samples_are_explicitly_unavailable(mutate, reason):
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    assert run(mutate(frame)) == {"status": "unavailable", "reason": reason}


def test_a_missing_group_is_unavailable_not_a_win():
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    only_models = group_gap_intervals(frame, "actual", COLUMNS, MODELS, ())
    assert only_models == {"status": "unavailable", "reason": "model_or_expert_group_missing"}


def test_served_model_verdict_does_not_inherit_the_best_of_four():
    # Ridge is exact, so the family's best-of-four wins outright; the served
    # model (attention) is worse than the experts on both metrics and loses.
    errors = {
        "ridge": lambda rng, n: np.zeros(n),
        "nn": lambda rng, n: rng.normal(0, 3, n),
        "attn_nn": lambda rng, n: rng.normal(0, 6, n),
        "lgbm": lambda rng, n: rng.normal(0, 3, n),
        "rotowire": lambda rng, n: rng.normal(0, 3, n),
        "espn": lambda rng, n: rng.normal(0, 3, n),
    }
    gaps = run(slate(errors, players=200))
    assert gaps["winner"] == "models"
    served = served_gap(gaps, "attn_nn")
    assert served["status"] == "available" and served["model"] == "attn_nn"
    assert served["requested"] == "attn_nn" and served["fallback"] is False
    assert served["winner"] == "experts"
    for metric in ("mae", "rmse"):
        assert served[metric]["verdict"] == "experts" and served[metric]["ci"][0] > 0
        assert served[metric]["best_expert"] in EXPERTS
        assert (
            served[metric]["minus_best_expert"]
            == gaps[metric]["models"]["attn_nn"]["minus_best_expert"]
        )
    assert served_gap(gaps, "ridge")["winner"] == "models"


def test_served_gap_walks_the_ranking_chain_like_the_board():
    # The board ranks by the first available forecast, so a chain whose first
    # model is not graded falls through to the next one and says so.
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    gaps = run(frame)
    served = served_gap(gaps, ("tabpfn", "attn_nn", "nn"))
    assert served["status"] == "available"
    assert served["model"] == "attn_nn" and served["requested"] == "tabpfn"
    assert served["fallback"] is True
    assert served_gap(gaps, ("tabpfn", "enet")) == {
        "status": "unavailable",
        "reason": "served_model_not_graded",
        "model": "tabpfn",
    }


def test_served_model_gap_needs_a_chain_and_an_available_report():
    frame = slate({source: lambda rng, n: rng.normal(0, 3, n) for source in COLUMNS})
    gaps = run(frame)
    for chain in (None, (), ""):
        assert served_gap(gaps, chain) == {
            "status": "unavailable",
            "reason": "served_model_unknown",
            "model": None,
        }
    unavailable = run(frame.iloc[:0])
    assert served_gap(unavailable, "ridge") == {
        "status": "unavailable",
        "reason": "no_common_rows",
        "model": "ridge",
    }


def test_verdict_policy_is_separate_from_the_bootstrap_settings():
    # The Timeline reuses METHOD for per-model MAE-only edges; the served-model
    # rule and the no-verdict cohorts belong to the Comparison tab only.
    assert not {"headline_gap", "winner_rule", "no_verdict_cohorts"} & set(METHOD)
    assert VERDICT_POLICY["headline_gap"] == "served_model_minus_best_expert"
    assert set(VERDICT_POLICY["no_verdict_cohorts"]) == {"weekly_reference_top24", "top30", "top12"}
