"""Paired, player-clustered intervals decide whether a comparison row has a winner."""

import numpy as np
import pandas as pd
import pytest

from src.shared.comparison_uncertainty import METHOD, group_gap_intervals

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
