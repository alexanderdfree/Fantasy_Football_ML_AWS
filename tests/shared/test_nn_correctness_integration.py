"""Combined checkpoint policies must survive legacy warm starts together."""

import numpy as np
import pytest
import torch

from src.shared.neural_net import (
    GatedHead,
    PoissonLogRateHead,
    initialize_poisson_heads,
    load_warm_start_state,
)

pytestmark = pytest.mark.unit


def test_legacy_warm_start_preserves_both_requested_output_links():
    legacy = torch.nn.Module()
    legacy.heads = torch.nn.ModuleDict(
        {
            "fumbles_lost": torch.nn.Sequential(torch.nn.Linear(2, 1)),
            "receptions": GatedHead(2, correct_ztnb_mean=False),
        }
    )
    with torch.no_grad():
        legacy.heads["fumbles_lost"][-1].bias.fill_(-100)

    current = torch.nn.Module()
    current.heads = torch.nn.ModuleDict(
        {
            "fumbles_lost": PoissonLogRateHead(torch.nn.Linear(2, 1)),
            "receptions": GatedHead(2, correct_ztnb_mean=True),
        }
    )
    initialize_poisson_heads(current, {"fumbles_lost": np.array([0.0, 1.0, 0.0, 0.0])})
    load_warm_start_state(current, legacy.state_dict())

    count = current.heads["fumbles_lost"]
    assert count.uses_log_rate and count._log_rate_version.item()
    assert count[-1].bias.item() == pytest.approx(np.log(0.25))
    assert current.heads["receptions"].correct_ztnb_mean
    assert current.heads["receptions"]._ztnb_mean_version.item() == 1
    log_rate = count(torch.zeros(1, 2)).squeeze()
    loss = torch.nn.functional.poisson_nll_loss(log_rate, torch.ones(()), log_input=True)
    loss.backward()
    assert count[-1].bias.grad.item() < 0

    # Ordinary inference loading still preserves the saved legacy interpretation.
    current.load_state_dict(legacy.state_dict())
    assert not count.uses_log_rate
    assert not current.heads["receptions"].correct_ztnb_mean
