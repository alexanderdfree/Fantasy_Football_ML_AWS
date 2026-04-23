"""Registry parity checks — the inference registry must wire every relevant
per-position config knob through to ``nn_kwargs``.

Regression guard for the "silent override drop" bug: WR's config defined
``WR_NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 64}`` but the registry's WR
entry built ``nn_kwargs`` without ``head_hidden_overrides``. Training used
the config directly (cfg.get), so the checkpoint had a 64-wide receptions
head, but ``app.py`` built the inference model from the registry kwargs,
got the default 32-wide head, and ``load_state_dict`` tripped on shape
mismatch. The site went 503 until someone added the missing line.
"""

import importlib

import pytest

from shared.registry import ALL_POSITIONS, INFERENCE_REGISTRY


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_head_hidden_overrides_match_position_config(pos):
    """If a position's config declares non-empty ``{POS}_NN_HEAD_HIDDEN_OVERRIDES``,
    the same dict must be plumbed into ``INFERENCE_REGISTRY[pos]["nn_kwargs"]``.

    An empty dict / missing attr means "no overrides" — registry may omit
    the kwarg (MultiHeadNet defaults ``head_hidden_overrides=None``).
    """
    config_module = importlib.import_module(f"{pos}.{pos.lower()}_config")
    cfg_overrides = getattr(config_module, f"{pos}_NN_HEAD_HIDDEN_OVERRIDES", None)

    reg_overrides = INFERENCE_REGISTRY[pos]["nn_kwargs"].get("head_hidden_overrides")

    if cfg_overrides:
        assert reg_overrides == cfg_overrides, (
            f"{pos}: config declares {cfg_overrides!r} but registry passes "
            f"{reg_overrides!r} — inference model will build the wrong head "
            f"widths and load_state_dict will fail on the trained checkpoint."
        )
    else:
        # None or {} → registry's absence (or empty/None) is fine.
        assert not reg_overrides, (
            f"{pos}: config has no overrides but registry passes {reg_overrides!r}"
        )
