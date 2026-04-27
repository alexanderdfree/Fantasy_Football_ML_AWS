"""Registry parity checks — the inference registry must wire every relevant
per-position config knob through to ``nn_kwargs``.

Regression guard for the "silent override drop" bug: WR's config defined
``NN_HEAD_HIDDEN_OVERRIDES = {"receptions": 64}`` but the registry's WR
entry built ``nn_kwargs`` without ``head_hidden_overrides``. Training used
the config directly (cfg.get), so the checkpoint had a 64-wide receptions
head, but ``app.py`` built the inference model from the registry kwargs,
got the default 32-wide head, and ``load_state_dict`` tripped on shape
mismatch. The site went 503 until someone added the missing line.
"""

import importlib

import pytest

from src.shared.registry import ALL_POSITIONS, INFERENCE_REGISTRY


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_head_hidden_overrides_match_position_config(pos):
    """If a position's config declares non-empty ``{POS}_NN_HEAD_HIDDEN_OVERRIDES``,
    the same dict must be plumbed into ``INFERENCE_REGISTRY[pos]["nn_kwargs"]``.

    An empty dict / missing attr means "no overrides" — registry may omit
    the kwarg (MultiHeadNet defaults ``head_hidden_overrides=None``).
    """
    config_module = importlib.import_module(f"src.{pos.lower()}.config")
    cfg_overrides = getattr(config_module, "NN_HEAD_HIDDEN_OVERRIDES", None)

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


@pytest.mark.unit
@pytest.mark.parametrize("pos", ALL_POSITIONS)
def test_attn_kwargs_backbone_layers_match_config(pos):
    """``INFERENCE_REGISTRY[pos]["attn_nn_kwargs_static"]["backbone_layers"]``
    must equal the position config's ``NN_BACKBONE_LAYERS``.

    Regression guard for the post-#154 silent-rename bug: ``_attn_kwargs_static``
    looked up ``f"{POS}_NN_BACKBONE_LAYERS"`` but the rename dropped the
    per-position prefix from every config attribute. Every getattr fell
    through to its default (``[]``), and inference built every attention
    model with no backbone — ``backbone_layers[-1]`` raised
    ``IndexError: list index out of range`` and ``attn_nn_pred`` came back
    NaN for QB/RB/WR/TE/DST in production. K was unaffected only because
    K's kwargs are constructed from direct imports, not via this helper.

    Asserting on backbone_layers specifically is the cheapest invariant
    that catches the whole class — every other attribute looked up by
    this helper had a benign default that wouldn't have surfaced the
    miswired prefix on its own.
    """
    config_module = importlib.import_module(f"src.{pos.lower()}.config")
    cfg_layers = list(getattr(config_module, "NN_BACKBONE_LAYERS", []))
    assert cfg_layers, (
        f"{pos}: config defines no NN_BACKBONE_LAYERS — every position with "
        f"an attention NN needs one. Update src/{pos.lower()}/config.py."
    )

    reg_kwargs = INFERENCE_REGISTRY[pos]["attn_nn_kwargs_static"]
    reg_layers = list(reg_kwargs.get("backbone_layers", []))
    assert reg_layers == cfg_layers, (
        f"{pos}: config declares NN_BACKBONE_LAYERS={cfg_layers!r} but registry "
        f"passes backbone_layers={reg_layers!r}. The two paths build different "
        f"architectures, so MultiHeadNetWithHistory.load_state_dict will fail "
        f"on the trained checkpoint."
    )
