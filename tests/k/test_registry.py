"""Sanity checks that K's src/shared/registry.py entry lines up with the
MultiHeadNetWithNestedHistory class it's meant to serve.

Regression guard for the "training-only ghost" bug: K's training pipeline
wrote k_attention_nn.pt to disk, but the registry was missing attn_nn_file
and the kwargs needed to rebuild the matching model at inference, so the
attention NN was never loaded in app.py. These tests fail if any of those
wiring points regress.
"""

from dataclasses import replace

import pytest
import torch

from src.k.config import POSITION_CONFIG
from src.shared.neural_net import (
    MultiHeadNetWithNestedHistory,
    build_multihead_net_with_nested_history,
)
from src.shared.position_pipeline import build_pipeline_config
from src.shared.registry import INFERENCE_REGISTRY, _nested_attn_kwargs_static


@pytest.mark.unit
class TestKAttentionRegistryWiring:
    @pytest.fixture
    def reg(self):
        return INFERENCE_REGISTRY["K"]

    def test_attention_wiring_keys_present(self, reg):
        """The flag + file + nested-history kwargs the inference branch needs."""
        required = {
            "train_attention_nn",
            "attn_nn_file",
            "attn_history_structure",
            "attn_static_from_df",
            "attn_static_features",
            "attn_kick_stats",
            "attn_max_games",
            "attn_max_kicks_per_game",
            "attn_nn_kwargs_static",
        }
        missing = required - reg.keys()
        assert not missing, f"K registry missing attention keys: {missing}"

    def test_attention_enabled_and_nested(self, reg):
        """K must declare itself as a nested-history attention consumer."""
        assert reg["train_attention_nn"] is True
        assert reg["attn_history_structure"] == "nested"
        assert reg["attn_static_from_df"] is True
        assert reg["attn_nn_file"] == "k_attention_nn.pt"

    def test_kwargs_build_model_with_matching_state_dict(self, reg):
        """Construct a MultiHeadNetWithNestedHistory with the registry kwargs
        + realistic runtime dims, then round-trip its state_dict through
        strict load. Fails if the kwargs set has drifted away from the model
        signature (e.g. a renamed/removed kwarg in the network class).
        """
        static_dim = len(reg["attn_static_features"])
        kick_dim = len(reg["attn_kick_stats"])
        targets = reg["targets"]
        assert static_dim > 0, "ATTN_STATIC_FEATURES must not be empty"
        assert kick_dim > 0, "ATTN_KICK_STATS must not be empty"

        model = MultiHeadNetWithNestedHistory(
            static_dim=static_dim,
            kick_dim=kick_dim,
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        )
        # Round-trip: save → load strict. Confirms kwargs fully determine shape.
        state = model.state_dict()
        fresh = MultiHeadNetWithNestedHistory(
            static_dim=static_dim,
            kick_dim=kick_dim,
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        )
        fresh.load_state_dict(state, strict=True)

    def test_kwargs_align_with_training_config(self, reg):
        """Registry kwargs must mirror the POSITION_CONFIG values used at training."""
        from src.k.config import POSITION_CONFIG as pc

        kw = reg["attn_nn_kwargs_static"]
        assert kw["d_kick"] == pc.attn_kick_dim
        assert kw["d_model"] == pc.attn_d_model
        assert kw["n_attn_heads"] == pc.attn_n_heads
        assert kw["encoder_hidden_dim"] == pc.attn_encoder_hidden_dim
        assert kw["max_games"] == pc.attn_max_games
        assert reg["attn_max_games"] == pc.attn_max_games
        assert reg["attn_max_kicks_per_game"] == pc.attn_max_kicks_per_game
        assert reg["attn_kick_stats"] == list(pc.attn_kick_stats)
        assert reg["attn_static_features"] == list(pc.attn_static_features)

    def test_predict_numpy_end_to_end(self, reg):
        """Tiny forward pass through predict_numpy — catches signature drift
        between the registry-built model and the inference call in app.py."""
        import numpy as np

        static_dim = len(reg["attn_static_features"])
        kick_dim = len(reg["attn_kick_stats"])
        targets = reg["targets"]
        model = MultiHeadNetWithNestedHistory(
            static_dim=static_dim,
            kick_dim=kick_dim,
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        )
        model.eval()
        B, G, K = 2, reg["attn_max_games"], reg["attn_max_kicks_per_game"]
        X = np.zeros((B, static_dim), dtype=np.float32)
        hist = np.zeros((B, G, K, kick_dim), dtype=np.float32)
        outer = np.ones((B, G), dtype=bool)
        inner = np.ones((B, G, K), dtype=bool)
        # When the registry opts into per-game aggregates, predict_numpy
        # expects a matching [B, G, game_dim] tensor; otherwise the kwarg stays
        # None and the model behaves as the legacy nested-only K did.
        game_history_stats = reg.get("attn_history_stats") or []
        game_hist = None
        if game_history_stats:
            game_hist = np.zeros((B, G, len(game_history_stats)), dtype=np.float32)
        preds = model.predict_numpy(
            X, hist, outer, inner, torch.device("cpu"), X_game_history=game_hist
        )
        for t in targets:
            assert t in preds
            assert preds[t].shape == (B,)


@pytest.mark.unit
class TestKHeadHiddenOverridesForwarded:
    """Regression guard for #1503: ``_nested_attn_kwargs_static`` must forward
    ``nn_head_hidden_overrides`` exactly like the training factory does
    (``build_multihead_net_with_nested_history`` reads
    ``cfg["nn_head_hidden_overrides"]``). Before the fix a K per-head override
    would have trained one head shape and served another — a state_dict shape
    mismatch at load time, the 2026-06-15 architecture-staleness class.
    """

    # Deliberately != K's production ``nn_head_hidden`` so the override really
    # changes the head shape (asserted in the fixture) — a same-width override
    # would make the shape-parity checks below vacuous.
    _OVERRIDE = {"fg_misses": 40}

    @pytest.fixture
    def pc_override(self):
        pc = replace(POSITION_CONFIG, nn_head_hidden_overrides=dict(self._OVERRIDE))
        assert set(self._OVERRIDE) <= set(pc.targets)
        assert all(width != pc.nn_head_hidden for width in self._OVERRIDE.values())
        return pc

    def test_unmodified_k_config_emits_no_override_key(self):
        """K configures no override today -> today's served kwargs are unchanged."""
        assert not POSITION_CONFIG.nn_head_hidden_overrides
        assert "head_hidden_overrides" not in _nested_attn_kwargs_static(POSITION_CONFIG)

    def test_override_is_forwarded_to_served_kwargs(self, pc_override):
        kw = _nested_attn_kwargs_static(pc_override)
        assert kw["head_hidden_overrides"] == self._OVERRIDE

    def test_trained_and_served_shapes_match_under_override(self, pc_override):
        """Shape parity between the two rebuild paths: the training factory
        (cfg-dict path, via ``build_pipeline_config`` exactly as K's
        ``run_pipeline`` does) and the served kwargs (registry path) must
        produce state_dicts with identical keys + shapes that strict-load."""
        cfg = build_pipeline_config("K", pc_override)
        static_dim = len(pc_override.attn_static_features)
        kick_dim = len(pc_override.attn_kick_stats)
        targets = list(pc_override.targets)

        trained = build_multihead_net_with_nested_history(
            cfg,
            static_dim=static_dim,
            kick_dim=kick_dim,
            max_games=pc_override.attn_max_games,
            targets=targets,
            game_dim=len(pc_override.attn_history_stats),
        )
        served = MultiHeadNetWithNestedHistory(
            static_dim=static_dim,
            kick_dim=kick_dim,
            target_names=targets,
            **_nested_attn_kwargs_static(pc_override),
        )

        def _shapes(model):
            return {k: tuple(v.shape) for k, v in model.state_dict().items()}

        assert _shapes(served) == _shapes(trained)
        served.load_state_dict(trained.state_dict(), strict=True)

        # Positive control — the override genuinely changes the head shape:
        # a model rebuilt from the UN-overridden served kwargs (what the
        # pre-fix builder emitted for this config) cannot load the checkpoint.
        stale = MultiHeadNetWithNestedHistory(
            static_dim=static_dim,
            kick_dim=kick_dim,
            target_names=targets,
            **_nested_attn_kwargs_static(POSITION_CONFIG),
        )
        with pytest.raises(RuntimeError, match="size mismatch"):
            stale.load_state_dict(trained.state_dict(), strict=True)
