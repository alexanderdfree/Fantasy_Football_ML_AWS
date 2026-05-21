"""Sanity checks that K's src/shared/registry.py entry lines up with the
MultiHeadNetWithNestedHistory class it's meant to serve.

Regression guard for the "training-only ghost" bug: K's training pipeline
wrote k_attention_nn.pt to disk, but the registry was missing attn_nn_file
and the kwargs needed to rebuild the matching model at inference, so the
attention NN was never loaded in app.py. These tests fail if any of those
wiring points regress.
"""

import pytest
import torch

from src.shared.neural_net import MultiHeadNetWithNestedHistory
from src.shared.registry import INFERENCE_REGISTRY


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
