"""Sanity checks that K's shared/registry.py entry lines up with the
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
        assert static_dim > 0, "K_ATTN_STATIC_FEATURES must not be empty"
        assert kick_dim > 0, "K_ATTN_KICK_STATS must not be empty"

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
        """Registry kwargs must mirror the K_ATTN_* values used at training."""
        import src.K.k_config as k_cfg

        kw = reg["attn_nn_kwargs_static"]
        assert kw["d_kick"] == k_cfg.K_ATTN_KICK_DIM
        assert kw["d_model"] == k_cfg.K_ATTN_D_MODEL
        assert kw["n_attn_heads"] == k_cfg.K_ATTN_N_HEADS
        assert kw["encoder_hidden_dim"] == k_cfg.K_ATTN_ENCODER_HIDDEN_DIM
        assert kw["max_games"] == k_cfg.K_ATTN_MAX_GAMES
        assert reg["attn_max_games"] == k_cfg.K_ATTN_MAX_GAMES
        assert reg["attn_max_kicks_per_game"] == k_cfg.K_ATTN_MAX_KICKS_PER_GAME
        assert reg["attn_kick_stats"] == list(k_cfg.K_ATTN_KICK_STATS)
        assert reg["attn_static_features"] == list(k_cfg.K_ATTN_STATIC_FEATURES)

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
        preds = model.predict_numpy(X, hist, outer, inner, torch.device("cpu"))
        for t in targets:
            assert t in preds
            assert preds[t].shape == (B,)
