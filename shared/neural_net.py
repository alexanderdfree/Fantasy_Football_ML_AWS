"""Generic multi-head neural network for fantasy point decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadNet(nn.Module):
    """Multi-head neural network for position-agnostic fantasy point decomposition.

    Architecture:
        Input (N features)
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x len(backbone_layers)
            -> One head per target: Linear -> ReLU -> Linear -> squeeze

    Total prediction = sum of all heads.
    """

    def __init__(
        self,
        input_dim: int,
        target_names: list[str],
        backbone_layers: list[int],
        head_hidden: int = 32,
        dropout: float = 0.3,
        head_hidden_overrides: dict = None,
        non_negative_targets: set = None,
    ):
        super().__init__()
        self.target_names = target_names
        # Which heads are clamped to >= 0. Default: all targets.
        # Override for targets that can be negative (e.g. DST pts_allowed_bonus).
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )

        # === Shared Backbone ===
        backbone_blocks = []
        prev_dim = input_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]
        overrides = head_hidden_overrides or {}

        # === Output Heads (one per target) ===
        self.heads = nn.ModuleDict()
        for name in target_names:
            h = overrides.get(name, head_hidden)
            self.heads[name] = nn.Sequential(
                nn.Linear(backbone_out_dim, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.backbone(x)
        preds = {}
        for name, head in self.heads.items():
            val = head(shared).squeeze(-1)
            if name in self.non_negative_targets:
                val = torch.clamp(val, min=0.0)
            preds[name] = val
        preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict with backward compatibility for old per-position models.

        Old models used attribute names like 'passing_head', 'rushing_head', 'td_head'.
        New models use ModuleDict: 'heads.passing_floor', 'heads.rushing_floor', etc.
        """
        # Check if state_dict uses old naming convention
        # Old models had attributes like 'passing_head', 'rushing_head', 'td_head'
        # New models use ModuleDict: 'heads.passing_floor', 'heads.td_points', etc.
        old_keys = [k for k in state_dict if "_head." in k and not k.startswith("heads.")]
        if old_keys:
            # Extract unique head prefixes (e.g. "rushing_head" from "rushing_head.0.weight")
            old_prefixes = set(k.split(".")[0] for k in old_keys if k.endswith("_head") or "." in k)
            old_prefixes = {p for p in old_prefixes if p.endswith("_head")}

            prefix_to_target = {}
            for prefix in old_prefixes:
                head_word = prefix[: -len("_head")]  # "rushing_head" -> "rushing"
                for target in self.target_names:
                    if target.startswith(head_word):
                        prefix_to_target[prefix] = target
                        break

            new_state_dict = {}
            for key, value in state_dict.items():
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0] in prefix_to_target:
                    new_key = f"heads.{prefix_to_target[parts[0]]}.{parts[1]}"
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}


class AttentionPool(nn.Module):
    """Learned-query attention pooling over a variable-length sequence.

    Uses n_heads learned query vectors to attend over a padded sequence,
    producing a fixed-size output regardless of sequence length.
    Masking ensures padded positions are ignored.
    """

    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_heads, d_model) * 0.02)
        self.scale = d_model ** -0.5
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, keys: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            keys: [batch, seq_len, d_model] — encoded game history
            mask: [batch, seq_len] — True where real game, False where padding

        Returns:
            [batch, n_heads * d_model] — pooled history representation
        """
        batch_size = keys.size(0)
        # queries: [n_heads, d_model] -> [batch, n_heads, d_model]
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        # attn scores: [batch, n_heads, seq_len]
        attn = torch.bmm(q, keys.transpose(1, 2)) * self.scale

        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, seq_len]
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))

        weights = F.softmax(attn, dim=-1)
        # Handle all-padding rows (softmax of all -inf = nan)
        weights = weights.nan_to_num(0.0)
        # pooled: [batch, n_heads, d_model]
        pooled = torch.bmm(weights, keys)
        return pooled.reshape(batch_size, -1)  # [batch, n_heads * d_model]


class MultiHeadNetWithHistory(nn.Module):
    """Multi-head network with a parallel attention branch over game history.

    Architecture:
        Static features [batch, static_dim]
            -> (passed through directly)

        Game history [batch, seq_len, game_dim]
            -> GameEncoder: Linear(game_dim, d_model)
            -> AttentionPool(d_model, n_heads) -> [batch, n_heads * d_model]

        Concatenated [batch, static_dim + n_heads * d_model]
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x N
            -> Output heads (one per target)
    """

    def __init__(
        self,
        static_dim: int,
        game_dim: int,
        target_names: list[str],
        backbone_layers: list[int],
        d_model: int = 32,
        n_attn_heads: int = 2,
        head_hidden: int = 32,
        dropout: float = 0.3,
        head_hidden_overrides: dict = None,
        non_negative_targets: set = None,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self.d_model = d_model

        # === Game History Branch ===
        self.game_encoder = nn.Sequential(
            nn.Linear(game_dim, d_model),
            nn.ReLU(),
        )
        self.attn_pool = AttentionPool(d_model, n_heads=n_attn_heads)
        self.history_norm = nn.LayerNorm(n_attn_heads * d_model)

        # === Shared Backbone (static + history concatenated) ===
        combined_dim = static_dim + n_attn_heads * d_model
        backbone_blocks = []
        prev_dim = combined_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]
        overrides = head_hidden_overrides or {}

        # === Output Heads ===
        self.heads = nn.ModuleDict()
        for name in target_names:
            h = overrides.get(name, head_hidden)
            self.heads[name] = nn.Sequential(
                nn.Linear(backbone_out_dim, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

    def forward(
        self,
        x_static: torch.Tensor,
        x_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            x_static: [batch, static_dim] — standard feature vector
            x_history: [batch, seq_len, game_dim] — padded game history
            history_mask: [batch, seq_len] — True for real games
        """
        # Encode and pool game history
        encoded = self.game_encoder(x_history)       # [batch, seq_len, d_model]
        history_vec = self.attn_pool(encoded, history_mask)  # [batch, n_heads * d_model]
        history_vec = self.history_norm(history_vec)

        # Concatenate with static features
        combined = torch.cat([x_static, history_vec], dim=-1)

        # Backbone + heads
        shared = self.backbone(combined)
        preds = {}
        for name, head in self.heads.items():
            val = head(shared).squeeze(-1)
            if name in self.non_negative_targets:
                val = torch.clamp(val, min=0.0)
            preds[name] = val
        preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def predict_numpy(
        self,
        X_static: np.ndarray,
        X_history: np.ndarray,
        history_mask: np.ndarray,
        device: torch.device,
    ) -> dict:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(X_static).to(device)
            h = torch.FloatTensor(X_history).to(device)
            m = torch.BoolTensor(history_mask).to(device)
            preds = self.forward(s, h, m)
            return {k: v.cpu().numpy() for k, v in preds.items()}
