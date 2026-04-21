"""Generic multi-head neural network for fantasy point decomposition."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTDHead(nn.Module):
    """Two-stage hurdle head for zero-inflated TD prediction.

    Stage 1 (gate): P(TD > 0) via sigmoid
    Stage 2 (value): E[TD | TD > 0] via Softplus (always positive)
    Output: gate_prob * cond_value = E[TD]
    """

    def __init__(self, in_dim: int, gate_hidden: int = 16, value_hidden: int = 48):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(in_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(x).squeeze(-1)
        cond_value = self.value(x).squeeze(-1)
        td_pred = torch.sigmoid(gate_logit) * cond_value
        return td_pred, gate_logit


class MultiHeadNet(nn.Module):
    """Multi-head neural network for position-agnostic fantasy point decomposition.

    Architecture:
        Input (N features)
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x len(backbone_layers)
            -> One head per target: Linear -> ReLU -> Linear -> squeeze
            -> Softplus on non-negative targets (preserves gradient flow near zero)

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
        aggregate_fn=None,
    ):
        super().__init__()
        self.target_names = target_names
        # Which heads are clamped to >= 0. Default: all targets.
        # Override for targets that can be negative.
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        # None → sum per-target heads (legacy). Pass aggregate_fn_for(pos) to
        # weight raw-stat preds into fantasy points via shared.aggregate_targets.
        self.aggregate_fn = aggregate_fn

        # === Shared Backbone ===
        backbone_blocks = []
        prev_dim = input_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
                val = F.softplus(val)
            preds[name] = val
        if self.aggregate_fn is not None:
            preds["total"] = self.aggregate_fn(preds)
        else:
            preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}


class AttentionPool(nn.Module):
    """Learned-query attention pooling over a variable-length sequence.

    Uses ``n_targets × n_heads`` learned query vectors to attend over a padded
    sequence, producing one fixed-size vector per target regardless of sequence
    length. Masking ensures padded positions are ignored.

    Per-target queries let each downstream target pull the exact slice of
    history it cares about (e.g. td_points queries focus on goal-line usage
    while rushing_floor queries focus on carry volume), instead of being
    forced through a shared summary bottleneck.

    When ``n_targets == 1`` the output is squeezed to preserve the legacy
    [batch, n_heads * d_model] shape.

    Optional K/V projections separate "what to attend to" from "what to extract".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        n_targets: int = 1,
        project_kv: bool = False,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_targets, n_heads, d_model) * 0.02)
        self.scale = d_model**-0.5
        self.n_heads = n_heads
        self.n_targets = n_targets
        self.d_model = d_model
        self.project_kv = project_kv

        if project_kv:
            self.key_proj = nn.Linear(d_model, d_model, bias=False)
            self.value_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(self, keys: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            keys: [batch, seq_len, d_model] — encoded game history
            mask: [batch, seq_len] — True where real game, False where padding

        Returns:
            [batch, n_targets, n_heads * d_model] when n_targets > 1,
            [batch, n_heads * d_model] when n_targets == 1.
        """
        batch_size = keys.size(0)

        if self.project_kv:
            k = self.key_proj(keys)
            v = self.value_proj(keys)
        else:
            k = keys
            v = keys

        # Flatten queries to [n_targets * n_heads, d_model] for a single matmul.
        q_flat = self.queries.reshape(self.n_targets * self.n_heads, self.d_model)
        # attn scores: [batch, n_targets * n_heads, seq_len]
        attn = torch.einsum("qd,bsd->bqs", q_flat, k) * self.scale

        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, seq_len]
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        weights = F.softmax(attn, dim=-1)
        # Handle all-padding rows (softmax of all -inf = nan)
        weights = weights.nan_to_num(0.0)
        weights = self.attn_drop(weights)
        # pooled: [batch, n_targets * n_heads, d_model]
        pooled = torch.bmm(weights, v)
        pooled = pooled.reshape(batch_size, self.n_targets, self.n_heads * self.d_model)
        if self.n_targets == 1:
            return pooled.squeeze(1)
        return pooled


class MultiHeadNetWithHistory(nn.Module):
    """Multi-head network with a parallel attention branch over game history.

    Architecture:
        Static features [batch, static_dim]
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x N
            -> shared_static_rep [batch, backbone_out_dim]

        Game history [batch, seq_len, game_dim]
            -> GameEncoder: Linear(game_dim, d_model) -> ReLU
            -> (optional) Learned positional encoding
            -> AttentionPool with per-target queries
               -> [batch, n_targets, n_heads * d_model]

        For each target t:
            history_vec_t = LayerNorm_t(pool[:, t, :])
            head_input_t  = concat(shared_static_rep, history_vec_t)
            pred_t        = head_t(head_input_t)

    Per-target queries let each target pull its own slice of history instead
    of sharing a single pooled summary through the backbone. The shared
    backbone still produces a common static representation that every head
    consumes alongside its target-specific history.
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
        project_kv: bool = False,
        use_positional_encoding: bool = False,
        max_seq_len: int = 17,
        use_gated_fusion: bool = False,
        attn_dropout: float = 0.0,
        encoder_hidden_dim: int = 0,
        gated_td: bool = False,
        td_gate_hidden: int = 16,
        gated_td_target=None,  # legacy str; kept for backward compat
        gated_td_targets: list[str] | None = None,
        aggregate_fn=None,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self.gated_td = gated_td
        # Normalize gated_td target(s): accept either legacy str or list.
        if gated_td_targets is None:
            if gated_td_target is None:
                gated_td_targets = []
            elif isinstance(gated_td_target, str):
                gated_td_targets = [gated_td_target]
            else:
                gated_td_targets = list(gated_td_target)
        self.gated_td_targets = list(gated_td_targets)
        # None → sum per-target heads (legacy); see MultiHeadNet for full comment.
        self.aggregate_fn = aggregate_fn
        self.d_model = d_model
        self.n_targets = len(target_names)

        # === Game History Branch ===
        if encoder_hidden_dim > 0:
            self.game_encoder = nn.Sequential(
                nn.Linear(game_dim, encoder_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(encoder_hidden_dim),
                nn.Linear(encoder_hidden_dim, d_model),
                nn.ReLU(),
            )
        else:
            self.game_encoder = nn.Sequential(
                nn.Linear(game_dim, d_model),
                nn.ReLU(),
            )

        # Positional encoding: gives the model temporal ordering signal
        # so it can distinguish recent games from older ones.
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Per-target queries: each target pulls its own slice of history
        # instead of routing through a single shared summary.
        self.attn_pool = AttentionPool(
            d_model,
            n_heads=n_attn_heads,
            n_targets=self.n_targets,
            project_kv=project_kv,
            attn_dropout=attn_dropout,
        )

        attn_out_dim = n_attn_heads * d_model
        # Per-target LayerNorms — each target's pooled stats live at different
        # scales (td_points ~3, rushing_floor ~8), so one norm per target.
        self.history_norms = nn.ModuleList([nn.LayerNorm(attn_out_dim) for _ in target_names])

        # Gated fusion: static features control how much to trust history.
        # Prevents noisy attention signal from degrading static feature quality.
        # Per-target gates since each head gets its own history vec.
        self.use_gated_fusion = use_gated_fusion
        if use_gated_fusion:
            self.fusion_gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(static_dim + attn_out_dim, attn_out_dim),
                        nn.Sigmoid(),
                    )
                    for _ in target_names
                ]
            )

        # === Shared Backbone (static features only) ===
        # History is routed per-target directly into the heads, so the
        # backbone no longer needs to see the attention output.
        backbone_blocks = []
        prev_dim = static_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]
        overrides = head_hidden_overrides or {}

        # === Output Heads (consume shared_static ⊕ per-target history) ===
        head_in_dim = backbone_out_dim + attn_out_dim
        self.heads = nn.ModuleDict()
        gated_set = set(self.gated_td_targets)
        for name in target_names:
            h = overrides.get(name, head_hidden)
            if gated_td and name in gated_set:
                self.heads[name] = GatedTDHead(
                    in_dim=head_in_dim,
                    gate_hidden=td_gate_hidden,
                    value_hidden=h,
                )
            else:
                self.heads[name] = nn.Sequential(
                    nn.Linear(head_in_dim, h),
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
        # Encode game history
        encoded = self.game_encoder(x_history)  # [batch, seq_len, d_model]

        # Add positional encoding so attention can weight by recency
        if self.use_positional_encoding:
            seq_len = encoded.size(1)
            positions = torch.arange(seq_len, device=encoded.device)
            encoded = encoded + self.pos_embedding(positions)

        # Per-target attention pool: [batch, n_targets, n_heads * d_model]
        history_per_target = self.attn_pool(encoded, history_mask)
        if history_per_target.dim() == 2:
            # Back-compat guard: single-target AttentionPool squeezes. Add axis.
            history_per_target = history_per_target.unsqueeze(1)

        # Shared backbone processes static features only.
        shared_static = self.backbone(x_static)

        preds = {}
        for i, name in enumerate(self.target_names):
            history_vec = self.history_norms[i](history_per_target[:, i, :])
            if self.use_gated_fusion:
                gate = self.fusion_gates[i](torch.cat([x_static, history_vec], dim=-1))
                history_vec = gate * history_vec

            head_input = torch.cat([shared_static, history_vec], dim=-1)
            head = self.heads[name]
            if isinstance(head, GatedTDHead):
                td_pred, gate_logit = head(head_input)
                preds[name] = td_pred
                preds[f"{name}_gate_logit"] = gate_logit
            else:
                val = head(head_input).squeeze(-1)
                if name in self.non_negative_targets:
                    val = F.softplus(val)
                preds[name] = val
        if self.aggregate_fn is not None:
            preds["total"] = self.aggregate_fn(preds)
        else:
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


class MultiHeadNetWithNestedHistory(nn.Module):
    """Multi-head network with a two-level attention branch.

    Inner level: a single-query AttentionPool collapses each game's
    variable-length kick sequence into a fixed d_kick vector. Outer level:
    per-target AttentionPool weighs games across the season, mirroring
    MultiHeadNetWithHistory. Static branch and head wiring match the flat
    version; no gating (K has no zero-inflated count targets).

    Shapes (see the forward docstring for details):
        x_static:   [B, static_dim]
        x_kicks:    [B, G, K, kick_dim]
        outer_mask: [B, G]
        inner_mask: [B, G, K]
    """

    def __init__(
        self,
        static_dim: int,
        kick_dim: int,
        target_names: list[str],
        backbone_layers: list[int],
        d_kick: int = 16,
        d_model: int = 32,
        n_attn_heads: int = 2,
        head_hidden: int = 32,
        dropout: float = 0.3,
        head_hidden_overrides: dict = None,
        non_negative_targets: set = None,
        project_kv: bool = False,
        use_positional_encoding: bool = False,
        max_games: int = 17,
        attn_dropout: float = 0.0,
        encoder_hidden_dim: int = 0,
        aggregate_fn=None,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self.aggregate_fn = aggregate_fn
        self.d_model = d_model
        self.d_kick = d_kick
        self.n_targets = len(target_names)

        # === Inner: per-kick encoder + single-query pool ===
        self.kick_encoder = nn.Sequential(
            nn.Linear(kick_dim, d_kick),
            nn.ReLU(),
        )
        # Single-query, single-head pool collapses [K, d_kick] → [d_kick].
        self.inner_pool = AttentionPool(
            d_kick,
            n_heads=1,
            n_targets=1,
            project_kv=False,
            attn_dropout=attn_dropout,
        )

        # === Outer: game encoder + per-target attention pool ===
        if encoder_hidden_dim > 0:
            self.game_encoder = nn.Sequential(
                nn.Linear(d_kick, encoder_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(encoder_hidden_dim),
                nn.Linear(encoder_hidden_dim, d_model),
                nn.ReLU(),
            )
        else:
            self.game_encoder = nn.Sequential(
                nn.Linear(d_kick, d_model),
                nn.ReLU(),
            )

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_games, d_model)

        self.attn_pool = AttentionPool(
            d_model,
            n_heads=n_attn_heads,
            n_targets=self.n_targets,
            project_kv=project_kv,
            attn_dropout=attn_dropout,
        )

        attn_out_dim = n_attn_heads * d_model
        self.history_norms = nn.ModuleList([nn.LayerNorm(attn_out_dim) for _ in target_names])

        # === Shared backbone (static only) ===
        backbone_blocks = []
        prev_dim = static_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]
        overrides = head_hidden_overrides or {}
        head_in_dim = backbone_out_dim + attn_out_dim
        self.heads = nn.ModuleDict()
        for name in target_names:
            h = overrides.get(name, head_hidden)
            self.heads[name] = nn.Sequential(
                nn.Linear(head_in_dim, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

    def forward(
        self,
        x_static: torch.Tensor,
        x_kicks: torch.Tensor,
        outer_mask: torch.Tensor,
        inner_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            x_static:   [B, static_dim]
            x_kicks:    [B, G, K, kick_dim] — zero-padded kick features
            outer_mask: [B, G]              — True where real game
            inner_mask: [B, G, K]           — True where real kick
        """
        B, G, K, _ = x_kicks.shape

        # Inner encode + pool: [B, G, d_kick]
        kick_enc = self.kick_encoder(x_kicks)  # [B, G, K, d_kick]
        flat = kick_enc.reshape(B * G, K, self.d_kick)
        flat_mask = inner_mask.reshape(B * G, K)
        per_game = self.inner_pool(flat, flat_mask)  # [B*G, d_kick] (single-target squeeze)
        per_game = per_game.reshape(B, G, self.d_kick)

        # Outer encode + attend
        encoded = self.game_encoder(per_game)  # [B, G, d_model]
        if self.use_positional_encoding:
            positions = torch.arange(G, device=encoded.device)
            encoded = encoded + self.pos_embedding(positions)

        history_per_target = self.attn_pool(encoded, outer_mask)  # [B, n_targets, attn_out]
        if history_per_target.dim() == 2:
            history_per_target = history_per_target.unsqueeze(1)

        shared_static = self.backbone(x_static)

        preds = {}
        for i, name in enumerate(self.target_names):
            history_vec = self.history_norms[i](history_per_target[:, i, :])
            head_input = torch.cat([shared_static, history_vec], dim=-1)
            val = self.heads[name](head_input).squeeze(-1)
            if name in self.non_negative_targets:
                val = F.softplus(val)
            preds[name] = val
        if self.aggregate_fn is not None:
            preds["total"] = self.aggregate_fn(preds)
        else:
            preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def predict_numpy(
        self,
        X_static: np.ndarray,
        X_kicks: np.ndarray,
        outer_mask: np.ndarray,
        inner_mask: np.ndarray,
        device: torch.device,
    ) -> dict:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(X_static).to(device)
            k = torch.FloatTensor(X_kicks).to(device)
            om = torch.BoolTensor(outer_mask).to(device)
            im = torch.BoolTensor(inner_mask).to(device)
            preds = self.forward(s, k, om, im)
            return {key: v.cpu().numpy() for key, v in preds.items()}
