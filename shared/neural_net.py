"""Generic multi-head neural network for fantasy point decomposition."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_non_negative(val: torch.Tensor, name: str, non_negative: set) -> torch.Tensor:
    """Clamp ``val`` to ``>= 0`` when ``name`` is in ``non_negative``.

    Must use ``torch.clamp(min=0.0)``: softplus introduces a ~0.693 floor per
    head that compounds across targets. See TODO.md archive entry
    "Softplus floor inflated low-scoring predictions".
    """
    if name in non_negative:
        return torch.clamp(val, min=0.0)
    return val


def _build_backbone(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Sequential:
    """Shared ``Linear → BN → ReLU → Dropout`` backbone used by every
    ``MultiHeadNet*`` variant.

    Kept as an ``nn.Sequential`` (not a custom module) so the state-dict
    keys produced by each caller remain ``self.backbone.0.weight`` etc. —
    existing checkpoints continue to load without migration.
    """
    blocks: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        blocks.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim
    return nn.Sequential(*blocks)


class GatedHead(nn.Module):
    """Two-stage hurdle head for zero-inflated count prediction.

    Stage 1 (gate): P(Y > 0) via sigmoid over ``gate_logit``.
    Stage 2 (value): the rate ``mu = E[Y | Y > 0]`` via Softplus on one trunk
    output, plus a per-sample ``log_alpha`` on the other. ``log_alpha`` is the
    NegBin-2 dispersion (``var = mu + exp(log_alpha) * mu^2``); it's unused by
    Poisson-hurdle losses but exposed on every GatedHead so the loss layer can
    choose its family without widening the module API.

    Forward returns ``(expected, gate_logit, mu, log_alpha)``:
        expected    = sigmoid(gate_logit) * mu   — E[Y] for reporting/metrics
        gate_logit  = pre-sigmoid logit          — BCE target
        mu          = E[Y | Y > 0], softplus + 1e-6 floor so log(mu) is finite
        log_alpha   = per-sample NegBin-2 log-dispersion, real-valued

    Value and dispersion share a single trunk so the extra capacity for
    dispersion is small (~49 params at value_hidden=48). At inference the 4-tuple
    is stored in the prediction dict; Poisson-family losses ignore log_alpha.
    """

    def __init__(self, in_dim: int, gate_hidden: int = 16, value_hidden: int = 48):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(in_dim, value_hidden),
            nn.ReLU(),
        )
        self.value_mu = nn.Sequential(
            nn.Linear(value_hidden, 1),
            nn.Softplus(),
        )
        self.value_log_alpha = nn.Linear(value_hidden, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(x).squeeze(-1)
        trunk = self.value_trunk(x)
        mu = self.value_mu(trunk).squeeze(-1) + 1e-6
        log_alpha = self.value_log_alpha(trunk).squeeze(-1)
        expected = torch.sigmoid(gate_logit) * mu
        return expected, gate_logit, mu, log_alpha


# Back-compat alias — older callers still import this name.
GatedTDHead = GatedHead


class MultiHeadNet(nn.Module):
    """Multi-head neural network for position-agnostic raw-stat prediction.

    Architecture:
        Input (N features)
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x len(backbone_layers)
            -> One head per target: Linear -> ReLU -> Linear -> squeeze
            -> Clamp >= 0 on non-negative targets
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
        # Override for targets that can be negative.
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )

        # === Shared Backbone ===
        self.backbone = _build_backbone(input_dim, backbone_layers, dropout)

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
            preds[name] = apply_non_negative(val, name, self.non_negative_targets)
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
    history it cares about (e.g. rushing_tds queries focus on goal-line usage
    while rushing_yards queries focus on carry volume), instead of being
    forced through a shared summary bottleneck.

    When ``n_targets == 1`` the output is squeezed to preserve the legacy
    [batch, n_heads * d_model] shape.

    Optional K/V projections separate "what to attend to" from "what to extract".

    Optional ``learn_temperature`` adds a per-target learned softmax temperature
    ``T_t = exp(log_temperature_t)`` so each target can sharpen (T<1) or soften
    (T>1) its attention distribution independently. Initialised to 0 so
    ``exp(0)=1`` → behaviour identical to the baseline at step zero.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        n_targets: int = 1,
        project_kv: bool = False,
        attn_dropout: float = 0.0,
        learn_temperature: bool = False,
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

        # Per-target learned temperature. Stored as log so T = exp(log_T) is
        # always positive; init to 0 (T=1) preserves baseline scores exactly
        # at the start of training.
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.zeros(n_targets))

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

        if self.learn_temperature:
            # Per-target inverse-temperature, replicated across heads so one
            # scalar controls all heads of the same target. Shape broadcasts
            # over [batch, ..., seq_len].
            inv_t = torch.exp(-self.log_temperature)  # [n_targets]
            inv_t = inv_t.repeat_interleave(self.n_heads).view(1, -1, 1)
            attn = attn * inv_t

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
        gated: bool = False,
        gate_hidden: int = 16,
        gated_targets: list[str] | None = None,
        learn_attn_temperature: bool = False,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self.gated = gated
        self.gated_targets = list(gated_targets) if gated_targets else []
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
            learn_temperature=learn_attn_temperature,
        )

        attn_out_dim = n_attn_heads * d_model
        # Per-target LayerNorms — each target's pooled stats live at different
        # scales (rushing_tds ~1, rushing_yards ~80), so one norm per target.
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
        self.backbone = _build_backbone(static_dim, backbone_layers, dropout)

        backbone_out_dim = backbone_layers[-1]
        overrides = head_hidden_overrides or {}

        # === Output Heads (consume shared_static ⊕ per-target history) ===
        head_in_dim = backbone_out_dim + attn_out_dim
        self.heads = nn.ModuleDict()
        gated_set = set(self.gated_targets)
        for name in target_names:
            h = overrides.get(name, head_hidden)
            if gated and name in gated_set:
                self.heads[name] = GatedHead(
                    in_dim=head_in_dim,
                    gate_hidden=gate_hidden,
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
            if isinstance(head, GatedHead):
                pred, gate_logit, mu, log_alpha = head(head_input)
                preds[name] = pred
                preds[f"{name}_gate_logit"] = gate_logit
                preds[f"{name}_value_mu"] = mu
                preds[f"{name}_value_log_alpha"] = log_alpha
            else:
                val = head(head_input).squeeze(-1)
                preds[name] = apply_non_negative(val, name, self.non_negative_targets)
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
        learn_attn_temperature: bool = False,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
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
            learn_temperature=learn_attn_temperature,
        )

        attn_out_dim = n_attn_heads * d_model
        self.history_norms = nn.ModuleList([nn.LayerNorm(attn_out_dim) for _ in target_names])

        # === Shared backbone (static only) ===
        self.backbone = _build_backbone(static_dim, backbone_layers, dropout)

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
            preds[name] = apply_non_negative(val, name, self.non_negative_targets)
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


# ---------------------------------------------------------------------------
# Factories
#
# Every ``MultiHeadNet*`` has several optional kwargs whose defaults silently
# change behavior (e.g. ``non_negative_targets=None`` clamps every head).
# Training paths like ``_train_nn`` / ``_train_attention_nn`` / the CV loop
# used to construct models directly with inline kwarg lists, which let one
# site drift from the others — see TODO.md archive entry "``run_cv_pipeline``
# missing ``non_negative_targets`` on MultiHeadNet". Routing every construction
# through a single factory per variant makes that drift architecturally
# impossible: there is one place that maps ``cfg`` → model kwargs.
# ---------------------------------------------------------------------------


def build_multihead_net(
    cfg: dict,
    *,
    input_dim: int,
    targets: list[str],
) -> "MultiHeadNet":
    """Construct a MultiHeadNet from a training ``cfg`` dict."""
    return MultiHeadNet(
        input_dim=input_dim,
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )


def build_multihead_net_with_history(
    cfg: dict,
    *,
    static_dim: int,
    game_dim: int,
    targets: list[str],
) -> "MultiHeadNetWithHistory":
    """Construct a MultiHeadNetWithHistory from a training ``cfg`` dict."""
    return MultiHeadNetWithHistory(
        static_dim=static_dim,
        game_dim=game_dim,
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        d_model=cfg.get("attn_d_model", 32),
        n_attn_heads=cfg.get("attn_n_heads", 2),
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
        project_kv=cfg.get("attn_project_kv", False),
        use_positional_encoding=cfg.get("attn_positional_encoding", False),
        max_seq_len=cfg.get("attn_max_seq_len", 17),
        use_gated_fusion=cfg.get("attn_gated_fusion", False),
        attn_dropout=cfg.get("attn_dropout", 0.0),
        encoder_hidden_dim=cfg.get("attn_encoder_hidden_dim", 0),
        gated=cfg.get("attn_gated", False),
        gate_hidden=cfg.get("attn_gate_hidden", 16),
        gated_targets=cfg.get("gated_targets"),
        learn_attn_temperature=cfg.get("attn_learn_temperature", False),
    )


def build_multihead_net_with_nested_history(
    cfg: dict,
    *,
    static_dim: int,
    kick_dim: int,
    max_games: int,
    targets: list[str],
) -> "MultiHeadNetWithNestedHistory":
    """Construct a MultiHeadNetWithNestedHistory from a training ``cfg`` dict.

    ``max_games`` is derived from the padded history tensor shape at the call
    site, not from ``cfg``, since it can differ between CV folds and holdout.
    """
    return MultiHeadNetWithNestedHistory(
        static_dim=static_dim,
        kick_dim=kick_dim,
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        d_kick=cfg.get("attn_kick_dim", 16),
        d_model=cfg.get("attn_d_model", 32),
        n_attn_heads=cfg.get("attn_n_heads", 2),
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
        head_hidden_overrides=cfg.get("nn_head_hidden_overrides"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
        project_kv=cfg.get("attn_project_kv", False),
        use_positional_encoding=cfg.get("attn_positional_encoding", False),
        max_games=max_games,
        attn_dropout=cfg.get("attn_dropout", 0.0),
        encoder_hidden_dim=cfg.get("attn_encoder_hidden_dim", 0),
        learn_attn_temperature=cfg.get("attn_learn_temperature", False),
    )
