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


def _apply_history_dropout(mask: torch.Tensor, rate: float, training: bool) -> torch.Tensor:
    """Randomly drop real-game positions from a history mask during training.

    Each ``True`` entry in ``mask`` is independently flipped to ``False`` with
    probability ``rate``. Padding positions (already ``False``) are never
    re-enabled. Rows where every real game would be dropped fall back to the
    original mask so the attention branch always sees at least one real game
    and the head receives a non-zero signal.

    No-op when ``rate <= 0`` or the model is in eval mode.
    """
    if not training or rate <= 0.0:
        return mask
    drop = torch.rand_like(mask, dtype=torch.float) < rate
    dropped = mask & ~drop
    # Rows that had any real games but lost them all — restore originals.
    had_real = mask.any(dim=-1, keepdim=True)
    keeps_some = dropped.any(dim=-1, keepdim=True)
    restore = had_real & ~keeps_some
    return torch.where(restore, mask, dropped)


class SwiGLU(nn.Module):
    """Single SwiGLU projection: ``silu(W_g(x)) * W_v(x)``.

    Maps ``d_in -> d_out`` with two parallel linear projections (gate + value),
    a SiLU (Swish) nonlinearity on the gate, and an element-wise product.
    Used as a drop-in replacement for ``Linear(d_in, d_out) -> ReLU`` in the
    game encoder when the SwiGLU knob is opted in.

    Parameter count is ~2× the vanilla Linear it replaces; in exchange SwiGLU
    gives smoother gradients near zero than ReLU and a learned multiplicative
    gate per output channel.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_out)
        self.value_proj = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.value_proj(x)


def _build_game_encoder(
    in_dim: int, d_model: int, encoder_hidden_dim: int, use_swiglu: bool
) -> nn.Sequential:
    """Construct the per-game encoder that feeds the outer AttentionPool.

    Four shapes, selected by (``encoder_hidden_dim``, ``use_swiglu``):
      * ``0, False``    — ``Linear -> ReLU``                  (legacy 1-layer)
      * ``0, True``     — ``SwiGLU``                          (1-layer SwiGLU)
      * ``>0, False``   — ``Linear -> ReLU -> LN -> Linear -> ReLU``
                                                              (legacy 2-layer)
      * ``>0, True``    — ``SwiGLU -> LN -> SwiGLU``          (2-layer SwiGLU)

    Shared by ``MultiHeadNetWithHistory`` and
    ``MultiHeadNetWithNestedHistory`` (where the inner pool has already
    collapsed kicks to a per-game vector, so ``in_dim == d_kick``).
    """
    if encoder_hidden_dim > 0:
        if use_swiglu:
            return nn.Sequential(
                SwiGLU(in_dim, encoder_hidden_dim),
                nn.LayerNorm(encoder_hidden_dim),
                SwiGLU(encoder_hidden_dim, d_model),
            )
        return nn.Sequential(
            nn.Linear(in_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(encoder_hidden_dim),
            nn.Linear(encoder_hidden_dim, d_model),
            nn.ReLU(),
        )
    if use_swiglu:
        return nn.Sequential(SwiGLU(in_dim, d_model))
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.ReLU(),
    )


class SelfAttentionBlock(nn.Module):
    """Pre-LN transformer encoder block for contextualising the game sequence.

    Each block does::

        x = x + Dropout(MultiHeadSelfAttention(LN(x)))
        x = x + Dropout(FFN(LN(x)))

    Pre-LN scheme (Xiong et al. 2020) is empirically more stable than
    post-LN with AdamW and small batches — which matches this project's
    training setup. The block preserves sequence shape ``[B, S, d_model]``
    so a stack of blocks sits transparently between the per-game encoder
    and the learned-query AttentionPool.

    ``mask`` follows the project convention: ``True`` at real positions,
    ``False`` at padding. The block inverts it to the
    ``key_padding_mask`` convention ``nn.MultiheadAttention`` expects.
    All-padding rows produce NaN attention outputs by construction; the
    ``nan_to_num`` guard after the MHA call replaces them with zeros so
    the residual path is well-defined.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        # All-padding rows ⇒ MHA divides by zero along seq dim ⇒ NaN. Zero
        # them so the residual path is well-defined and downstream pools can
        # still mask those rows out properly.
        attn_out = attn_out.nan_to_num(0.0)
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


def _build_self_attention_stack(
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.ModuleList | None:
    """Build ``n_layers`` transformer encoder blocks (Pre-LN).

    Returns ``None`` when ``n_layers == 0`` so callers can cheaply opt out.
    """
    if n_layers <= 0:
        return None
    return nn.ModuleList(
        [
            SelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
    )


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

    Optional ``use_alibi_bias`` adds an ALiBi-style time-decay bias to the
    pre-softmax attention scores: ``score(b, h, s) += -slope[h] * games_ago(b, s)``.
    ``games_ago`` is computed per row from the mask (caller must place real
    games at positions ``0..n_real-1``, oldest first) so older games get
    a larger additive penalty. Complements or replaces the absolute learned
    ``pos_embedding`` on the parent models — ALiBi generalises to variable
    sequence lengths without an extra parameter table.

    Optional ``cond_dim`` enables opponent/context-conditioned queries. When
    set >0, forward expects a ``context: [batch, cond_dim]`` tensor; queries
    become per-row ``q_base + cond_proj(context)`` so each target can ask a
    different question depending on who the player is facing. ``cond_proj``
    is zero-initialised so ``q_per_row == q_base`` at step zero, preserving
    baseline behaviour exactly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        n_targets: int = 1,
        project_kv: bool = False,
        attn_dropout: float = 0.0,
        learn_temperature: bool = False,
        compute_entropy: bool = False,
        use_alibi_bias: bool = False,
        cond_dim: int = 0,
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

        # When ``compute_entropy`` is set, forward stores the mean entropy of
        # this forward's attention weights on ``self.last_attn_entropy`` so a
        # regulariser can pull it post-forward without an extra return tuple.
        # Disabled by default to keep the hot path allocation-free.
        self.compute_entropy = compute_entropy

        # ALiBi-style linear distance bias. Slopes are the standard geometric
        # sequence from Press et al. 2022 (``2^(-8*i/n_heads)``) — a sensible
        # generic default. Stored as a buffer (not a parameter) so they're
        # constant across training. Empty when the feature is off.
        self.use_alibi_bias = use_alibi_bias
        if use_alibi_bias:
            self.register_buffer("alibi_slopes", self._alibi_slopes(n_heads))
        else:
            self.alibi_slopes = None

        # Context conditioning: per-row query delta ``cond_proj(context)``
        # added onto the static ``q_base``. Zero-init the projection so the
        # conditioned pool is numerically identical to the non-conditioned
        # baseline at step zero — any drift has to be learned explicitly.
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, n_targets * n_heads * d_model)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

    @staticmethod
    def _alibi_slopes(n_heads: int) -> torch.Tensor:
        """Geometric ALiBi slope sequence: ``slope_h = 2^(-8 * (h+1) / n_heads)``.

        Matches Press, Smith & Lewis 2022. For small ``n_heads`` (e.g. 2)
        this gives slopes ≈ (0.0625, 0.00390625); the larger slope's head
        learns a recency-biased distribution while the smaller slope's head
        stays close to a uniform-over-history pattern.
        """
        return torch.tensor(
            [2.0 ** (-8.0 * (h + 1) / n_heads) for h in range(n_heads)],
            dtype=torch.float32,
        )

    def forward(
        self,
        keys: torch.Tensor,
        mask: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            keys: [batch, seq_len, d_model] — encoded game history
            mask: [batch, seq_len] — True where real game, False where padding
            context: [batch, cond_dim] — per-row conditioning tensor. Ignored
                when ``cond_dim == 0``. Required when ``cond_dim > 0``.

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
        if self.cond_dim > 0 and context is not None:
            # Per-row query delta from the conditioning tensor. Zero-init of
            # ``cond_proj`` makes delta exactly zero at step 0; gradients
            # flow through ``context`` and the projection weights thereafter.
            delta = self.cond_proj(context).view(
                batch_size, self.n_targets * self.n_heads, self.d_model
            )
            q_per_row = q_flat.unsqueeze(0) + delta  # [batch, n_t*n_h, d_model]
            attn = torch.einsum("bqd,bsd->bqs", q_per_row, k) * self.scale
        else:
            # attn scores: [batch, n_targets * n_heads, seq_len]
            attn = torch.einsum("qd,bsd->bqs", q_flat, k) * self.scale

        if self.learn_temperature:
            # Per-target inverse-temperature, replicated across heads so one
            # scalar controls all heads of the same target. Shape broadcasts
            # over [batch, ..., seq_len].
            inv_t = torch.exp(-self.log_temperature)  # [n_targets]
            inv_t = inv_t.repeat_interleave(self.n_heads).view(1, -1, 1)
            attn = attn * inv_t

        if self.use_alibi_bias:
            # Per-row games_ago: the most-recent real game is 0, the oldest is
            # n_real-1. With right-padding, position ``s`` of a row with
            # ``n_real`` real games has games_ago = n_real - 1 - s. When mask
            # is None (tests), all positions are treated as real.
            seq_len = keys.size(1)
            if mask is not None:
                n_real = mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            else:
                n_real = torch.full((batch_size, 1), seq_len, device=keys.device, dtype=torch.long)
            positions = torch.arange(seq_len, device=keys.device)  # [seq_len]
            games_ago = (n_real - 1 - positions).clamp(min=0).float()  # [batch, seq_len]
            # Slopes repeated across targets so target t shares the same slope
            # set across its n_heads. Flat order matches q_flat reshape, so
            # ``slopes.repeat(n_targets)`` yields [h0, h1, ..., h0, h1, ...].
            slopes = self.alibi_slopes.repeat(self.n_targets).view(1, -1, 1)
            attn = attn - slopes * games_ago.unsqueeze(1)

        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, seq_len]
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        weights = F.softmax(attn, dim=-1)
        # Handle all-padding rows (softmax of all -inf = nan)
        weights = weights.nan_to_num(0.0)

        if self.compute_entropy:
            # H(p) = -sum_s p_s * log(p_s). For rows of all-zero (all padding)
            # the sum degenerates to 0 thanks to the +eps floor inside log.
            # Averaged across batch and all n_targets * n_heads distributions,
            # yielding a single scalar for the regulariser to consume.
            eps = 1e-12
            per_dist_entropy = -(weights * (weights + eps).log()).sum(dim=-1)
            self.last_attn_entropy = per_dist_entropy.mean()

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
        history_dropout: float = 0.0,
        use_swiglu_encoder: bool = False,
        attn_entropy_coeff: float = 0.0,
        use_alibi_bias: bool = False,
        self_attn_layers: int = 0,
        self_attn_heads: int = 2,
        self_attn_ffn_dim: int | None = None,
        self_attn_dropout: float = 0.0,
        condition_queries_on_static: bool = False,
        opp_game_dim: int | None = None,
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
        # Opponent/context-conditioned queries. When enabled, the forward
        # passes the raw static feature vector to the AttentionPool as
        # ``context``; the pool adds a per-row projection of that vector to
        # the static ``q_base``. Zero-init on the projection preserves
        # baseline behaviour at step 0.
        self.condition_queries_on_static = condition_queries_on_static
        # Sequence-level dropout over the outer history mask. Regularises
        # attention against fixating on any one game and forces every head to
        # derive signal from a rotating subset of the season.
        self.history_dropout = history_dropout
        self.use_swiglu_encoder = use_swiglu_encoder

        # Entropy regulariser: loss += coeff * mean(H(attn_weights)).
        # Positive coeff ⇒ penalises high entropy ⇒ sharper attention.
        # Negative coeff ⇒ penalises low  entropy ⇒ smoother attention.
        # Zero is a strict no-op: the pool skips the entropy computation and
        # MultiHeadTrainer never calls ``attention_entropy_loss``.
        self.attn_entropy_coeff = attn_entropy_coeff

        # === Game History Branch ===
        self.game_encoder = _build_game_encoder(
            in_dim=game_dim,
            d_model=d_model,
            encoder_hidden_dim=encoder_hidden_dim,
            use_swiglu=use_swiglu_encoder,
        )

        # Positional encoding: gives the model temporal ordering signal
        # so it can distinguish recent games from older ones.
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Optional Pre-LN transformer stack that lets games contextualise each
        # other (hot streak, bye-week rest, post-injury ramp) before the
        # learned-query pool extracts per-target summaries. Default ffn dim
        # follows the standard transformer 4× rule when not explicitly set.
        ffn_dim = self_attn_ffn_dim if self_attn_ffn_dim is not None else 4 * d_model
        self.self_attn_stack = _build_self_attention_stack(
            d_model=d_model,
            n_layers=self_attn_layers,
            n_heads=self_attn_heads,
            dim_feedforward=ffn_dim,
            dropout=self_attn_dropout,
        )

        # Per-target queries: each target pulls its own slice of history
        # instead of routing through a single shared summary.
        self.attn_pool = AttentionPool(
            d_model,
            n_heads=n_attn_heads,
            n_targets=self.n_targets,
            project_kv=project_kv,
            attn_dropout=attn_dropout,
            learn_temperature=learn_attn_temperature,
            compute_entropy=(attn_entropy_coeff != 0.0),
            use_alibi_bias=use_alibi_bias,
            cond_dim=static_dim if condition_queries_on_static else 0,
        )

        attn_out_dim = n_attn_heads * d_model
        # Per-target LayerNorms — each target's pooled stats live at different
        # scales (rushing_tds ~1, rushing_yards ~80), so one norm per target.
        self.history_norms = nn.ModuleList([nn.LayerNorm(attn_out_dim) for _ in target_names])

        # === Optional Opponent-Defense History Branch ===
        # Parallel attention branch over the opposing defense's game log.
        # Active when ``opp_game_dim`` is provided (a second per-game feature
        # dimension). Disabled (None) ⇒ model behaviour is byte-identical to
        # the single-branch version, which keeps RB / K / DST unaffected.
        self.use_opp_history = opp_game_dim is not None
        if self.use_opp_history:
            self.opp_game_encoder = _build_game_encoder(
                in_dim=opp_game_dim,
                d_model=d_model,
                encoder_hidden_dim=encoder_hidden_dim,
                use_swiglu=use_swiglu_encoder,
            )
            if use_positional_encoding:
                # Separate embedding so the opp branch learns its own recency
                # weighting independent of the player-history branch.
                self.opp_pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.opp_attn_pool = AttentionPool(
                d_model,
                n_heads=n_attn_heads,
                n_targets=self.n_targets,
                project_kv=project_kv,
                attn_dropout=attn_dropout,
                learn_temperature=learn_attn_temperature,
                compute_entropy=(attn_entropy_coeff != 0.0),
            )
            self.opp_history_norms = nn.ModuleList(
                [nn.LayerNorm(attn_out_dim) for _ in target_names]
            )

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

        # === Output Heads (consume shared_static ⊕ per-target histories) ===
        # Head input: [static_rep, player_hist_vec, (opp_hist_vec if enabled)]
        head_in_dim = (
            backbone_out_dim + attn_out_dim + (attn_out_dim if self.use_opp_history else 0)
        )
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
        x_opp_history: torch.Tensor | None = None,
        opp_history_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            x_static: [batch, static_dim] — standard feature vector
            x_history: [batch, seq_len, game_dim] — padded player-game history
            history_mask: [batch, seq_len] — True for real games
            x_opp_history: [batch, opp_seq_len, opp_game_dim] — padded opponent
                defense game history. Required iff the model was built with
                ``opp_game_dim``; ignored otherwise.
            opp_history_mask: [batch, opp_seq_len] — True for real opp games.
        """
        # Encode game history
        encoded = self.game_encoder(x_history)  # [batch, seq_len, d_model]

        # Add positional encoding so attention can weight by recency
        if self.use_positional_encoding:
            seq_len = encoded.size(1)
            positions = torch.arange(seq_len, device=encoded.device)
            encoded = encoded + self.pos_embedding(positions)

        # Sequence dropout: during training, randomly mask real games so the
        # pool never locks onto any one game. Eval uses the full mask.
        history_mask = _apply_history_dropout(history_mask, self.history_dropout, self.training)

        # Optional self-attention stack: lets games attend to each other
        # before the learned-query pool extracts per-target summaries. Sits
        # after positional encoding so temporal signal is available to the
        # MHA, and after sequence dropout so it sees the same dropped-games
        # view as the pool (consistent regularisation).
        if self.self_attn_stack is not None:
            for block in self.self_attn_stack:
                encoded = block(encoded, mask=history_mask)

        # Per-target attention pool: [batch, n_targets, n_heads * d_model].
        # When query conditioning is on, the pool adds ``cond_proj(x_static)``
        # to the static ``q_base`` so each row's query reflects this matchup.
        context = x_static if self.condition_queries_on_static else None
        history_per_target = self.attn_pool(encoded, history_mask, context=context)
        if history_per_target.dim() == 2:
            # Back-compat guard: single-target AttentionPool squeezes. Add axis.
            history_per_target = history_per_target.unsqueeze(1)

        # Optional opponent-defense history branch — parallel encode + pool.
        opp_per_target = None
        if self.use_opp_history:
            if x_opp_history is None or opp_history_mask is None:
                raise ValueError(
                    "Model was built with opp_game_dim set; forward() requires "
                    "x_opp_history and opp_history_mask."
                )
            opp_encoded = self.opp_game_encoder(x_opp_history)
            if self.use_positional_encoding:
                opp_seq_len = opp_encoded.size(1)
                opp_positions = torch.arange(opp_seq_len, device=opp_encoded.device)
                opp_encoded = opp_encoded + self.opp_pos_embedding(opp_positions)
            opp_mask = _apply_history_dropout(opp_history_mask, self.history_dropout, self.training)
            opp_per_target = self.opp_attn_pool(opp_encoded, opp_mask)
            if opp_per_target.dim() == 2:
                opp_per_target = opp_per_target.unsqueeze(1)

        # Shared backbone processes static features only.
        shared_static = self.backbone(x_static)

        preds = {}
        for i, name in enumerate(self.target_names):
            history_vec = self.history_norms[i](history_per_target[:, i, :])
            if self.use_gated_fusion:
                gate = self.fusion_gates[i](torch.cat([x_static, history_vec], dim=-1))
                history_vec = gate * history_vec

            parts = [shared_static, history_vec]
            if self.use_opp_history:
                opp_vec = self.opp_history_norms[i](opp_per_target[:, i, :])
                parts.append(opp_vec)
            head_input = torch.cat(parts, dim=-1)
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

    def attention_entropy_loss(self) -> torch.Tensor | None:
        """Post-forward regulariser: ``coeff * E[H(attn_weights)]``.

        Returns ``None`` when the coefficient is 0 (the feature is off) or no
        forward has materialised ``last_attn_entropy`` yet. MultiHeadTrainer
        treats ``None`` as "skip" to keep the hot path allocation-free.

        When the opp-history branch is active its entropy is averaged with
        the player-history branch's so one coefficient controls both.
        """
        if self.attn_entropy_coeff == 0.0:
            return None
        entropies = []
        player_entropy = getattr(self.attn_pool, "last_attn_entropy", None)
        if player_entropy is not None:
            entropies.append(player_entropy)
        if self.use_opp_history:
            opp_entropy = getattr(self.opp_attn_pool, "last_attn_entropy", None)
            if opp_entropy is not None:
                entropies.append(opp_entropy)
        if not entropies:
            return None
        mean_entropy = entropies[0] if len(entropies) == 1 else sum(entropies) / len(entropies)
        return self.attn_entropy_coeff * mean_entropy

    def predict_numpy(
        self,
        X_static: np.ndarray,
        X_history: np.ndarray,
        history_mask: np.ndarray,
        device: torch.device,
        X_opp_history: np.ndarray | None = None,
        opp_history_mask: np.ndarray | None = None,
    ) -> dict:
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(X_static).to(device)
            h = torch.FloatTensor(X_history).to(device)
            m = torch.BoolTensor(history_mask).to(device)
            opp_h = None
            opp_m = None
            if self.use_opp_history:
                if X_opp_history is None or opp_history_mask is None:
                    raise ValueError(
                        "predict_numpy requires X_opp_history and opp_history_mask "
                        "for a model built with opp_game_dim."
                    )
                opp_h = torch.FloatTensor(X_opp_history).to(device)
                opp_m = torch.BoolTensor(opp_history_mask).to(device)
            preds = self.forward(s, h, m, opp_h, opp_m)
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
        history_dropout: float = 0.0,
        use_swiglu_encoder: bool = False,
        attn_entropy_coeff: float = 0.0,
        use_alibi_bias: bool = False,
        self_attn_layers: int = 0,
        self_attn_heads: int = 2,
        self_attn_ffn_dim: int | None = None,
        self_attn_dropout: float = 0.0,
        condition_queries_on_static: bool = False,
    ):
        super().__init__()
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self.d_model = d_model
        self.d_kick = d_kick
        self.n_targets = len(target_names)
        # Opponent/context-conditioned queries. Outer (game) pool only — the
        # inner kick pool has no notion of matchup context.
        self.condition_queries_on_static = condition_queries_on_static
        # Game-level sequence dropout. Only the outer (game) mask is
        # perturbed — kick-level inner masks stay intact since K already has
        # <=10 kicks per game and dropping kicks is a different granularity.
        self.history_dropout = history_dropout
        self.use_swiglu_encoder = use_swiglu_encoder
        # Entropy regulariser. Same contract as MultiHeadNetWithHistory —
        # applied to the *outer* game-level pool only; the inner kick pool
        # is a different granularity and is left untouched.
        self.attn_entropy_coeff = attn_entropy_coeff

        # === Inner: per-kick encoder + single-query pool ===
        # Kick encoder stays as Linear+ReLU regardless of the SwiGLU flag —
        # the flag targets the outer *game* encoder per the attention-layer
        # spec. If kick_encoder ever becomes a SwiGLU target too, expose a
        # separate flag rather than overloading this one.
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
        self.game_encoder = _build_game_encoder(
            in_dim=d_kick,
            d_model=d_model,
            encoder_hidden_dim=encoder_hidden_dim,
            use_swiglu=use_swiglu_encoder,
        )

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_games, d_model)

        # Optional Pre-LN transformer stack on the outer (game) sequence only.
        # Inner kick pool is intentionally untouched — kicks within a game
        # aren't a "games attending to each other" problem.
        ffn_dim = self_attn_ffn_dim if self_attn_ffn_dim is not None else 4 * d_model
        self.self_attn_stack = _build_self_attention_stack(
            d_model=d_model,
            n_layers=self_attn_layers,
            n_heads=self_attn_heads,
            dim_feedforward=ffn_dim,
            dropout=self_attn_dropout,
        )

        # Outer pool receives the ALiBi bias when enabled. Inner kick pool
        # stays unchanged — kicks within a game have no "games-ago" notion,
        # so the bias doesn't apply at that granularity.
        self.attn_pool = AttentionPool(
            d_model,
            n_heads=n_attn_heads,
            n_targets=self.n_targets,
            project_kv=project_kv,
            attn_dropout=attn_dropout,
            learn_temperature=learn_attn_temperature,
            compute_entropy=(attn_entropy_coeff != 0.0),
            use_alibi_bias=use_alibi_bias,
            cond_dim=static_dim if condition_queries_on_static else 0,
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

        # Game-level sequence dropout on the outer mask only.
        outer_mask = _apply_history_dropout(outer_mask, self.history_dropout, self.training)

        # Optional self-attention stack over games (outer sequence only).
        if self.self_attn_stack is not None:
            for block in self.self_attn_stack:
                encoded = block(encoded, mask=outer_mask)

        context = x_static if self.condition_queries_on_static else None
        history_per_target = self.attn_pool(
            encoded, outer_mask, context=context
        )  # [B, n_targets, attn_out]
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

    def attention_entropy_loss(self) -> torch.Tensor | None:
        """Post-forward regulariser for the outer (game-level) attention pool.

        See ``MultiHeadNetWithHistory.attention_entropy_loss`` for the
        contract. The inner kick pool is deliberately excluded.
        """
        if self.attn_entropy_coeff == 0.0:
            return None
        entropy = getattr(self.attn_pool, "last_attn_entropy", None)
        if entropy is None:
            return None
        return self.attn_entropy_coeff * entropy

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
    opp_game_dim: int | None = None,
) -> "MultiHeadNetWithHistory":
    """Construct a MultiHeadNetWithHistory from a training ``cfg`` dict.

    When ``opp_game_dim`` is provided, the model adds a parallel attention
    branch over the opposing defense's per-game history (see
    ``MultiHeadNetWithHistory.__init__``). When ``None`` (the default), the
    model is identical to the single-branch version — the code path used by
    RB / K / DST is unchanged.
    """
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
        history_dropout=cfg.get("attn_history_dropout", 0.0),
        use_swiglu_encoder=cfg.get("attn_use_swiglu_encoder", False),
        attn_entropy_coeff=cfg.get("attn_entropy_coeff", 0.0),
        use_alibi_bias=cfg.get("attn_use_alibi_bias", False),
        self_attn_layers=cfg.get("attn_self_layers", 0),
        self_attn_heads=cfg.get("attn_self_heads", 2),
        self_attn_ffn_dim=cfg.get("attn_self_ffn_dim"),
        self_attn_dropout=cfg.get("attn_self_dropout", 0.0),
        condition_queries_on_static=cfg.get("attn_condition_queries_on_static", False),
        opp_game_dim=opp_game_dim,
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
        history_dropout=cfg.get("attn_history_dropout", 0.0),
        use_swiglu_encoder=cfg.get("attn_use_swiglu_encoder", False),
        attn_entropy_coeff=cfg.get("attn_entropy_coeff", 0.0),
        use_alibi_bias=cfg.get("attn_use_alibi_bias", False),
        self_attn_layers=cfg.get("attn_self_layers", 0),
        self_attn_heads=cfg.get("attn_self_heads", 2),
        self_attn_ffn_dim=cfg.get("attn_self_ffn_dim"),
        self_attn_dropout=cfg.get("attn_self_dropout", 0.0),
        condition_queries_on_static=cfg.get("attn_condition_queries_on_static", False),
    )
