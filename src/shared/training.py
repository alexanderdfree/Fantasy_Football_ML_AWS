"""Generic training infrastructure: loss, dataset, dataloaders, and trainer."""

import contextlib
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.shared.utils import (
    amp_dtype,
    cuda_graph_enabled,
    cuda_graph_full_enabled,
    cuda_graph_opt_enabled,
)

SUPPORTED_HEAD_LOSSES = ("huber", "mse", "poisson_nll", "hurdle_negbin", "hurdle_poisson")
_TRUE_ENV = {"1", "true", "yes", "on"}
_FIXED_SCALE_ENV = "FF_AMP_FIXED_SCALE"
_INIT_SCALE_ENV = "FF_AMP_INIT_SCALE"
_GRADSCALER_TRACE_PATH_ENV = "FF_GRADSCALER_TRACE_PATH"
_GRADSCALER_TRACE_LABEL_ENV = "FF_GRADSCALER_TRACE_LABEL"
_CUDA_GRAPH_RESTORE_BN_ENV = "FF_CUDA_GRAPH_RESTORE_BN"

# DataLoader worker count is fixed at 0 (the PyTorch default). PR #309's
# GPU-resident batcher path takes over on CUDA hosts, so DataLoader is only
# reached on CPU/MPS, where prefetching gives no benefit and ``spawn``-context
# worker startup adds hundreds of ms per loader on macOS dev boxes. The
# previous ``NN_DATALOADER_NUM_WORKERS`` env var (with a CUDA-conditional
# default) was load-bearing only on the CUDA DataLoader path that PR #309
# obsoleted; the Batch job-definition still sets ``NN_DATALOADER_NUM_WORKERS=3``
# in batch-image.yml but it is now a no-op the orchestrator can clear at
# leisure.


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _snapshot_batchnorm_state(module: nn.Module) -> list[tuple[nn.Module, dict[str, torch.Tensor]]]:
    """Clone BatchNorm running buffers so CUDA-graph warmup can be made symmetric."""
    snapshot: list[tuple[nn.Module, dict[str, torch.Tensor]]] = []
    for child in module.modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        state = {
            name: buf.detach().clone()
            for name in ("running_mean", "running_var", "num_batches_tracked")
            if (buf := getattr(child, name, None)) is not None
        }
        if state:
            snapshot.append((child, state))
    return snapshot


def _restore_batchnorm_state(
    snapshot: list[tuple[nn.Module, dict[str, torch.Tensor]]],
) -> None:
    for module, state in snapshot:
        for name, saved in state.items():
            getattr(module, name).copy_(saved)


def _optimizer_is_fused_capturable(optimizer) -> bool:
    """True iff every param group runs fused + capturable AdamW.

    The optimizer-tail CUDA graph (Lever A3, :class:`_GraphedFullStep`) bakes
    the AdamW step; ``capturable=True`` keeps its step counter + LR on-device so
    the step is graph-safe, and ``fused=True`` collapses the update to one
    kernel. ``_run_nn_training`` sets both together under the same A3 gate; this
    is the trainer-side defensive precondition so a non-capturable optimizer
    never reaches capture.
    """
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return False
    return all(g.get("fused") and g.get("capturable") for g in groups)


def negbin2_log_prob(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Log-pmf of the NB-2 parameterization: mean ``mu``, ``var = mu + alpha*mu^2``.

    Equivalent to ``NegBin(r=1/alpha, p=r/(r+mu))``. Supports ``y=0``.
    """
    alpha = torch.clamp(alpha, min=1e-6)
    mu = torch.clamp(mu, min=1e-10)
    r = 1.0 / alpha
    log_coeff = torch.lgamma(y + r) - torch.lgamma(y + 1.0) - torch.lgamma(r)
    log_r_ratio = torch.log(r) - torch.log(r + mu)
    log_mu_ratio = torch.log(mu) - torch.log(r + mu)
    return log_coeff + r * log_r_ratio + y * log_mu_ratio


def ztnb2_log_prob(y: torch.Tensor, mu: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
    """Zero-truncated NB-2 log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu, alpha) = log P_NB(k) - log(1 - P_NB(0))``.
    """
    alpha = torch.exp(log_alpha)
    log_p = negbin2_log_prob(y, mu, alpha)
    log_p_zero = negbin2_log_prob(torch.zeros_like(y), mu, alpha)
    # log(1 - p_zero) via log1p for numerical stability when p_zero is small.
    log_survival = torch.log1p(-torch.exp(log_p_zero).clamp(max=1.0 - 1e-7))
    return log_p - log_survival


def ztp_log_prob(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Zero-truncated Poisson log-pmf. Only valid for ``y >= 1``.

    ``log P(Y=k | Y>0, mu) = k*log(mu) - mu - lgamma(k+1) - log(1 - exp(-mu))``.
    Mirrors ztnb2 but without the dispersion parameter — appropriate when
    empirical var/mean ≈ 1 (e.g. RB rushing_tds ≈ 1.16, fumbles_lost ≈ 1.01).
    """
    mu = torch.clamp(mu, min=1e-10)
    log_p = y * torch.log(mu) - mu - torch.lgamma(y + 1.0)
    log_survival = torch.log1p(-torch.exp(-mu).clamp(max=1.0 - 1e-7))
    return log_p - log_survival


def hurdle_negbin_value_loss(preds: dict, targets: dict, name: str) -> torch.Tensor:
    """Zero-truncated NB-2 NLL on positive samples, scaled by fraction positive.

    The gate component (BCE on ``y > 0``) is emitted separately by
    ``MultiTargetLoss`` via its ``gated_targets`` loop, so this function only
    returns the conditional-value contribution. Scaling by ``frac_pos`` makes
    the magnitude directly comparable to the full-batch Huber/Poisson losses
    on neighbouring heads (same per-sample basis over the batch of N).

    Requires ``preds[f"{name}_value_mu"]`` and ``preds[f"{name}_value_log_alpha"]``.
    """
    y = targets[name]
    mu = preds[f"{name}_value_mu"]
    log_alpha = preds[f"{name}_value_log_alpha"]
    pos_mask = y > 0
    if pos_mask.any():
        ztnb_nll = -ztnb2_log_prob(y[pos_mask], mu[pos_mask], log_alpha[pos_mask]).mean()
        frac_pos = pos_mask.float().mean()
        return frac_pos * ztnb_nll
    return torch.zeros((), device=y.device, dtype=y.dtype)


def hurdle_poisson_value_loss(preds: dict, targets: dict, name: str) -> torch.Tensor:
    """Zero-truncated Poisson NLL on positives, scaled by fraction positive.

    Mirrors ``hurdle_negbin_value_loss`` but uses ZTP (no dispersion). Gate
    component (BCE on ``y > 0``) is added separately via
    ``MultiTargetLoss.gated_targets``. Requires ``preds[f"{name}_value_mu"]``;
    ``log_alpha`` is unused (GatedHead emits it regardless but ZTP ignores it).
    """
    y = targets[name]
    mu = preds[f"{name}_value_mu"]
    pos_mask = y > 0
    if pos_mask.any():
        ztp_nll = -ztp_log_prob(y[pos_mask], mu[pos_mask]).mean()
        frac_pos = pos_mask.float().mean()
        return frac_pos * ztp_nll
    return torch.zeros((), device=y.device, dtype=y.dtype)


def hurdle_negbin_value_loss_capturable(preds: dict, targets: dict, name: str) -> torch.Tensor:
    """Branch-free ``hurdle_negbin_value_loss`` for CUDA-graph capture.

    The branchy original's ``if pos_mask.any():`` is a host-side, data-
    dependent branch — it forces a GPU→CPU sync AND aborts graph capture.
    Algebraic identity used here:
    ``frac_pos * mean_over_pos(nll) == masked_nll_sum / N``, and the
    no-positives case degenerates to the same 0. Masked-out lanes are fed
    ``y=1`` substitutes so the log-prob stays finite before the mask zeroes
    it (``0 * inf`` would poison the sum).
    """
    y = targets[name]
    mu = preds[f"{name}_value_mu"]
    log_alpha = preds[f"{name}_value_log_alpha"]
    mask = (y > 0).to(y.dtype)
    y_safe = torch.where(y > 0, y, torch.ones_like(y))
    log_prob = ztnb2_log_prob(y_safe, mu, log_alpha)
    return -(log_prob * mask).sum() / y.shape[0]


def hurdle_poisson_value_loss_capturable(preds: dict, targets: dict, name: str) -> torch.Tensor:
    """Branch-free ``hurdle_poisson_value_loss`` — same identity as
    :func:`hurdle_negbin_value_loss_capturable`, ZTP value family."""
    y = targets[name]
    mu = preds[f"{name}_value_mu"]
    mask = (y > 0).to(y.dtype)
    y_safe = torch.where(y > 0, y, torch.ones_like(y))
    log_prob = ztp_log_prob(y_safe, mu)
    return -(log_prob * mask).sum() / y.shape[0]


class MultiTargetLoss(nn.Module):
    """Per-head dispatchable loss for a multi-head network.

    Each target is assigned a loss family via ``head_losses[name]``; supported
    values are in ``SUPPORTED_HEAD_LOSSES``:
      - ``"huber"`` — standard Huber loss with per-target delta.
      - ``"mse"`` — plain ``MSELoss`` (squared error). Unlike Huber it does not
        cap large-residual gradients, so it chases the heavy upper tail (elite
        weeks) instead of shrinking toward the mean; ``huber_deltas`` is ignored
        for these heads. Pair with a per-target ``loss_weights`` of ``1/delta``
        (gradient-matched to the old ``2/delta`` Huber weighting at the
        characteristic error) so the head does not dominate the combined loss.
      - ``"poisson_nll"`` — ``PoissonNLLLoss(log_input=False)``. Treats the head
        output as the rate lambda directly; requires a non-negative clamp on
        that head (``MultiHeadNet`` provides this via ``non_negative_targets``).
      - ``"hurdle_negbin"`` — zero-truncated NB-2 NLL on positives only (value
        component). The gate component (BCE on ``y>0``) is added through the
        ``gated_targets`` mechanism. Requires the target's head to emit
        ``{name}_gate_logit``, ``{name}_value_mu``, and ``{name}_value_log_alpha``
        in the prediction dict — ``GatedHead`` does this.
      - ``"hurdle_poisson"`` — same hurdle structure as ``hurdle_negbin`` but
        with a zero-truncated Poisson value loss (no dispersion). Appropriate
        when empirical var/mean ≈ 1 (TD heads, fumbles_lost). Consumes
        ``{name}_value_mu`` and the gate logit; ``{name}_value_log_alpha`` is
        emitted by ``GatedHead`` but ignored by this loss.

    ``poisson_targets`` is a back-compat shorthand accepted alongside
    ``head_losses``: each listed target is treated as if it had
    ``head_losses[t] = "poisson_nll"``. Prefer ``head_losses`` for new code.

    ``gated_targets`` is the list of target names whose heads emit a
    ``{name}_gate_logit`` key; they receive an additional
    ``gate_weight * BCE(gate_logit, (target > 0))`` component. Must be a
    superset of the hurdle targets (``"hurdle_negbin"`` / ``"hurdle_poisson"``)
    so the hurdle gate is trained.

    Loss:
        sum(weight[t] * loss_fn[t](pred[t], target[t]) for t in targets)
        + sum(gate_weight * BCE(gate_logit_t, (target_t > 0)) for t in gated_targets)
    """

    def __init__(
        self,
        target_names: list[str],
        loss_weights: dict[str, float],
        huber_deltas: dict[str, float] = None,
        head_losses: dict[str, str] | None = None,
        gate_weight: float = 1.0,
        gated_targets: list[str] | None = None,
        poisson_targets: list[str] | None = None,
    ):
        super().__init__()
        self.target_names = target_names
        self.gated_targets = list(gated_targets) if gated_targets else []
        self.loss_weights = {n: loss_weights.get(n, 1.0) for n in target_names}
        self.gate_weight = gate_weight
        if huber_deltas is None:
            huber_deltas = {}
        if head_losses is None:
            head_losses = {}
        if poisson_targets:
            head_losses = {**head_losses, **{t: "poisson_nll" for t in poisson_targets}}
        self.head_losses = {n: head_losses.get(n, "huber") for n in target_names}

        unknown = {n: lt for n, lt in self.head_losses.items() if lt not in SUPPORTED_HEAD_LOSSES}
        if unknown:
            raise ValueError(
                f"Unsupported head_losses (supported: {SUPPORTED_HEAD_LOSSES}): {unknown}"
            )

        # Hurdle families need the gate pathway: preds must carry value_mu
        # (and value_log_alpha for NegBin), which only GatedHead emits (enabled
        # for targets in gated_targets). Catch the misconfiguration at
        # construction time rather than crashing with a KeyError on the first batch.
        hurdle_set = {
            n for n, lt in self.head_losses.items() if lt in ("hurdle_negbin", "hurdle_poisson")
        }
        gated_set = set(self.gated_targets)
        missing_gates = hurdle_set - gated_set
        if missing_gates:
            raise ValueError(
                f"head_losses='hurdle_negbin' or 'hurdle_poisson' requires the "
                f"target to also be in gated_targets (so GatedHead emits "
                f"value_mu). Missing from gated_targets: {sorted(missing_gates)}"
            )

        # Hurdle families need the full preds dict (value_mu, optionally
        # value_log_alpha), so they're dispatched inline in ``forward`` rather
        # than through loss_fns.
        def _plain_loss_fn(lt: str, name: str) -> nn.Module:
            if lt == "poisson_nll":
                return nn.PoissonNLLLoss(log_input=False, full=False)
            if lt == "mse":
                return nn.MSELoss()
            return nn.HuberLoss(delta=huber_deltas.get(name, 1.0))

        self.loss_fns = nn.ModuleDict(
            {
                name: _plain_loss_fn(lt, name)
                for name, lt in self.head_losses.items()
                if lt in ("huber", "mse", "poisson_nll")
            }
        )

    def _compute_loss_components(self, preds: dict, targets: dict) -> tuple:
        """Internal: compute combined loss + per-target tensor components.

        Returns ``(combined, components)`` where ``components`` values are
        on-device scalar tensors (not Python floats). This is the form the
        trainer's val branch accumulates to avoid per-batch GPU→CPU sync via
        ``.item()`` (PR #305 removed the train-branch sync; this is the
        remaining val-branch piece).

        ``forward`` calls this and ``.item()``s each component for backward
        compatibility with callers that expect float values
        (``tests/{te,dst}/test_training.py::test_components_are_scalars``).
        """
        per_target_losses = {}
        # FP32 accumulator on purpose: under AMP autocast the per-target losses
        # may be FP16, but ``combined`` is the value handed to .backward(), so
        # reducing in FP32 keeps the weighted multi-head sum numerically stable
        # (no FP16 round-off accumulating across heads). Not a missed-cast perf
        # regression (#369 F20). ``torch.zeros`` (device-native fill), NOT
        # ``torch.tensor(0.0, device=)`` — the latter stages a Python scalar
        # through pageable CPU memory, a hidden H2D copy per step on the eager
        # path and a capture-abort inside a CUDA graph.
        combined = torch.zeros((), device=next(iter(preds.values())).device, dtype=torch.float32)
        for name in self.target_names:
            lt = self.head_losses[name]
            if lt == "hurdle_negbin":
                loss = hurdle_negbin_value_loss(preds, targets, name)
            elif lt == "hurdle_poisson":
                loss = hurdle_poisson_value_loss(preds, targets, name)
            else:
                loss = self.loss_fns[name](preds[name], targets[name])
            per_target_losses[name] = loss
            combined = combined + self.loss_weights[name] * loss

        components = {f"loss_{name}": loss for name, loss in per_target_losses.items()}

        for gated_name in self.gated_targets:
            gate_key = f"{gated_name}_gate_logit"
            if gate_key in preds:
                gate_loss = F.binary_cross_entropy_with_logits(
                    preds[gate_key], (targets[gated_name] > 0).float()
                )
                combined = combined + self.gate_weight * gate_loss
                components[f"loss_gate_{gated_name}"] = gate_loss

        components["loss_combined"] = combined
        return combined, components

    def forward(self, preds: dict, targets: dict) -> tuple:
        combined, tensor_components = self._compute_loss_components(preds, targets)
        # Preserve the historical float-valued ``components`` contract that
        # ``tests/{te,dst}/test_training.py::test_components_are_scalars``
        # depends on. The val branch in ``MultiHeadTrainer.train`` calls
        # ``_compute_loss_components`` directly to keep tensors on-device.
        components = {k: v.item() for k, v in tensor_components.items()}
        return combined, components

    def compute_combined_capturable(self, preds: dict, targets: dict) -> torch.Tensor:
        """``combined`` only, with branch-free hurdle dispatch — CUDA-graph-safe.

        Numerically equivalent to ``_compute_loss_components``'s ``combined``
        for every head family: Huber / MSE / PoissonNLL / gate-BCE are already
        pure tensor ops, and the hurdle families swap their data-dependent
        ``if pos_mask.any():`` host branch for the algebraically identical
        masked-sum form (see ``hurdle_*_value_loss_capturable``). The
        per-head dispatch below is config-static (same branch every step), so
        it does not break capture. Used by ``_GraphedTrainStep``; eager paths
        keep the branchy originals byte-identical.

        The accumulator MUST be ``torch.zeros`` (device-native fill):
        ``torch.tensor(0.0, device=)`` stages the scalar through pageable CPU
        memory, and that unpinned H2D copy aborts CUDA graph capture
        ("Cannot copy between CPU and CUDA tensors during CUDA graph
        capture") — measured on the first Batch graphfull smoke, 2026-06-11,
        where every trial fell back to the model-only graph. FP32 dtype
        mirrors ``_compute_loss_components``'s deliberate FP32 accumulation.
        """
        combined = torch.zeros((), device=next(iter(preds.values())).device, dtype=torch.float32)
        for name in self.target_names:
            lt = self.head_losses[name]
            if lt == "hurdle_negbin":
                loss = hurdle_negbin_value_loss_capturable(preds, targets, name)
            elif lt == "hurdle_poisson":
                loss = hurdle_poisson_value_loss_capturable(preds, targets, name)
            else:
                loss = self.loss_fns[name](preds[name], targets[name])
            combined = combined + self.loss_weights[name] * loss

        for gated_name in self.gated_targets:
            gate_key = f"{gated_name}_gate_logit"
            if gate_key in preds:
                gate_loss = F.binary_cross_entropy_with_logits(
                    preds[gate_key], (targets[gated_name] > 0).float()
                )
                combined = combined + self.gate_weight * gate_loss
        return combined

    def _compute_loss_components_capturable(self, preds: dict, targets: dict) -> tuple:
        """``_compute_loss_components`` with branch-free hurdle dispatch.

        Same (combined, tensor-components) contract as the branchy original —
        the graphed VAL pass needs the per-component tensors for the epoch
        accumulators, unlike the train graph which only needs ``combined``
        (``compute_combined_capturable``, untouched so the train capture
        stays bitwise-identical). Per-head dispatch is config-static; the
        hurdle masked-sum forms are the same algebraic identities used by
        the train-side capturable loss.
        """
        per_target_losses = {}
        combined = torch.zeros((), device=next(iter(preds.values())).device, dtype=torch.float32)
        for name in self.target_names:
            lt = self.head_losses[name]
            if lt == "hurdle_negbin":
                loss = hurdle_negbin_value_loss_capturable(preds, targets, name)
            elif lt == "hurdle_poisson":
                loss = hurdle_poisson_value_loss_capturable(preds, targets, name)
            else:
                loss = self.loss_fns[name](preds[name], targets[name])
            per_target_losses[name] = loss
            combined = combined + self.loss_weights[name] * loss

        components = {f"loss_{name}": loss for name, loss in per_target_losses.items()}

        for gated_name in self.gated_targets:
            gate_key = f"{gated_name}_gate_logit"
            if gate_key in preds:
                gate_loss = F.binary_cross_entropy_with_logits(
                    preds[gate_key], (targets[gated_name] > 0).float()
                )
                combined = combined + self.gate_weight * gate_loss
                components[f"loss_gate_{gated_name}"] = gate_loss

        components["loss_combined"] = combined
        return combined, components


class _GPUResidentBatcher:
    """Iterates pre-loaded GPU tensors in shuffled mini-batches without DataLoader.

    Yields batches in the same nested-tuple format the corresponding Dataset's
    ``__getitem__`` + default ``collate_fn`` would produce. ``drop_last`` mirrors
    the DataLoader path (``drop_last=True`` on train, ``False`` on val) so epoch
    arithmetic and number-of-batches semantics stay identical to the DataLoader.

    Construction side (``make_*_dataloaders``) is responsible for moving feature
    tensors to ``device`` once and passing them in the order the Dataset's
    ``__getitem__`` produces. The y_dict tensors are likewise on-device. The
    iterator generates a permutation per epoch via ``torch.randperm`` on the
    same device (using the global RNG state, matching how PyTorch's default
    sampler seeds itself when ``generator=None``); each batch slices the
    pre-loaded tensors with ``index_select`` — a near-pure GPU op that
    eliminates the DataLoader's worker IPC, pinned-memory H2D copy, and
    per-sample ``__getitem__`` Python overhead.
    """

    def __init__(
        self,
        feature_tensors: tuple,
        y_dict: dict,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ):
        if not feature_tensors:
            raise ValueError("_GPUResidentBatcher requires at least one feature tensor")
        self._features = feature_tensors  # tuple of on-device tensors, all same N
        self._y_dict = y_dict  # dict[str, on-device tensor], all same N
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        n = feature_tensors[0].shape[0]
        self._n = n
        self._device = feature_tensors[0].device
        # Sanity: every feature tensor and every y tensor must share N. This is
        # enforced by the corresponding ``MultiTarget*Dataset`` for the CPU path
        # via list-indexing semantics; surface it explicitly here so a caller
        # mismatch fails fast instead of producing silently mis-aligned batches.
        for t in feature_tensors[1:]:
            if t.shape[0] != n:
                raise ValueError(
                    f"_GPUResidentBatcher feature tensors must share N: got {t.shape[0]} vs {n}"
                )
        for k, v in y_dict.items():
            if v.shape[0] != n:
                raise ValueError(f"_GPUResidentBatcher target '{k}' shape {v.shape[0]} != N={n}")

    def __len__(self) -> int:
        """Number of batches yielded per pass — mirrors ``DataLoader.__len__``."""
        if self._drop_last:
            return self._n // self._batch_size
        return (self._n + self._batch_size - 1) // self._batch_size

    @property
    def features(self) -> tuple:
        """The resident on-device feature tensors, in model-positional order."""
        return self._features

    @property
    def y_dict(self) -> dict:
        """The resident on-device target tensors."""
        return self._y_dict

    def index_batches(self):
        """Yield per-step index tensors instead of sliced batches.

        The full-step CUDA-graph path (``_GraphedTrainStep``) does its own
        ``index_select`` *inside* the captured graph, so the loop only needs
        the indices. Consumes the CPU global RNG identically to ``__iter__``
        (one ``torch.randperm`` per pass) so both paths see the same
        per-epoch batch order at the same seed.
        """
        if self._n == 0:
            return
        if self._shuffle:
            perm = torch.randperm(self._n).to(self._device)
        else:
            perm = torch.arange(self._n, device=self._device)
        last = (self._n // self._batch_size) * self._batch_size if self._drop_last else self._n
        for i in range(0, last, self._batch_size):
            yield perm[i : i + self._batch_size]

    def __iter__(self):
        if self._n == 0:
            return
        if self._shuffle:
            # Draw the permutation from the CPU global RNG (``torch.default_generator``),
            # then H2D copy to the batcher's device. CPU and CUDA have *independent*
            # default generators in PyTorch — ``torch.manual_seed(N)`` seeds both, but
            # ``torch.cuda.manual_seed_all(N)`` seeds a separate stream that produces
            # a different sequence at the same seed. The old ``DataLoader(shuffle=True)``
            # path that this class replaces (and the CPU/MPS fallback path below in
            # ``make_*_dataloaders``) consumes the CPU stream via ``RandomSampler``;
            # generating the permutation on ``device='cuda'`` here would silently
            # diverge from that, giving a different per-epoch batch order at the same
            # global seed and shifting the SGD trajectory away from the CPU/MPS path.
            #
            # The H2D copy is N int64s (~160 KB for N≈20k) per epoch — negligible vs
            # the ~100s of NN forward/backward per epoch, and orders of magnitude
            # smaller than the DataLoader IPC + H2D copies PR #309 removed. Not
            # bit-identical to ``RandomSampler`` (which constructs a transient
            # ``torch.Generator`` per ``__iter__`` and consumes one int64 from the
            # global RNG as its seed), but the contract — "draw from CPU RNG so the
            # same global seed maps to the same batch order across CPU/MPS/CUDA hosts
            # that share this code path" — now holds.
            perm = torch.randperm(self._n).to(self._device)
        else:
            perm = torch.arange(self._n, device=self._device)
        last = (self._n // self._batch_size) * self._batch_size if self._drop_last else self._n
        for i in range(0, last, self._batch_size):
            idx = perm[i : i + self._batch_size]
            sliced_features = tuple(t.index_select(0, idx) for t in self._features)
            sliced_y = {k: v.index_select(0, idx) for k, v in self._y_dict.items()}
            # Yield in the same nested shape the existing Dataset + default
            # collate path produces (positional features..., target dict).
            yield (*sliced_features, sliced_y)


class _GraphedTrainStep(nn.Module):
    """Batch gather + model forward + combined loss as ONE graphable callable.

    ``make_graphed_callables`` captures this module's forward+backward, so the
    train step's eager remainder shrinks to: the idx slice handoff, one graph
    replay, and the GradScaler/clip/optimizer tail. The wrapped model module
    itself stays eager — validation, prediction, and ``state_dict`` are
    untouched (unlike the model-only capture, which patches
    ``model.forward``).

    The resident feature/target tensors are held by reference (plain attrs,
    NOT registered buffers — they are the batcher's storage, not state to
    serialize) and must keep stable addresses for the capture's lifetime;
    ``_GPUResidentBatcher`` never reallocates them after construction. The
    gather runs on whatever index tensor is copied into the graph's static
    input each replay, so shuffled epochs work unchanged.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, features: tuple, y_dict: dict):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self._features = features
        self._y_dict = y_dict

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        feats = tuple(t.index_select(0, idx) for t in self._features)
        y = {k: v.index_select(0, idx) for k, v in self._y_dict.items()}
        preds = self.model(*feats)
        return self.criterion.compute_combined_capturable(preds, y)


class _GraphedValPass:
    """One CUDA graph over ALL K full-size validation batches, replayed once
    per epoch.

    The val batcher is ``shuffle=False, drop_last=False``, so batch i is
    always rows ``[i*bs, (i+1)*bs)`` of the resident tensors — the gather is
    baked as ``narrow`` views (zero gather kernels, no per-replay inputs).
    Inside the capture: zero the FP32 accumulators, then per batch
    eval-forward + capturable loss components + accumulate + write preds into
    preallocated FP32 buffers (``narrow().copy_()``, D2D). The 0–1 ragged
    tail batch stays on the eager body via :meth:`tail_batches`.

    Pointer semantics: the graph bakes buffer/param ADDRESSES, so replays read
    each epoch's updated weights and BatchNorm running stats in place (nothing
    reallocates those storages mid-training; the early-stop best-state restore
    is an in-place ``load_state_dict`` after the loop). Eval-mode kernels are
    baked at capture: dropout is identity and BN reads (never updates) its
    running stats, so warmup mutates nothing and no BN snapshot is needed.
    """

    def __init__(self, model, criterion, val_loader, target_names, device):
        self._model = model
        self._criterion = criterion
        self._feats = val_loader.features
        self._y = val_loader.y_dict
        self._target_names = list(target_names)
        self._device = device
        self._bs = val_loader._batch_size
        n = self._feats[0].shape[0]
        self.k = n // self._bs
        self._n_fixed = self.k * self._bs
        self._rem = n - self._n_fixed
        self.loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        self.comp_sums: dict[str, torch.Tensor] = {}
        self.pred_bufs = {
            t: torch.zeros(self._n_fixed, device=device, dtype=torch.float32)
            for t in self._target_names
        }
        # Static read-only views; row order matches pred_bufs so the epoch-end
        # MAE cat sees aligned (pred, target) pairs.
        self.target_prefix = {t: self._y[t].narrow(0, 0, self._n_fixed) for t in self._target_names}
        self._graph: torch.cuda.CUDAGraph | None = None

    def _run_body(self) -> None:
        self.loss_sum.zero_()
        for acc in self.comp_sums.values():
            acc.zero_()
        for i in range(self.k):
            offs = i * self._bs
            feats = tuple(t.narrow(0, offs, self._bs) for t in self._feats)
            y_batch = {k: v.narrow(0, offs, self._bs) for k, v in self._y.items()}
            preds = self._model(*feats)
            combined, comps = self._criterion._compute_loss_components_capturable(preds, y_batch)
            self.loss_sum += combined.float()
            for key, val in comps.items():
                self.comp_sums[key] += val.float()
            for t in self._target_names:
                self.pred_bufs[t].narrow(0, offs, self._bs).copy_(preds[t].float())

    def build(self, autocast_factory) -> None:
        """Discover component keys eagerly, warm up on a side stream, capture.

        ``autocast_factory`` is a zero-arg callable returning the capture
        autocast context (``cache_enabled=False`` — weight casts must happen
        inside the graph so replays re-read the live FP32 params).
        """
        with torch.no_grad(), autocast_factory():
            f0 = tuple(t.narrow(0, 0, self._bs) for t in self._feats)
            y0 = {k: v.narrow(0, 0, self._bs) for k, v in self._y.items()}
            _, comps = self._criterion._compute_loss_components_capturable(self._model(*f0), y0)
        self.comp_sums = {
            key: torch.zeros((), device=self._device, dtype=torch.float32) for key in comps
        }
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                with torch.no_grad(), autocast_factory():
                    self._run_body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), autocast_factory(), torch.cuda.graph(graph):
            self._run_body()
        self._graph = graph

    def replay(self) -> None:
        self._graph.replay()

    def tail_batches(self):
        """Yield the 0-1 leftover ragged batch in the eager body's tuple shape."""
        if self._rem == 0:
            return
        offs = self._n_fixed
        feats = tuple(t.narrow(0, offs, self._rem) for t in self._feats)
        y_batch = {k: v.narrow(0, offs, self._rem) for k, v in self._y.items()}
        yield (*feats, y_batch)


class _GraphedFullStep:
    """The WHOLE train iteration — gather + forward + backward + combined-loss
    AND the per-step tail (zero_grad / clip / AdamW step / loss copy) — captured
    into ONE manual ``torch.cuda.CUDAGraph`` and replayed per step (Lever A3).

    A2 (``_maybe_graph_full_step``) graphs only {gather+fwd+bwd+loss} via
    ``make_graphed_callables`` and leaves the optimizer tail eager (8.6% RB /
    12.7% WR / 24.2% QB of attn-NN step time). ``make_graphed_callables`` can't
    include the optimizer, and its output callable CANNOT be re-captured inside
    an outer graph ("Cannot prepare for replay during capturing stage"), so A3
    re-captures the step EAGERLY: it runs a fresh, un-graphed
    :class:`_GraphedTrainStep` (the identical gather+fwd+loss ops) plus an eager
    ``loss.backward()`` + clip + ``optimizer.step()`` inside one manual capture,
    in the ``_GraphedValPass`` style (static I/O buffers, side-stream warmup).
    The A2 ``make_graphed_callables`` build still runs first — A3 reuses it ONLY
    to reproduce A2-only's BatchNorm warmup perturbation (below), then captures
    its own graph and the train loop replays A3 instead of A2's ``_graphed_step``.

    Unlocked by the FP32 production default (#1311): with no ``GradScaler`` the
    optimizer has no data-dependent inf/NaN skip branch, so the step is
    capture-safe and the trajectory is **bit-identical to A2-only** (validated
    Δ=0 micro-test + the end-to-end gate). Two mechanisms hold that:

    - **LR refresh:** ``AdamW(capturable=True)`` keeps each param group's ``lr``
      as a device tensor the graph bakes by address. A per-epoch
      ``scheduler.step()`` (cosine: constant within an epoch) may rebind
      ``param_group['lr']`` to a NEW tensor → replays would read a STALE baked
      LR. :meth:`refresh_lr_from_scheduler` writes the scheduler's fresh value
      INTO the baked tensor in place and restores the binding.
    - **Warmup snapshot/reset:** the priming step, side-stream warmup, and
      capture each run REAL optimizer steps, mutating weights, BN running stats,
      and Adam moments. :meth:`build` snapshots params + BN at the point A2's
      build left them (A2-only's step-0 state: initial params, warmup-perturbed
      BN), then after capture restores params + BN and resets the optimizer
      moments to the pristine step-0 values — so the first real replay
      reproduces A2-only's first real step exactly.
    """

    def __init__(self, eager_step, model, optimizer, batch_size, device, autocast_factory):
        # ``eager_step`` is a fresh, UN-graphed _GraphedTrainStep (plain ops).
        self._eager_step = eager_step
        self._model = model
        self.optimizer = optimizer
        self._device = device
        self._autocast_factory = autocast_factory
        # Stable param list (gradient-bearing only) reused for clip + the
        # snapshot/reset; materialized ONCE so addresses stay constant for the
        # capture's lifetime.
        self._params = [p for p in model.parameters() if p.requires_grad]
        # Static graph I/O: the per-replay index input and the loss output.
        self._idx_static = torch.empty(batch_size, dtype=torch.long, device=device)
        self._loss_static = torch.zeros((), dtype=torch.float32, device=device)
        # The baked LR tensors, discovered post-capture (capturable AdamW stores
        # ``lr`` as a device tensor); the train loop's between-epoch refresh
        # writes into these in place. One per param group (production uses one).
        self._baked_lr_tensors: list[torch.Tensor] = []
        self._graph: torch.cuda.CUDAGraph | None = None

    def _run_body(self) -> None:
        # In-place grad zero (NOT set_to_none): the grad buffers must keep stable
        # addresses across replays. Numerically identical to set_to_none here —
        # backward OVERWRITES (does not accumulate into) the grads each step, so
        # a pre-zeroed buffer and a freshly-allocated one both end identical.
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()
        loss = self._eager_step(self._idx_static)  # gather + fwd + combined loss (eager)
        loss.backward()
        # foreach clip with a capture-safe clamp; ``error_if_nonfinite=False``
        # avoids the host ``.isfinite()`` sync that would abort capture (FP32 has
        # no overflow skip path, so there is nothing to assert on anyway).
        torch.nn.utils.clip_grad_norm_(self._params, max_norm=1.0, error_if_nonfinite=False)
        self.optimizer.step()
        self._loss_static.copy_(loss.detach())

    def build(self) -> None:
        """Snapshot, prime, side-stream warmup, capture, then reset to step 0.

        ``build`` is called right after A2's ``make_graphed_callables`` build, so
        the model is already at A2-only's step-0 state (initial params,
        warmup-perturbed BN). Snapshot that, allocate the capturable AdamW state
        with one priming step, warm up + capture (more real steps), then restore
        params + BN to the snapshot and reset the optimizer moments — so the
        first real replay continues from A2-only's exact step-0 state.
        """
        param_snapshot = [p.detach().clone() for p in self._params]
        bn_snapshot = _snapshot_batchnorm_state(self._model)

        # LR LOAD-BEARING: force each param group's ``lr`` to a DEVICE TENSOR
        # before capture. Fused+capturable AdamW happily reads a Python-float lr
        # and the graph would then bake that VALUE constant (replays stuck on the
        # build-time LR; the cosine schedule would silently no-op → ~1% trajectory
        # fork vs A2-only). A device tensor is read on-device each replay, so
        # refresh_lr_from_scheduler() can update the schedule in place. The value
        # is unchanged (same float), so the captured steps stay bit-identical to
        # the A2-only eager tail at the initial LR (verified 1-epoch Δ=0). FP32
        # dtype matches AdamW's expectation for the capturable lr tensor.
        self._baked_lr_tensors = []
        for g in self.optimizer.param_groups:
            lr = g["lr"]
            lr_t = (
                lr
                if torch.is_tensor(lr)
                else torch.tensor(float(lr), device=self._device, dtype=torch.float32)
            )
            g["lr"] = lr_t
            self._baked_lr_tensors.append(lr_t)

        # Priming step — allocate capturable AdamW's state tensors (step,
        # exp_avg, exp_avg_sq) BEFORE capture, on a valid arange batch (RNG-free,
        # like A2's capture sample_idx).
        self._idx_static.copy_(torch.arange(self._idx_static.numel(), device=self._device))
        with self._autocast_factory():
            self._run_body()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                with self._autocast_factory():
                    self._run_body()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with self._autocast_factory(), torch.cuda.graph(graph):
            self._run_body()
        self._graph = graph

        # Restore params + BN to the pre-build state and reset the optimizer
        # moments to the pristine step-0 values (the state tensors keep their
        # baked addresses; only their VALUES are reset).
        for p, saved in zip(self._params, param_snapshot, strict=True):
            p.detach().copy_(saved)
        _restore_batchnorm_state(bn_snapshot)
        for p in self._params:
            st = self.optimizer.state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                buf = st.get(key)
                if torch.is_tensor(buf):
                    buf.zero_()
            step = st.get("step")
            if torch.is_tensor(step):
                step.zero_()
            elif step is not None:
                st["step"] = 0
        # ``_baked_lr_tensors`` was populated at the top of build() (the lr
        # tensors the graph captured); the train loop refreshes them in place.

    def refresh_lr_from_scheduler(self) -> None:
        """Write the scheduler's current LR into the baked device LR tensors.

        Called after each per-epoch ``scheduler.step()``: the scheduler assigns
        ``param_group['lr']`` a fresh Python float (computed from the float
        ``base_lrs`` it captured at construction), REPLACING the device tensor the
        graph baked. Copy that float INTO the baked tensor in place and restore
        the binding, so the captured graph (which reads the tensor's address each
        replay) uses the new epoch's LR. Param groups and ``_baked_lr_tensors``
        share construction order, so the match is positional (robust whether the
        scheduler left a float or the baked tensor in place).
        """
        if not self._baked_lr_tensors:
            return
        for g, baked in zip(self.optimizer.param_groups, self._baked_lr_tensors, strict=True):
            lr = g["lr"]
            if lr is baked:
                continue  # scheduler did not rebind (e.g. ReduceLROnPlateau, no change)
            baked.copy_(torch.as_tensor(float(lr), device=baked.device, dtype=baked.dtype))
            g["lr"] = baked

    def replay(self, idx: torch.Tensor) -> None:
        self._idx_static.copy_(idx, non_blocking=True)
        self._graph.replay()

    def loss_value(self) -> torch.Tensor:
        return self._loss_static


def _gpu_resident_device(device=None) -> torch.device | None:
    """Return the CUDA device for the GPU-resident batcher, else ``None``.

    Residency is **opt-in by the caller's training device**: the caller passes
    the device its trainer will run on, and the resident path engages only when
    that device is CUDA. ``device=None`` (the default — unit/integration tests,
    and any code that builds a ``device="cpu"`` trainer on a CUDA-visible host)
    returns ``None`` so the plain ``DataLoader`` path is used. Production
    (``src/shared/pipeline.py``) passes the resolved ``_nn_device()``, so on a
    GPU host residency engages exactly as before — but a CPU-device trainer no
    longer receives mismatched GPU-resident loaders (the source of the
    non-deterministic GPU-box ``@integration`` regression-test flakes; see
    ``todo/fixed-archive.md``). ``run_pipeline --device cpu`` / ``FF_DEVICE=cpu``
    resolves ``_nn_device()`` to CPU, so it short-circuits here too.
    """
    if device is not None and getattr(device, "type", None) == "cuda":
        return device
    return None


def _to_gpu_float(arr, device: torch.device) -> torch.Tensor:
    """Contiguous float tensor on ``device`` — matches the ``torch.FloatTensor``
    dtype the ``MultiTarget*Dataset`` classes produce on the DataLoader path."""
    return torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)


def _to_gpu_mask(arr, device: torch.device) -> torch.Tensor:
    """Bool mask tensor on ``device`` — mirrors the Datasets' bool history/kick masks."""
    return torch.from_numpy(np.asarray(arr, dtype=bool)).to(device)


def _ydict_to_gpu(y_dict: dict, device: torch.device) -> dict:
    """Move every target array in ``y_dict`` to ``device`` as a float tensor."""
    return {k: _to_gpu_float(v, device) for k, v in y_dict.items()}


def _resident_loader_pair(train_feats, y_train_dict, val_feats, y_val_dict, batch_size):
    """Train/val :class:`_GPUResidentBatcher` pair shared by every CUDA branch:
    train shuffles + drops the last partial batch; val keeps order and every row."""
    return (
        _GPUResidentBatcher(
            tuple(train_feats), y_train_dict, batch_size=batch_size, shuffle=True, drop_last=True
        ),
        _GPUResidentBatcher(
            tuple(val_feats), y_val_dict, batch_size=batch_size, shuffle=False, drop_last=False
        ),
    )


def _dataloader_pair(train_ds, val_ds, batch_size):
    """Train/val :class:`DataLoader` pair shared by every CPU/MPS branch
    (``pin_memory`` preserved verbatim; a no-op under CPU-only runs)."""
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
    )


class MultiTargetDataset(Dataset):
    """Dataset that returns features + dict of targets."""

    def __init__(self, X: np.ndarray, y_dict: dict):
        self.X = torch.FloatTensor(X)
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {k: v[idx] for k, v in self.targets.items()}
        return x, y


class MultiTargetHistoryDataset(Dataset):
    """Dataset returning static features + fixed-shape padded history + mask + targets.

    History tensors arrive already padded to ``[n_samples, max_seq_len, game_dim]``
    from ``build_game_history_arrays``. Storing them as a single fixed-shape tensor
    (instead of a list of variable-length tensors that the collate function
    re-pads per batch) lets the default PyTorch collate stack samples directly.
    """

    def __init__(
        self,
        X_static: np.ndarray,
        X_history: np.ndarray,
        history_mask: np.ndarray,
        y_dict: dict,
    ):
        """
        Args:
            X_static: [n_samples, static_dim] static feature array
            X_history: [n_samples, max_seq_len, game_dim] zero-padded history
            history_mask: [n_samples, max_seq_len] bool mask (True = real game)
            y_dict: dict of target arrays
        """
        self.X_static = torch.FloatTensor(X_static)
        self.X_history = torch.FloatTensor(X_history)
        self.history_mask = torch.from_numpy(np.asarray(history_mask, dtype=bool))
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X_static)

    def __getitem__(self, idx):
        return (
            self.X_static[idx],
            self.X_history[idx],
            self.history_mask[idx],
            {k: v[idx] for k, v in self.targets.items()},
        )


def make_history_dataloaders(
    X_train_static,
    X_train_history,
    train_history_mask,
    y_train_dict,
    X_val_static,
    X_val_history,
    val_history_mask,
    y_val_dict,
    batch_size=256,
    device=None,
):
    """Create DataLoaders for attention model with game history.

    All history tensors must be pre-padded to a uniform ``[n, max_seq_len, game_dim]``
    shape so the default PyTorch collate can stack samples without per-batch padding.

    On CUDA hosts, ``(X_static, X_history, history_mask, y_dict)`` are moved to
    the GPU once and iterated by :class:`_GPUResidentBatcher` — see
    :func:`make_dataloaders` for the rationale. CPU/MPS hosts retain the
    original DataLoader path bit-for-bit.
    """
    gpu_device = _gpu_resident_device(device)
    if gpu_device is not None:
        # Bool masks match ``MultiTargetHistoryDataset.__init__``'s
        # ``torch.from_numpy(np.asarray(..., dtype=bool))``; the downstream
        # attention forward path expects ``bool``.
        return _resident_loader_pair(
            [
                _to_gpu_float(X_train_static, gpu_device),
                _to_gpu_float(X_train_history, gpu_device),
                _to_gpu_mask(train_history_mask, gpu_device),
            ],
            _ydict_to_gpu(y_train_dict, gpu_device),
            [
                _to_gpu_float(X_val_static, gpu_device),
                _to_gpu_float(X_val_history, gpu_device),
                _to_gpu_mask(val_history_mask, gpu_device),
            ],
            _ydict_to_gpu(y_val_dict, gpu_device),
            batch_size,
        )

    train_ds = MultiTargetHistoryDataset(
        X_train_static, X_train_history, train_history_mask, y_train_dict
    )
    val_ds = MultiTargetHistoryDataset(X_val_static, X_val_history, val_history_mask, y_val_dict)
    return _dataloader_pair(train_ds, val_ds, batch_size)


def make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256, device=None):
    """Create DataLoaders for multi-target training.

    On CUDA hosts the full training+val tensors are moved to the GPU once and
    iterated via :class:`_GPUResidentBatcher` (no DataLoader workers, no H2D
    copy per batch). The largest position fits in a small fraction of T4's
    16 GB; trading host RAM for residency removes the dominant residual
    per-epoch overhead.

    On CPU/MPS hosts the original DataLoader path is preserved bit-for-bit —
    unit tests run there and the byte-identical-convergence guarantee for the
    CPU path is what makes the new CUDA path safe to deploy incrementally.
    ``pin_memory=True`` is a no-op under CPU-only runs; on CUDA it would
    allocate page-locked host tensors but the CUDA branch above bypasses
    DataLoader entirely so the flag is no longer relevant there.
    """
    gpu_device = _gpu_resident_device(device)
    if gpu_device is not None:
        # One-time numpy → tensor → GPU move (see ``_to_gpu_float`` for the
        # dtype-match-the-Dataset rationale).
        return _resident_loader_pair(
            [_to_gpu_float(X_train, gpu_device)],
            _ydict_to_gpu(y_train_dict, gpu_device),
            [_to_gpu_float(X_val, gpu_device)],
            _ydict_to_gpu(y_val_dict, gpu_device),
            batch_size,
        )

    train_ds = MultiTargetDataset(X_train, y_train_dict)
    val_ds = MultiTargetDataset(X_val, y_val_dict)
    return _dataloader_pair(train_ds, val_ds, batch_size)


class MultiHeadTrainer:
    """Training loop for any multi-head position network.

    Subclass and override _forward_batch() to support different input formats
    (e.g., attention models with game history).
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        target_names,
        patience=15,
        scheduler_per_batch=False,
        log_every=10,
        epoch_callback=None,
        use_amp=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.target_names = target_names
        self.patience = patience
        self.scheduler_per_batch = scheduler_per_batch
        self.log_every = log_every
        # Optional ``fn(epoch: int, avg_val_loss: float) -> None`` invoked once
        # per epoch right after ``avg_val_loss`` is computed. Used by
        # ``src/tuning/tune_nn.py`` to feed Optuna's pruner the trial's val
        # trajectory; if the callback raises (e.g. ``optuna.TrialPruned``) the
        # exception propagates and stops training. Default ``None`` keeps the
        # base behaviour for every other caller.
        self.epoch_callback = epoch_callback
        self.best_val_metric = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        # AMP dtype on the forward + loss path, chosen by ``amp_dtype()``
        # (src/shared/utils.py) — one source of truth for every host:
        #
        #   - CUDA, default: float16 + GradScaler. FP16 is the proven default on
        #     ALL CUDA (T4 sm_75 AND Blackwell sm_120). T4 has FP16 Tensor Cores
        #     (~65 TFLOPS vs ~8 FP32); on the 5080 a deterministic A/B showed
        #     BF16 *regresses* high-magnitude heads (QB passing_yards +2.2-3.1%)
        #     and FP16 runs full-throughput there too, so FP16 wins on both.
        #   - CUDA, FF_AMP_DTYPE=bf16 (opt-in, sm_80+ only): bfloat16, no
        #     GradScaler (BF16 keeps the FP32 exponent range). amp_dtype()
        #     refuses BF16 on T4 (degrades to FP16) so the opt-in can't
        #     reintroduce the #293/#301 T4 hang.
        #   - Non-CUDA (CPU/MPS — local Mac dev, CI) or FF_AMP_DTYPE=fp32:
        #     amp_dtype() is None → AMP off, byte-identical to the FP32 path.
        #
        # GradScaler handles FP16 gradient underflow (dynamic loss scale,
        # auto-adjusting on inf/NaN); it is enabled ONLY for the float16 path.
        # For the BF16 opt-in and the no-AMP path it is constructed with
        # enabled=False so the .scale/.unscale_/.step/.update calls in the loop
        # are no-ops and a single code path covers all three dtypes. The
        # optimizer step and parameter master copies stay in FP32 in every case.
        self._amp_dtype = (
            amp_dtype() if (bool(use_amp) and getattr(device, "type", None) == "cuda") else None
        )
        self._use_amp = self._amp_dtype is not None
        self._fixed_amp_scale = _env_truthy(_FIXED_SCALE_ENV)
        scaler_kwargs = {}
        if self._fixed_amp_scale:
            scaler_kwargs = {
                "init_scale": _env_float(_INIT_SCALE_ENV, 65536.0),
                # Keep the normal initial scale but prevent growth. Any overflow
                # is treated as an invalid diagnostic run below rather than
                # silently changing the scale schedule.
                "growth_interval": 2**31 - 1,
            }
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._use_amp and self._amp_dtype is torch.float16,
            **scaler_kwargs,
        )
        self._scaler_trace_path = os.environ.get(_GRADSCALER_TRACE_PATH_ENV, "").strip()
        self._scaler_trace_label = os.environ.get(_GRADSCALER_TRACE_LABEL_ENV, "").strip()
        self._scaler_trace_fh = None
        # Flipped True once make_graphed_callables has wrapped self.model
        # (FF_CUDA_GRAPH path); guards _maybe_graph_model against re-capturing.
        self._graphed = False
        # Set to the captured _GraphedTrainStep when FULL-STEP capture engages
        # (FF_CUDA_GRAPH_FULL path); the train loop then iterates index
        # batches and the model-only capture is skipped.
        self._graphed_step = None
        # Graphed VAL pass (same FF_CUDA_GRAPH_FULL regime): built lazily at
        # the first val epoch when the train-side full-step capture engaged;
        # a build failure flips the _failed latch -> permanent eager val.
        self._graphed_val = None
        self._graphed_val_failed = False
        # Set to the captured _GraphedFullStep when OPTIMIZER-TAIL capture
        # engages (Lever A3, FF_CUDA_GRAPH_OPT path); the train loop then
        # replays the whole iteration (including the optimizer step) and skips
        # the eager tail. A build failure flips _graphed_opt_failed and keeps
        # the A2 eager tail.
        self._graphed_opt = None
        self._graphed_opt_failed = False

    def _autocast(self):
        """Return the autocast context when AMP is active, else nullcontext.

        Wrapped around forward + criterion so matmul / linear / attention ops
        downcast to the selected ``self._amp_dtype`` (BF16 on sm_80+, FP16 on
        T4); the optimizer step and gradient update happen outside the context
        so master copies stay in FP32 and GradScaler (FP16 only) can rescale
        gradients between backward() and step().
        """
        if self._use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _open_scaler_trace(self) -> None:
        if not self._scaler_trace_path:
            return
        parent = os.path.dirname(self._scaler_trace_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._scaler_trace_fh = open(  # noqa: SIM115 - kept open across the batch loop
            self._scaler_trace_path,
            "w",
            encoding="utf-8",
            buffering=1,
        )
        self._write_scaler_trace(
            {
                "kind": "meta",
                "label": self._scaler_trace_label,
                "amp_dtype": str(self._amp_dtype),
                "enabled": bool(self._scaler.is_enabled()),
                "fixed_scale": bool(self._fixed_amp_scale),
                "initial_scale": (
                    float(self._scaler.get_scale()) if self._scaler.is_enabled() else None
                ),
                "graphed": bool(self._graphed),
            }
        )

    def _close_scaler_trace(self) -> None:
        if self._scaler_trace_fh is None:
            return
        self._scaler_trace_fh.close()
        self._scaler_trace_fh = None

    def _write_scaler_trace(self, row: dict) -> None:
        if self._scaler_trace_fh is None:
            return
        self._scaler_trace_fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _forward_batch(self, batch) -> tuple[dict, dict]:
        """Unpack a DataLoader batch, move to device, and run the forward pass.

        Returns:
            (preds_dict, targets_dict) — both on device.
        """
        X_batch, y_batch = batch
        X_batch = X_batch.to(self.device, non_blocking=True)
        y_batch = {k: v.to(self.device, non_blocking=True) for k, v in y_batch.items()}
        preds = self.model(X_batch)
        return preds, y_batch

    def _graph_inputs(self, batch) -> tuple:
        """Device-resident positional tensor args for ``self.model(...)``.

        Mirrors the model-call side of :meth:`_forward_batch`; used only as
        ``sample_args`` for ``make_graphed_callables`` (:meth:`_maybe_graph_model`).
        Keep in lockstep with ``_forward_batch`` — the CUDA-graph A/B's Part-1
        gate (``torch.equal`` on a fixed batch) catches any drift. Subclasses with
        extra history tensors override this.
        """
        X_batch, _ = batch
        return (X_batch.to(self.device, non_blocking=True),)

    def _maybe_graph_val(self, val_loader) -> None:
        """Lazily capture the val pass (same FF_CUDA_GRAPH_FULL regime).

        Engages only when the train-side full-step capture engaged (which
        already implies the knob, CUDA, sm_80+, and a resident train loader —
        and the nested-K trainer never reaches here since its
        ``_maybe_graph_full_step`` no-ops), the val loader is a resident
        in-order batcher with at least one full-size batch, and no prior
        build failed. Called from the val section AFTER ``model.eval()`` so
        eval-mode kernels are what gets captured. Failure logs and latches
        permanent eager fallback — never fails the trial.
        """
        if (
            self._graphed_val is not None
            or self._graphed_val_failed
            or self._graphed_step is None
            or not isinstance(val_loader, _GPUResidentBatcher)
            or val_loader._shuffle
            or val_loader._drop_last
            or val_loader._n < val_loader._batch_size
        ):
            return
        try:
            gv = _GraphedValPass(
                self.model, self.criterion, val_loader, self.target_names, self.device
            )
            gv.build(
                lambda: (
                    torch.amp.autocast(
                        device_type="cuda", dtype=self._amp_dtype, cache_enabled=False
                    )
                    if self._use_amp
                    else contextlib.nullcontext()
                )
            )
        except Exception as e:
            print(
                f"[cuda-graph] val capture failed ({e!r}); keeping the eager val pass",
                flush=True,
            )
            torch.cuda.synchronize()
            self._graphed_val_failed = True
            return
        self._graphed_val = gv

    def _maybe_graph_full_step(self, train_loader) -> bool:
        """Autodetect-ON FULL-STEP capture for CUDA sm_80+ (``FF_CUDA_GRAPH_FULL``
        is a force-off override on the base sm_80+ gate): one graph covering
        batch gather + model forward + combined loss via :class:`_GraphedTrainStep`.

        Strictly wider capture than :meth:`_maybe_graph_model` — the eager
        per-step remainder drops from {gather, forward-replay, criterion
        (~dozens of launches), backward-replay} to {idx handoff, one
        fwd+bwd replay} plus the shared GradScaler/clip/optimizer tail.
        Returns True when engaged (sets ``self._graphed_step``); the train
        loop then iterates ``index_batches()``, and the model-only capture
        MUST be skipped (capturing a wrapper around an already-graphed model
        would nest replays inside a capture — invalid).

        Engagement requirements, all config-static:
        - ``cuda_graph_full_enabled()`` and a CUDA-device trainer;
        - a ``_GPUResidentBatcher`` train loader with ``drop_last`` (static
          shapes; the resident tensors give the gather stable storage);
        - the attention entropy regulariser dormant (``attn_entropy_coeff``
          0/absent) — its side-channel buffer read is not validated under
          this capture;
        - capture itself succeeding — any failure logs and falls back to the
          model-only path, so a position that can't capture still trains.
        """
        if (
            self._graphed
            or self._graphed_step is not None
            or self.device.type != "cuda"
            or not cuda_graph_full_enabled()
            or not isinstance(train_loader, _GPUResidentBatcher)
            or not train_loader._drop_last
            or getattr(self.model, "attn_entropy_coeff", 0.0)
        ):
            return False
        step = _GraphedTrainStep(
            self.model, self.criterion, train_loader.features, train_loader.y_dict
        )
        step.train()
        # arange, not a sampled permutation slice: valid indices with the
        # static train batch shape, and it consumes NO RNG (the model-only
        # capture's ``next(iter(train_loader))`` draws a permutation; this
        # path keeps capture RNG-free so epoch batch order matches the eager
        # iterator at the same seed).
        sample_idx = torch.arange(train_loader._batch_size, device=self.device)
        capture_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype, cache_enabled=False)
            if self._use_amp
            else contextlib.nullcontext()
        )
        bn_snapshot = (
            _snapshot_batchnorm_state(self.model)
            if _env_truthy(_CUDA_GRAPH_RESTORE_BN_ENV)
            else None
        )
        try:
            with capture_ctx:
                graphed = torch.cuda.make_graphed_callables(
                    step,
                    (sample_idx,),
                    num_warmup_iters=3,
                    allow_unused_input=True,
                )
        except Exception as e:
            print(
                f"[cuda-graph] full-step capture failed ({e!r}); "
                "falling back to model-only capture",
                flush=True,
            )
            return False
        finally:
            if bn_snapshot is not None:
                _restore_batchnorm_state(bn_snapshot)
        self._graphed_step = graphed
        self._graphed = True
        return True

    def _maybe_graph_full_opt(self, train_loader) -> bool:
        """Lever A3: extend A2's full-step capture to bake the optimizer tail.

        Engages only ON TOP of A2 (``self._graphed_step`` set) and captures the
        whole iteration — gather + forward + backward + combined loss + zero_grad
        + clip + ``AdamW.step`` + loss copy — into one manual
        :class:`_GraphedFullStep` graph, so the eager per-step tail disappears.
        Returns True when engaged (sets ``self._graphed_opt``); the train loop
        then replays it instead of calling A2's ``_graphed_step``.

        STRICTLY INERT (bit-identical to A2-only; no rebaseline). The FP32
        production default removed ``GradScaler`` (no inf/NaN skip branch), so
        the requirements force the only regime where that holds:
        - A2 engaged (``_graphed_step`` set) on a CUDA-device trainer;
        - ``cuda_graph_opt_enabled()`` (A3 ⊆ A2 ⊆ base gate);
        - a per-EPOCH scheduler (``not scheduler_per_batch``) — OneCycleLR
          mutates the LR every step, which the baked single-LR-tensor graph
          can't track (the ``capturable`` wiring in ``_run_nn_training`` also
          disables it there, this is the defensive mirror);
        - AMP off (``not self._use_amp``) and the GradScaler disabled / not
          fixed-scale / not traced — A3 only covers the no-skip FP32 path;
        - the optimizer fused + capturable (so its step is graph-safe; a
          defensive precondition — ``_run_nn_training`` sets these together).
        Any capture failure logs, synchronizes, latches ``_graphed_opt_failed``,
        and falls back to A2's eager tail — never fails the run.
        """
        if (
            self._graphed_opt is not None
            or self._graphed_opt_failed
            or self._graphed_step is None
            or self.device.type != "cuda"
            or not cuda_graph_opt_enabled()
            or self.scheduler_per_batch
            or self._use_amp
            or self._scaler.is_enabled()
            or self._fixed_amp_scale
            or bool(self._scaler_trace_path)
            or not isinstance(train_loader, _GPUResidentBatcher)
            or not train_loader._drop_last
        ):
            return False
        # Defensive: the capturable AdamW step is what makes the optimizer
        # graph-safe (on-device step counter + LR). ``_run_nn_training`` sets
        # fused+capturable together under the same A3 gate; bail rather than
        # capture a non-capturable optimizer.
        if not _optimizer_is_fused_capturable(self.optimizer):
            return False
        try:
            eager_step = _GraphedTrainStep(
                self.model, self.criterion, train_loader.features, train_loader.y_dict
            )
            eager_step.train()
            gfs = _GraphedFullStep(
                eager_step,
                self.model,
                self.optimizer,
                train_loader._batch_size,
                self.device,
                # A3 is FP32-only (gated on ``not self._use_amp``), so the
                # capture autocast is a nullcontext — mirrors the AMP-off branch
                # of _GraphedValPass.build's factory.
                contextlib.nullcontext,
            )
            gfs.build()
        except Exception as e:
            print(
                f"[cuda-graph] optimizer-tail capture failed ({e!r}); keeping the A2 eager tail",
                flush=True,
            )
            torch.cuda.synchronize()
            self._graphed_opt_failed = True
            return False
        print(
            "[cuda-graph] optimizer-tail capture engaged (Lever A3); "
            "the per-step tail is now graphed",
            flush=True,
        )
        self._graphed_opt = gfs
        return True

    def _maybe_graph_model(self, train_loader) -> None:
        """Autodetect-ON for CUDA sm_80+ (``FF_CUDA_GRAPH`` is the force-off
        override): replace ``self.model`` with a CUDA graph capture of its
        forward+backward, replayed each train step.

        The tiny attention model is GPU-launch-bound; capturing collapses its
        per-step kernel launches into one replay. ``make_graphed_callables``
        patches ``model.forward`` in place (so ``.train()`` / ``.eval()`` /
        ``.parameters()`` keep working) and routes ``model.eval()`` back to the
        original eager forward — so the un-graphed, ragged-last-batch validation
        pass is unaffected and only the fixed-shape (``drop_last=True``) train
        forward is graphed. GradScaler / optimizer step stay OUTSIDE the graph
        (they have data-dependent inf/NaN control flow); only fwd+bwd are captured.

        Inert by construction — replay runs the same eager kernels, no Inductor
        fusion. With dropout live the replay's RNG stream diverges from the
        un-graphed run (capture warmup iters + offset stepping), so flag-on MAE
        differs from eager by seed-noise, not 0 — see the two-part A/B gate in
        todo/gpu_launch_bound_levers.md.

        NOTE: if ``attn_entropy_coeff`` is ever set > 0, ``last_attn_entropy``
        becomes a static graph buffer written at capture; the entropy term (read
        outside the forward in :meth:`train`) would then need re-validation that it
        sees post-replay state. It is 0 / off in every current config.
        """
        # Gate on THIS trainer's device, not just the global capability.
        # cuda_graph_enabled() reports the *host* is sm_80+ CUDA, but a trainer
        # constructed with device=cpu (FF_DEVICE=cpu runs, and several unit tests)
        # on a GPU host must NOT be graphed: torch.cuda.make_graphed_callables on
        # a CPU model captures garbage → NaN MAE / non-decreasing loss. Mirrors
        # the device.type=="cuda" AMP gate in __init__ and the _cuda guard in
        # train(); FF_CUDA_GRAPH=1 still can't force capture on a CPU trainer.
        if self._graphed or self.device.type != "cuda" or not cuda_graph_enabled():
            return
        sample_args = self._graph_inputs(next(iter(train_loader)))
        self.model.train()
        # make_graphed_callables requires autocast caching disabled; the optimizer
        # step/GradScaler stay outside the captured fwd+bwd region.
        capture_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype, cache_enabled=False)
            if self._use_amp
            else contextlib.nullcontext()
        )
        bn_snapshot = (
            _snapshot_batchnorm_state(self.model)
            if _env_truthy(_CUDA_GRAPH_RESTORE_BN_ENV)
            else None
        )
        try:
            with capture_ctx:
                # Patches model.forward in place and returns the same module.
                # allow_unused_input=True: per-position gated/plain head mixes leave
                # some params unused for a given graph's outputs.
                self.model = torch.cuda.make_graphed_callables(
                    self.model,
                    sample_args,
                    num_warmup_iters=3,
                    allow_unused_input=True,
                )
        finally:
            if bn_snapshot is not None:
                _restore_batchnorm_state(bn_snapshot)
        self._graphed = True

    def train(self, train_loader, val_loader, n_epochs) -> dict:
        # CUDA graph capture, widest applicable scope first: autodetect-ON
        # full-step (gather+fwd+loss; FF_CUDA_GRAPH_FULL=0 forces eager)
        # subsumes the autodetect-ON model-only capture (FF_CUDA_GRAPH=0 forces
        # eager); no-op otherwise.
        if self._maybe_graph_full_step(train_loader):
            # A2 engaged: try to also capture the optimizer tail (Lever A3).
            # Inert on the FP32 path; falls back to A2's eager tail otherwise.
            self._maybe_graph_full_opt(train_loader)
        else:
            self._maybe_graph_model(train_loader)
        # FF_NN_FIXED_EPOCHS=<N> (test-only): train exactly N epochs, never
        # early-stop, keep last-epoch weights (skip best-val restore). Makes
        # model selection independent of the (graph-perturbed) val curve, so a
        # CUDA-graph A/B isolates trajectory divergence from best-epoch-selection
        # drift. Unset / non-positive → normal early-stopping behaviour.
        _fixed_raw = os.environ.get("FF_NN_FIXED_EPOCHS", "").strip()
        _fixed_n = int(_fixed_raw) if _fixed_raw.isdigit() and int(_fixed_raw) > 0 else 0
        _fixed_epochs = _fixed_n > 0
        if _fixed_epochs:
            n_epochs = _fixed_n
        history = {
            k: []
            for k in [
                "train_loss",
                "val_loss",
                "epoch_sec",
                "peak_mem_gb",
                *[f"val_loss_{t}" for t in self.target_names],
                *[f"val_mae_{t}" for t in self.target_names],
            ]
        }
        # Weighted MAE used for early stopping mirrors the training loss's
        # per-target weighting so high-scale targets (yards) don't dominate
        # the selection criterion.
        loss_weights = getattr(self.criterion, "loss_weights", None) or {}
        weight_sum = sum(loss_weights.get(t, 1.0) for t in self.target_names) or 1.0
        _cuda = torch.cuda.is_available() and self.device.type == "cuda"
        # Hoisted out of the batch loop: the model's optional attention-entropy
        # method is a fixed attribute; only the *call* (which reads the latest
        # attention weights) needs to happen per batch.
        entropy_fn = getattr(self.model, "attention_entropy_loss", None)
        trace_scaler = bool(self._scaler_trace_path)
        need_scale_state = self.scheduler_per_batch or trace_scaler or self._fixed_amp_scale
        global_step = 0
        self._open_scaler_trace()

        for epoch in range(n_epochs):
            if _cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            _epoch_t0 = time.perf_counter()
            # --- Training pass ---
            self.model.train()
            # FP32 GPU-resident accumulator avoids a per-batch
            # cudaStreamSynchronize from ``loss.item()``. We cast to FP32 on
            # accumulation (PR #301 made ``loss`` FP16 under AMP) to preserve
            # precision over many batches. Single sync at end of epoch via
            # ``.item()`` below.
            epoch_train_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            n_train_batches = 0

            # Full-step graph path iterates bare index tensors (the gather
            # happens inside the captured graph); eager / model-only-graph
            # paths iterate sliced batches. Same RNG consumption either way.
            # A3 (optimizer-tail graph) replays the WHOLE iteration including the
            # optimizer step, so it iterates bare index tensors like A2 but skips
            # the eager tail entirely. A2 (full-step) iterates index tensors and
            # runs the eager tail; the model-only / eager paths iterate sliced
            # batches. Same RNG consumption (one randperm/pass) in every case.
            opt_step = self._graphed_opt
            full_step = self._graphed_step
            batch_iter = (
                train_loader.index_batches()
                if (opt_step is not None or full_step is not None)
                else train_loader
            )
            for batch in batch_iter:
                if opt_step is not None:
                    # ``batch`` is the bare idx from index_batches(); the replay
                    # runs zero_grad/fwd/bwd/clip/AdamW.step/loss-copy as one
                    # graph. epoch_train_loss reads the static FP32 loss buffer
                    # (on-GPU; .item() deferred to the epoch-end sync below).
                    opt_step.replay(batch)
                    epoch_train_loss += opt_step.loss_value()
                    n_train_batches += 1
                    global_step += 1
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    if full_step is not None:
                        loss = full_step(batch)
                    else:
                        preds, y_batch = self._forward_batch(batch)
                        # _compute_loss_components, NOT forward(): forward()'s
                        # float components dict costs one .item() GPU sync per
                        # target/gate head (~7-9 syncs per step) and this branch
                        # discards it. The val branch below already calls the
                        # tensor-returning path for the same reason.
                        loss, _ = self.criterion._compute_loss_components(preds, y_batch)
                    # Attention entropy regulariser (additive). ``entropy_fn``
                    # is hoisted above the epoch loop; it returns ``None`` when
                    # the feature is off.
                    if entropy_fn is not None:
                        entropy_term = entropy_fn()
                        if entropy_term is not None:
                            loss = loss + entropy_term
                # Only the FP16 AMP path (T4) uses GradScaler, to keep gradients
                # in representable range. In every other case — CPU / MPS,
                # use_amp=False, AND the BF16 path on sm_80+ (BF16 keeps the FP32
                # exponent range so no scaling is needed) — GradScaler is
                # constructed with enabled=False and these scaler.* methods are
                # pass-through no-ops, so the path is bit-identical to a plain
                # loss.backward() + optimizer.step().
                #
                # scaler.unscale_() must run BEFORE clip_grad_norm_ so the
                # clip threshold (max_norm=1.0) is applied to real-scale
                # gradients, not the loss-scaled magnitudes.
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # GradScaler.step skips ``optimizer.step()`` on inf/NaN grads
                # (and ``update()`` backs the scale off). ``get_scale()`` forces
                # a CPU-GPU sync, so read it ONLY on the per-batch-scheduler path
                # that actually needs the skip signal: advancing a per-batch
                # schedule (OneCycleLR) on a skipped step would drift its
                # warmup/cooldown. A strict scale *decrease* across step+update
                # is GradScaler's "I skipped" signal. The default
                # ``cosine_warm_restarts`` is scheduler_per_batch=False, so
                # step()/update() still run every batch but we pay ZERO
                # get_scale() syncs (was 2/batch here, both discarded). When the
                # scaler is disabled (BF16 / CPU / use_amp=False) get_scale() is
                # a constant 1.0 anyway, so behaviour is identical either way.
                scale_before = self._scaler.get_scale() if need_scale_state else None
                self._scaler.step(self.optimizer)
                self._scaler.update()
                scale_after = self._scaler.get_scale() if need_scale_state else None
                skipped_step = (
                    bool(scale_after < scale_before)
                    if scale_before is not None and scale_after is not None
                    else False
                )
                if trace_scaler:
                    self._write_scaler_trace(
                        {
                            "kind": "step",
                            "label": self._scaler_trace_label,
                            "epoch": epoch,
                            "batch": n_train_batches,
                            "step": global_step,
                            "scale": float(scale_before) if scale_before is not None else None,
                            "next_scale": float(scale_after) if scale_after is not None else None,
                            "skipped": skipped_step,
                            "scale_changed": (
                                bool(scale_after != scale_before)
                                if scale_before is not None and scale_after is not None
                                else False
                            ),
                            "graphed": bool(self._graphed),
                            "fixed_scale": bool(self._fixed_amp_scale),
                        }
                    )
                if self._fixed_amp_scale and skipped_step:
                    self._close_scaler_trace()
                    raise RuntimeError(
                        "FF_AMP_FIXED_SCALE detected an overflow/skip; fixed-scale "
                        "diagnostic run is invalid."
                    )
                if self.scheduler_per_batch and not skipped_step:
                    self.scheduler.step()
                global_step += 1

                # ``loss.detach().float()`` keeps the accumulator on-GPU and
                # in FP32; no host sync per batch (cf. ``loss.item()``).
                epoch_train_loss += loss.detach().float()
                n_train_batches += 1

            # Single end-of-epoch sync (forces accumulator off-GPU). Guard
            # against ``n_train_batches == 0`` — possible on tiny datasets
            # where ``len(train_loader) * batch_size < drop_last_threshold``
            # produces an empty iterator (the GPU-resident batcher's
            # ``drop_last=True`` floors to 0 when ``n < batch_size``). Without
            # the guard, ``0 / 0`` produces NaN, which silently corrupts the
            # history dict and the downstream early-stop comparison.
            avg_train_loss = (
                (epoch_train_loss / n_train_batches).item() if n_train_batches > 0 else 0.0
            )
            history["train_loss"].append(avg_train_loss)

            # --- Validation pass ---
            self.model.eval()
            self._maybe_graph_val(val_loader)
            all_preds = {k: [] for k in self.target_names}
            all_targets = {k: [] for k in self.target_names}
            # FP32 GPU-resident accumulator: see train-pass comment above.
            epoch_val_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            # Per-target tensor accumulators (on-device, FP32) so we can mirror
            # the train branch's "single end-of-epoch sync" pattern. PR #305
            # explicitly left the val per-batch ``.item()`` syncs in place; this
            # commit completes the migration by calling
            # ``MultiTargetLoss._compute_loss_components`` (tensor-valued
            # components) instead of ``forward`` (float-valued).
            val_components_accum: dict[str, torch.Tensor] = {}
            n_val_batches = 0

            with torch.no_grad():
                if self._graphed_val is not None:
                    # One replay covers the K full-size batches; only the
                    # ragged tail (if any) runs the eager body below. The
                    # accumulator/pred-buffer values are consumed before the
                    # next epoch's replay overwrites them (the epoch-end cat/
                    # .item() syncs below).
                    gval = self._graphed_val
                    gval.replay()
                    epoch_val_loss = epoch_val_loss + gval.loss_sum
                    for k, acc in gval.comp_sums.items():
                        if k not in val_components_accum:
                            val_components_accum[k] = torch.zeros(
                                (), device=self.device, dtype=torch.float32
                            )
                        val_components_accum[k] = val_components_accum[k] + acc
                    n_val_batches += gval.k
                    for k in self.target_names:
                        all_preds[k].append(gval.pred_bufs[k])
                        all_targets[k].append(gval.target_prefix[k])
                    val_batch_iter = gval.tail_batches()
                else:
                    val_batch_iter = val_loader
                for batch in val_batch_iter:
                    with self._autocast():
                        preds, y_batch = self._forward_batch(batch)
                        loss, components = self.criterion._compute_loss_components(preds, y_batch)

                    epoch_val_loss = epoch_val_loss + loss.detach().float()
                    for k, v in components.items():
                        if k not in val_components_accum:
                            val_components_accum[k] = torch.zeros(
                                (), device=self.device, dtype=torch.float32
                            )
                        val_components_accum[k] = val_components_accum[k] + v.detach().float()
                    n_val_batches += 1

                    for k in self.target_names:
                        # Defer device→host transfer to one ``torch.cat(...)``
                        # per target at end of epoch (see below). Detach to
                        # drop autograd refs; ``.float()`` upcasts FP16 preds
                        # so the eventual ``numpy()`` round-trip is FP32.
                        all_preds[k].append(preds[k].detach().float())
                        all_targets[k].append(y_batch[k].detach())

            # Single end-of-epoch sync (forces accumulator off-GPU). Guard
            # against ``n_val_batches == 0`` — same rationale as the train
            # NaN guard above.
            avg_val_loss = (epoch_val_loss / n_val_batches).item() if n_val_batches > 0 else 0.0
            history["val_loss"].append(avg_val_loss)

            if self.epoch_callback is not None:
                # Raises (e.g. optuna.TrialPruned) propagate up to whoever
                # called trainer.train() — that is the intended control flow
                # for tuner-driven pruning. Do NOT swallow.
                self.epoch_callback(epoch, avg_val_loss)

            # Per-target val losses — single host sync per target per epoch
            # (was per-batch via ``.item()`` inside ``MultiTargetLoss.forward``).
            for t in self.target_names:
                key = f"loss_{t}"
                if key in val_components_accum and n_val_batches > 0:
                    history[f"val_loss_{t}"].append(
                        (val_components_accum[key] / n_val_batches).item()
                    )
                else:
                    history[f"val_loss_{t}"].append(0.0)

            # Per-target MAE — single GPU→CPU transfer per target per epoch
            # (was per-batch). ``all_preds[k]`` / ``all_targets[k]`` are lists
            # of GPU tensors accumulated above; ``torch.cat`` stays on-device
            # and the trailing ``.cpu().numpy()`` is the only host transfer.
            # Same empty-batch guard as the loss accumulators above:
            # ``torch.cat([])`` raises, and a NaN feeds into the early-stop
            # comparison as ``inf < float('inf')`` = False, silently disabling
            # the early-stop reset and the best-checkpoint save.
            for k in self.target_names:
                if n_val_batches > 0:
                    y_pred_all = torch.cat(all_preds[k]).cpu().numpy()
                    y_true_all = torch.cat(all_targets[k]).cpu().numpy()
                    history[f"val_mae_{k}"].append(np.mean(np.abs(y_pred_all - y_true_all)))
                else:
                    history[f"val_mae_{k}"].append(float("inf"))

            # Per-epoch wall-clock is a host-side measurement; the
            # ``.item()`` on the loss accumulators and the ``.cpu().numpy()``
            # for MAE above are the only end-of-epoch syncs and they
            # implicitly bound the GPU work, so _epoch_sec captures the
            # correct boundary. Explicit torch.cuda.synchronize() here was
            # hanging on Batch g4dn instances (suspected interaction with
            # PR #267's ThreadPoolExecutor overlap and the nvidia-smi
            # sidecar's NVML polling); not needed for the investigation's
            # accuracy.
            _epoch_sec = time.perf_counter() - _epoch_t0
            _peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3 if _cuda else 0.0
            history["epoch_sec"].append(_epoch_sec)
            history["peak_mem_gb"].append(_peak_mem_gb)

            # --- LR Scheduler ---
            if not self.scheduler_per_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()
                # A3 LOAD-BEARING: capturable AdamW's LR is a baked device
                # tensor. A per-epoch scheduler.step() may rebind
                # param_group['lr'] to a NEW object → replays would read a STALE
                # LR. Write the fresh value INTO the baked tensor in place so the
                # next epoch's replays use the current LR (no-op when A3 is off).
                if self._graphed_opt is not None:
                    self._graphed_opt.refresh_lr_from_scheduler()

            # --- Early Stopping (loss-weighted MAE) ---
            val_mae_weighted = (
                sum(
                    loss_weights.get(t, 1.0) * history[f"val_mae_{t}"][-1]
                    for t in self.target_names
                )
                / weight_sum
            )
            if val_mae_weighted < self.best_val_metric:
                self.best_val_metric = val_mae_weighted
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if not _fixed_epochs and self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                    else:
                        print("  WARNING: no valid checkpoint saved (all epochs had NaN MAE)")
                    break

            # --- Logging ---
            if (epoch + 1) % self.log_every == 0:
                target_maes = " | ".join(
                    f"{t}: {history[f'val_mae_{t}'][-1]:.3f}" for t in self.target_names
                )
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {avg_val_loss:.4f} | "
                    f"MAE wtd: {val_mae_weighted:.3f} | "
                    f"epoch_sec={_epoch_sec:.2f} peak_mem_gb={_peak_mem_gb:.3f} | "
                    f"{target_maes}"
                )
        else:
            # Loop completed all n_epochs without early stopping. Without this,
            # the caller would get the last-epoch weights instead of the best
            # checkpoint, silently degrading model quality.
            if not _fixed_epochs and self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)

        self._close_scaler_trace()
        return history


class MultiHeadHistoryTrainer(MultiHeadTrainer):
    """Training loop for the attention-based model with game history input.

    Only overrides _forward_batch to handle the 4-tuple (static, history, mask, targets)
    batch format from the history DataLoader.
    """

    def _forward_batch(self, batch) -> tuple[dict, dict]:
        X_static, X_hist, hist_mask, y_batch = batch
        X_static = X_static.to(self.device, non_blocking=True)
        X_hist = X_hist.to(self.device, non_blocking=True)
        hist_mask = hist_mask.to(self.device, non_blocking=True)
        y_batch = {k: v.to(self.device, non_blocking=True) for k, v in y_batch.items()}
        preds = self.model(X_static, X_hist, hist_mask)
        return preds, y_batch

    def _graph_inputs(self, batch) -> tuple:
        X_static, X_hist, hist_mask, _ = batch
        return (
            X_static.to(self.device, non_blocking=True),
            X_hist.to(self.device, non_blocking=True),
            hist_mask.to(self.device, non_blocking=True),
        )


class MultiTargetHistoryWithOppDataset(Dataset):
    """Dataset for the two-branch attention model.

    Returns ``(X_static, player_history, hist_mask, opp_history, opp_mask, targets)``
    per sample. Both histories arrive pre-padded to fixed shapes from
    ``build_game_history_arrays`` / ``build_opp_defense_history_arrays`` so the
    default PyTorch collate stacks samples directly.
    """

    def __init__(
        self,
        X_static: np.ndarray,
        X_history: np.ndarray,
        history_mask: np.ndarray,
        X_opp_history: np.ndarray,
        opp_history_mask: np.ndarray,
        y_dict: dict,
    ):
        if len(X_opp_history) != len(X_static):
            raise ValueError(f"opp history len {len(X_opp_history)} != static len {len(X_static)}")
        self.X_static = torch.FloatTensor(X_static)
        self.X_history = torch.FloatTensor(X_history)
        self.history_mask = torch.from_numpy(np.asarray(history_mask, dtype=bool))
        self.X_opp_history = torch.FloatTensor(X_opp_history)
        self.opp_history_mask = torch.from_numpy(np.asarray(opp_history_mask, dtype=bool))
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X_static)

    def __getitem__(self, idx):
        return (
            self.X_static[idx],
            self.X_history[idx],
            self.history_mask[idx],
            self.X_opp_history[idx],
            self.opp_history_mask[idx],
            {k: v[idx] for k, v in self.targets.items()},
        )


def make_history_with_opp_dataloaders(
    X_train_static,
    X_train_history,
    train_history_mask,
    X_train_opp_history,
    train_opp_history_mask,
    y_train_dict,
    X_val_static,
    X_val_history,
    val_history_mask,
    X_val_opp_history,
    val_opp_history_mask,
    y_val_dict,
    batch_size=256,
    device=None,
):
    """Create DataLoaders for the two-branch attention model.

    All history tensors must be pre-padded to uniform shapes so the default
    PyTorch collate can stack samples without per-batch padding.

    On CUDA hosts, the six feature tensors plus targets are moved to the GPU
    once and iterated by :class:`_GPUResidentBatcher` — see
    :func:`make_dataloaders` for the rationale. CPU/MPS hosts retain the
    original DataLoader path bit-for-bit.
    """
    gpu_device = _gpu_resident_device(device)
    if gpu_device is not None:
        return _resident_loader_pair(
            [
                _to_gpu_float(X_train_static, gpu_device),
                _to_gpu_float(X_train_history, gpu_device),
                _to_gpu_mask(train_history_mask, gpu_device),
                _to_gpu_float(X_train_opp_history, gpu_device),
                _to_gpu_mask(train_opp_history_mask, gpu_device),
            ],
            _ydict_to_gpu(y_train_dict, gpu_device),
            [
                _to_gpu_float(X_val_static, gpu_device),
                _to_gpu_float(X_val_history, gpu_device),
                _to_gpu_mask(val_history_mask, gpu_device),
                _to_gpu_float(X_val_opp_history, gpu_device),
                _to_gpu_mask(val_opp_history_mask, gpu_device),
            ],
            _ydict_to_gpu(y_val_dict, gpu_device),
            batch_size,
        )

    train_ds = MultiTargetHistoryWithOppDataset(
        X_train_static,
        X_train_history,
        train_history_mask,
        X_train_opp_history,
        train_opp_history_mask,
        y_train_dict,
    )
    val_ds = MultiTargetHistoryWithOppDataset(
        X_val_static,
        X_val_history,
        val_history_mask,
        X_val_opp_history,
        val_opp_history_mask,
        y_val_dict,
    )
    return _dataloader_pair(train_ds, val_ds, batch_size)


class MultiHeadHistoryWithOppTrainer(MultiHeadTrainer):
    """Training loop for the attention model with both player and opp history.

    Overrides ``_forward_batch`` for the 6-tuple
    ``(static, hist, hist_mask, opp_hist, opp_mask, targets)`` produced by the
    default collate over :class:`MultiTargetHistoryWithOppDataset`.
    """

    def _forward_batch(self, batch) -> tuple[dict, dict]:
        X_static, X_hist, hist_mask, X_opp, opp_mask, y_batch = batch
        X_static = X_static.to(self.device, non_blocking=True)
        X_hist = X_hist.to(self.device, non_blocking=True)
        hist_mask = hist_mask.to(self.device, non_blocking=True)
        X_opp = X_opp.to(self.device, non_blocking=True)
        opp_mask = opp_mask.to(self.device, non_blocking=True)
        y_batch = {k: v.to(self.device, non_blocking=True) for k, v in y_batch.items()}
        preds = self.model(X_static, X_hist, hist_mask, X_opp, opp_mask)
        return preds, y_batch

    def _graph_inputs(self, batch) -> tuple:
        X_static, X_hist, hist_mask, X_opp, opp_mask, _ = batch
        return (
            X_static.to(self.device, non_blocking=True),
            X_hist.to(self.device, non_blocking=True),
            hist_mask.to(self.device, non_blocking=True),
            X_opp.to(self.device, non_blocking=True),
            opp_mask.to(self.device, non_blocking=True),
        )


class MultiTargetNestedKickDataset(Dataset):
    """Dataset returning static features + nested per-game kick history + targets.

    Unlike MultiTargetHistoryDataset the nested arrays are pre-padded to fixed
    shape `[G, K, kick_dim]` so the default collate works — no custom collation.

    Optional ``X_history`` adds a pre-padded `[G, game_dim]` per-game aggregate
    tensor; when present the dataset yields a 6-tuple instead of a 5-tuple and
    ``MultiHeadNestedHistoryTrainer`` dispatches on the length to pick the
    right model forward signature.
    """

    def __init__(
        self,
        X_static: np.ndarray,
        X_kicks: np.ndarray,
        outer_mask: np.ndarray,
        inner_mask: np.ndarray,
        y_dict: dict,
        X_history: np.ndarray | None = None,
    ):
        self.X_static = torch.FloatTensor(X_static)
        self.X_kicks = torch.FloatTensor(X_kicks)
        self.outer_mask = torch.from_numpy(np.asarray(outer_mask, dtype=bool))
        self.inner_mask = torch.from_numpy(np.asarray(inner_mask, dtype=bool))
        self.X_history = None if X_history is None else torch.FloatTensor(X_history)
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X_static)

    def __getitem__(self, idx):
        targets = {k: v[idx] for k, v in self.targets.items()}
        if self.X_history is None:
            return (
                self.X_static[idx],
                self.X_kicks[idx],
                self.outer_mask[idx],
                self.inner_mask[idx],
                targets,
            )
        return (
            self.X_static[idx],
            self.X_kicks[idx],
            self.outer_mask[idx],
            self.inner_mask[idx],
            self.X_history[idx],
            targets,
        )


def make_nested_kick_dataloaders(
    X_train_static,
    X_train_kicks,
    train_outer_mask,
    train_inner_mask,
    y_train_dict,
    X_val_static,
    X_val_kicks,
    val_outer_mask,
    val_inner_mask,
    y_val_dict,
    batch_size=256,
    X_train_history=None,
    X_val_history=None,
    device=None,
):
    """Build train/val DataLoaders for the nested-history attention model.

    Mirrors :class:`MultiTargetNestedKickDataset.__getitem__`'s 5/6-tuple
    dispatch: when ``X_*_history`` are absent the batcher yields a 5-tuple
    ``(X_static, X_kicks, outer_mask, inner_mask, y_dict)``; when present the
    per-game aggregate tensor is inserted between ``inner_mask`` and the
    target dict, producing a 6-tuple — :class:`MultiHeadNestedHistoryTrainer`
    branches on ``len(batch)`` to pick the right forward signature.

    On CUDA hosts the GPU-resident batcher replaces the DataLoader path; see
    :func:`make_dataloaders` for the rationale.
    """
    gpu_device = _gpu_resident_device(device)
    if gpu_device is not None:
        # Outer/inner masks are bool in ``MultiTargetNestedKickDataset.__init__``.
        train_feats = [
            _to_gpu_float(X_train_static, gpu_device),
            _to_gpu_float(X_train_kicks, gpu_device),
            _to_gpu_mask(train_outer_mask, gpu_device),
            _to_gpu_mask(train_inner_mask, gpu_device),
        ]
        val_feats = [
            _to_gpu_float(X_val_static, gpu_device),
            _to_gpu_float(X_val_kicks, gpu_device),
            _to_gpu_mask(val_outer_mask, gpu_device),
            _to_gpu_mask(val_inner_mask, gpu_device),
        ]
        if X_train_history is not None:
            train_feats.append(_to_gpu_float(X_train_history, gpu_device))
        if X_val_history is not None:
            val_feats.append(_to_gpu_float(X_val_history, gpu_device))
        return _resident_loader_pair(
            train_feats,
            _ydict_to_gpu(y_train_dict, gpu_device),
            val_feats,
            _ydict_to_gpu(y_val_dict, gpu_device),
            batch_size,
        )

    train_ds = MultiTargetNestedKickDataset(
        X_train_static,
        X_train_kicks,
        train_outer_mask,
        train_inner_mask,
        y_train_dict,
        X_history=X_train_history,
    )
    val_ds = MultiTargetNestedKickDataset(
        X_val_static,
        X_val_kicks,
        val_outer_mask,
        val_inner_mask,
        y_val_dict,
        X_history=X_val_history,
    )
    return _dataloader_pair(train_ds, val_ds, batch_size)


class MultiHeadNestedHistoryTrainer(MultiHeadTrainer):
    """Training loop for the nested-attention model.

    Dispatches on tuple length: the 5-tuple (static, kicks, outer_mask,
    inner_mask, targets) is the legacy K path (no per-game aggregates); the
    6-tuple inserts ``x_game_history`` between ``inner_mask`` and ``targets``
    and feeds it through to the model as ``x_game_history``.
    """

    def _maybe_graph_model(self, train_loader) -> None:
        """Nested K-style attention intentionally stays eager under FF_CUDA_GRAPH.

        The generic graph wrapper only supports positional tensor arguments.
        This trainer calls the model with the optional ``x_game_history`` kwarg,
        and Batch tune jobs enable FF_CUDA_GRAPH globally across all six
        positions. Treat nested history as an explicit no-op so K tuning runs
        correctly while the flat-history positions use graph capture.
        """
        return

    def _maybe_graph_full_step(self, train_loader) -> bool:
        """Full-step capture no-ops for the same kwarg/None-leaf reasons as
        ``_maybe_graph_model`` above — ``_GraphedTrainStep`` calls
        ``model(*feats)`` positionally and the 5-tuple legacy path carries a
        ``None`` history leaf. K stays eager under FF_CUDA_GRAPH_FULL too."""
        return False

    def _maybe_graph_full_opt(self, train_loader) -> bool:
        """Optimizer-tail capture (Lever A3) no-ops here too: A3 builds on the
        full-step graph this trainer never engages (above), so it can never fire.
        Explicit override is a defensive guard against a future base-class change
        accidentally enabling it for the nested-K kwarg/None-leaf path."""
        return False

    def _forward_batch(self, batch) -> tuple[dict, dict]:
        if len(batch) == 5:
            X_static, X_kicks, outer_mask, inner_mask, y_batch = batch
            X_history = None
        else:
            X_static, X_kicks, outer_mask, inner_mask, X_history, y_batch = batch
            X_history = X_history.to(self.device, non_blocking=True)
        X_static = X_static.to(self.device, non_blocking=True)
        X_kicks = X_kicks.to(self.device, non_blocking=True)
        outer_mask = outer_mask.to(self.device, non_blocking=True)
        inner_mask = inner_mask.to(self.device, non_blocking=True)
        y_batch = {k: v.to(self.device, non_blocking=True) for k, v in y_batch.items()}
        preds = self.model(X_static, X_kicks, outer_mask, inner_mask, x_game_history=X_history)
        return preds, y_batch


def plot_training_curves(history: dict, target_names: list[str], save_path: str) -> None:
    """Multi-panel figure for multi-head training."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Overall loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # Panel 2: Per-target val losses
    for t in target_names:
        key = f"val_loss_{t}"
        if key in history:
            axes[1].plot(history[key], label=t.replace("_", " ").title())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Per-Target Loss")
    axes[1].set_title("Per-Target Validation Loss")
    axes[1].legend()

    # Panel 3: Per-target MAE
    for t in target_names:
        key = f"val_mae_{t}"
        if key in history:
            axes[2].plot(history[key], label=t.replace("_", " ").title())
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE")
    axes[2].set_title("Per-Target Validation MAE")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
