"""Generic training infrastructure: loss, dataset, dataloaders, and trainer."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SUPPORTED_HEAD_LOSSES = ("huber", "poisson_nll", "hurdle_negbin", "hurdle_poisson")

# DataLoader worker-process count. The default auto-detects: 2 workers on
# CUDA so the GPU sees a prefetched batch N+1 while computing batch N, 0
# workers on CPU/MPS where prefetching gives no benefit and ``spawn``-context
# worker startup adds hundreds of ms per loader on macOS dev boxes.
# ``NN_DATALOADER_NUM_WORKERS`` overrides the default in either direction
# (e.g. set to 4 on a host with more vCPU, or 0 to debug a fork issue).
_NUM_WORKERS = int(
    os.environ.get(
        "NN_DATALOADER_NUM_WORKERS",
        "2" if torch.cuda.is_available() else "0",
    )
)
_PERSISTENT_WORKERS = _NUM_WORKERS > 0


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


class MultiTargetLoss(nn.Module):
    """Per-head dispatchable loss for a multi-head network.

    Each target is assigned a loss family via ``head_losses[name]``; supported
    values are in ``SUPPORTED_HEAD_LOSSES``:
      - ``"huber"`` — standard Huber loss with per-target delta.
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
        self.loss_fns = nn.ModuleDict(
            {
                name: (
                    nn.PoissonNLLLoss(log_input=False, full=False)
                    if lt == "poisson_nll"
                    else nn.HuberLoss(delta=huber_deltas.get(name, 1.0))
                )
                for name, lt in self.head_losses.items()
                if lt in ("huber", "poisson_nll")
            }
        )

    def forward(self, preds: dict, targets: dict) -> tuple:
        per_target_losses = {}
        combined = torch.tensor(0.0, device=next(iter(preds.values())).device)
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

        components = {f"loss_{name}": loss.item() for name, loss in per_target_losses.items()}

        for gated_name in self.gated_targets:
            gate_key = f"{gated_name}_gate_logit"
            if gate_key in preds:
                gate_loss = F.binary_cross_entropy_with_logits(
                    preds[gate_key], (targets[gated_name] > 0).float()
                )
                combined = combined + self.gate_weight * gate_loss
                components[f"loss_gate_{gated_name}"] = gate_loss.item()

        components["loss_combined"] = combined.item()
        return combined, components


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
):
    """Create DataLoaders for attention model with game history.

    All history tensors must be pre-padded to a uniform ``[n, max_seq_len, game_dim]``
    shape so the default PyTorch collate can stack samples without per-batch padding.

    ``pin_memory=True`` is a no-op under CPU-only runs; on CUDA it allocates
    page-locked host tensors so the subsequent ``.to(device, non_blocking=True)``
    in the trainer can overlap the H2D copy with compute.
    """
    train_ds = MultiTargetHistoryDataset(
        X_train_static, X_train_history, train_history_mask, y_train_dict
    )
    val_ds = MultiTargetHistoryDataset(X_val_static, X_val_history, val_history_mask, y_val_dict)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    """Create DataLoaders for multi-target training.

    ``pin_memory=True`` is a no-op under CPU-only runs; on CUDA it allocates
    page-locked host tensors so the subsequent ``.to(device, non_blocking=True)``
    in the trainer can overlap the H2D copy with compute.
    """
    train_ds = MultiTargetDataset(X_train, y_train_dict)
    val_ds = MultiTargetDataset(X_val, y_val_dict)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


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
        self.best_val_metric = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

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

    def train(self, train_loader, val_loader, n_epochs) -> dict:
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

        for epoch in range(n_epochs):
            if _cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            _epoch_t0 = time.perf_counter()
            # --- Training pass ---
            self.model.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            for batch in train_loader:
                preds, y_batch = self._forward_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self.criterion(preds, y_batch)
                # Attention entropy regulariser: additive term that models can
                # optionally expose via ``attention_entropy_loss``. Returns
                # ``None`` when the feature is off so the hot path is a single
                # attribute check.
                entropy_fn = getattr(self.model, "attention_entropy_loss", None)
                if entropy_fn is not None:
                    entropy_term = entropy_fn()
                    if entropy_term is not None:
                        loss = loss + entropy_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler_per_batch:
                    self.scheduler.step()

                epoch_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = epoch_train_loss / n_train_batches
            history["train_loss"].append(avg_train_loss)

            # --- Validation pass ---
            self.model.eval()
            all_preds = {k: [] for k in self.target_names}
            all_targets = {k: [] for k in self.target_names}
            epoch_val_loss = 0.0
            val_components_accum = {}
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    preds, y_batch = self._forward_batch(batch)
                    loss, components = self.criterion(preds, y_batch)

                    epoch_val_loss += loss.item()
                    for k in components:
                        val_components_accum[k] = val_components_accum.get(k, 0) + components[k]
                    n_val_batches += 1

                    for k in self.target_names:
                        all_preds[k].append(preds[k].cpu().numpy())
                        all_targets[k].append(y_batch[k].cpu().numpy())

            avg_val_loss = epoch_val_loss / n_val_batches
            history["val_loss"].append(avg_val_loss)

            # Per-target val losses
            for t in self.target_names:
                history[f"val_loss_{t}"].append(
                    val_components_accum.get(f"loss_{t}", 0) / n_val_batches
                )

            # Per-target MAE
            for k in self.target_names:
                y_pred_all = np.concatenate(all_preds[k])
                y_true_all = np.concatenate(all_targets[k])
                history[f"val_mae_{k}"].append(np.mean(np.abs(y_pred_all - y_true_all)))

            if _cuda:
                torch.cuda.synchronize(self.device)
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
                if self.epochs_without_improvement >= self.patience:
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
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)

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
):
    """Create DataLoaders for the two-branch attention model.

    All history tensors must be pre-padded to uniform shapes so the default
    PyTorch collate can stack samples without per-batch padding.
    """
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


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
):
    """Build train/val DataLoaders for the nested-history attention model."""
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


class MultiHeadNestedHistoryTrainer(MultiHeadTrainer):
    """Training loop for the nested-attention model.

    Dispatches on tuple length: the 5-tuple (static, kicks, outer_mask,
    inner_mask, targets) is the legacy K path (no per-game aggregates); the
    6-tuple inserts ``x_game_history`` between ``inner_mask`` and ``targets``
    and feeds it through to the model as ``x_game_history``.
    """

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
