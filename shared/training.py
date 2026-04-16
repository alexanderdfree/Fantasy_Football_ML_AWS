"""Generic training infrastructure: loss, dataset, dataloaders, and trainer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class MultiTargetLoss(nn.Module):
    """Combined Huber loss for a multi-head network.

    Loss = sum(weight[t] * Huber(pred[t], target[t]) for t in targets)
           + w_total * Huber(total_pred, total_actual)
           + td_gate_weight * BCE(gate_logit, td > 0)   [when gated TD is active]

    Uses Huber loss for robustness to outlier games.
    Per-target deltas allow different MSE-to-MAE thresholds.
    """

    def __init__(
        self,
        target_names: list[str],
        loss_weights: dict[str, float],
        huber_deltas: dict[str, float] = None,
        w_total: float = 0.5,
        td_gate_weight: float = 1.0,
    ):
        super().__init__()
        self.target_names = target_names
        self.loss_weights = {n: loss_weights.get(n, 1.0) for n in target_names}
        self.w_total = w_total
        self.td_gate_weight = td_gate_weight
        if huber_deltas is None:
            huber_deltas = {}
        self.huber_fns = nn.ModuleDict({
            name: nn.HuberLoss(delta=huber_deltas.get(name, 1.0))
            for name in target_names
        })
        self.huber_total = nn.HuberLoss(delta=huber_deltas.get("total", 1.0))

    def forward(self, preds: dict, targets: dict) -> tuple:
        per_target_losses = {}
        combined = torch.tensor(0.0, device=next(iter(preds.values())).device)
        for name in self.target_names:
            loss = self.huber_fns[name](preds[name], targets[name])
            per_target_losses[name] = loss
            combined = combined + self.loss_weights[name] * loss

        loss_total = self.huber_total(preds["total"], targets["total"])
        combined = combined + self.w_total * loss_total

        components = {f"loss_{name}": loss.item() for name, loss in per_target_losses.items()}
        components["loss_total_aux"] = loss_total.item()

        # Gated TD: add BCE supervision on the gate logit
        gate_key = "td_points_gate_logit"
        if gate_key in preds:
            gate_loss = F.binary_cross_entropy_with_logits(
                preds[gate_key], (targets["td_points"] > 0).float()
            )
            combined = combined + self.td_gate_weight * gate_loss
            components["loss_td_gate"] = gate_loss.item()

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
    """Dataset that returns static features + variable-length game history + targets."""

    def __init__(self, X_static: np.ndarray, X_history: list[np.ndarray], y_dict: dict):
        """
        Args:
            X_static: [n_samples, static_dim] static feature array
            X_history: list of n_samples arrays, each [seq_len_i, game_dim]
            y_dict: dict of target arrays
        """
        self.X_static = torch.FloatTensor(X_static)
        self.histories = [torch.FloatTensor(h) for h in X_history]
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X_static)

    def __getitem__(self, idx):
        return self.X_static[idx], self.histories[idx], {k: v[idx] for k, v in self.targets.items()}


def collate_with_history(batch):
    """Custom collate that pads variable-length game histories within each batch."""
    statics, histories, targets = zip(*batch)
    statics = torch.stack(statics)

    # Pad histories to the longest sequence in this batch
    game_dim = histories[0].size(-1) if histories[0].dim() > 0 and histories[0].size(0) > 0 else 0
    max_len = max(h.size(0) for h in histories) if histories else 0
    max_len = max(max_len, 1)  # at least 1 to avoid empty tensors

    if game_dim == 0:
        # Edge case: determine game_dim from any non-empty history
        for h in histories:
            if h.dim() > 0 and h.size(0) > 0:
                game_dim = h.size(-1)
                break

    padded = torch.zeros(len(histories), max_len, game_dim)
    masks = torch.zeros(len(histories), max_len, dtype=torch.bool)
    for i, h in enumerate(histories):
        seq_len = h.size(0) if h.dim() > 0 else 0
        if seq_len > 0:
            padded[i, :seq_len] = h
            masks[i, :seq_len] = True

    target_dict = {k: torch.stack([t[k] for t in targets]) for k in targets[0]}
    return statics, padded, masks, target_dict


def make_history_dataloaders(X_train_static, X_train_history, y_train_dict,
                             X_val_static, X_val_history, y_val_dict,
                             batch_size=256):
    """Create DataLoaders for attention model with game history."""
    train_ds = MultiTargetHistoryDataset(X_train_static, X_train_history, y_train_dict)
    val_ds = MultiTargetHistoryDataset(X_val_static, X_val_history, y_val_dict)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True,
                              collate_fn=collate_with_history)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False,
                            collate_fn=collate_with_history)
    return train_loader, val_loader


def make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    """Create DataLoaders for multi-target training."""
    train_ds = MultiTargetDataset(X_train, y_train_dict)
    val_ds = MultiTargetDataset(X_val, y_val_dict)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader


class MultiHeadTrainer:
    """Training loop for any multi-head position network.

    Subclass and override _forward_batch() to support different input formats
    (e.g., attention models with game history).
    """

    def __init__(self, model, optimizer, scheduler, criterion, device,
                 target_names, patience=15, scheduler_per_batch=False,
                 log_every=10):
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
        X_batch = X_batch.to(self.device)
        y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
        preds = self.model(X_batch)
        return preds, y_batch

    def train(self, train_loader, val_loader, n_epochs) -> dict:
        all_keys = self.target_names + ["total"]
        history = {k: [] for k in [
            "train_loss", "val_loss",
            *[f"val_loss_{t}" for t in self.target_names],
            "val_mae_total", *[f"val_mae_{t}" for t in self.target_names],
            "val_rmse_total",
        ]}

        for epoch in range(n_epochs):
            # --- Training pass ---
            self.model.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            for batch in train_loader:
                preds, y_batch = self._forward_batch(batch)

                self.optimizer.zero_grad()
                loss, _ = self.criterion(preds, y_batch)
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
            all_preds = {k: [] for k in all_keys}
            all_targets = {k: [] for k in all_keys}
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

                    for k in all_keys:
                        all_preds[k].append(preds[k].cpu().numpy())
                        all_targets[k].append(y_batch[k].cpu().numpy())

            avg_val_loss = epoch_val_loss / n_val_batches
            history["val_loss"].append(avg_val_loss)

            # Per-target val losses
            for t in self.target_names:
                history[f"val_loss_{t}"].append(
                    val_components_accum.get(f"loss_{t}", 0) / n_val_batches
                )

            # Compute MAE per target
            for k in all_keys:
                y_pred_all = np.concatenate(all_preds[k])
                y_true_all = np.concatenate(all_targets[k])
                mae = np.mean(np.abs(y_pred_all - y_true_all))
                history[f"val_mae_{k}"].append(mae)
                if k == "total":
                    history["val_rmse_total"].append(
                        np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
                    )

            # --- LR Scheduler ---
            if not self.scheduler_per_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # --- Early Stopping (on total MAE, not combined loss) ---
            val_mae_total = history["val_mae_total"][-1]
            if val_mae_total < self.best_val_metric:
                self.best_val_metric = val_mae_total
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

            # --- Logging ---
            if (epoch + 1) % self.log_every == 0:
                target_maes = " | ".join(
                    f"{t}: {history[f'val_mae_{t}'][-1]:.3f}"
                    for t in self.target_names
                )
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {avg_val_loss:.4f} | "
                    f"MAE total: {history['val_mae_total'][-1]:.3f} | "
                    f"{target_maes}"
                )

        return history


class MultiHeadHistoryTrainer(MultiHeadTrainer):
    """Training loop for the attention-based model with game history input.

    Only overrides _forward_batch to handle the 4-tuple (static, history, mask, targets)
    batch format from the history DataLoader.
    """

    def _forward_batch(self, batch) -> tuple[dict, dict]:
        X_static, X_hist, hist_mask, y_batch = batch
        X_static = X_static.to(self.device)
        X_hist = X_hist.to(self.device)
        hist_mask = hist_mask.to(self.device)
        y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
        preds = self.model(X_static, X_hist, hist_mask)
        return preds, y_batch


def plot_training_curves(history: dict, target_names: list[str], save_path: str) -> None:
    """Multi-panel figure for multi-head training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Overall loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Combined Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()

    # Panel 2: Per-target val losses
    for t in target_names:
        key = f"val_loss_{t}"
        if key in history:
            axes[0, 1].plot(history[key], label=t.replace("_", " ").title())
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Huber Loss")
    axes[0, 1].set_title("Per-Target Validation Loss")
    axes[0, 1].legend()

    # Panel 3: Per-target MAE
    for t in target_names:
        key = f"val_mae_{t}"
        if key in history:
            axes[1, 0].plot(history[key], label=t.replace("_", " ").title())
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].set_title("Per-Target Validation MAE")
    axes[1, 0].legend()

    # Panel 4: Total MAE and RMSE
    axes[1, 1].plot(history["val_mae_total"], label="Total MAE")
    axes[1, 1].plot(history["val_rmse_total"], label="Total RMSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Error")
    axes[1, 1].set_title("Total Fantasy Points Metrics")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
