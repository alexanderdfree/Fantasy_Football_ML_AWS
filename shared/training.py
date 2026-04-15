"""Generic training infrastructure: loss, dataset, dataloaders, and trainer."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class MultiTargetLoss(nn.Module):
    """Combined Huber loss for a multi-head network.

    Loss = sum(weight[t] * Huber(pred[t], target[t]) for t in targets)
           + w_total * Huber(total_pred, total_actual)

    Uses Huber loss for robustness to outlier games.
    Per-target deltas allow different MSE-to-MAE thresholds.
    """

    def __init__(
        self,
        target_names: list[str],
        loss_weights: dict[str, float],
        huber_deltas: dict[str, float] = None,
        w_total: float = 0.5,
    ):
        super().__init__()
        self.target_names = target_names
        self.loss_weights = loss_weights
        self.w_total = w_total
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
            combined = combined + self.loss_weights.get(name, 1.0) * loss

        loss_total = self.huber_total(preds["total"], targets["total"])
        combined = combined + self.w_total * loss_total

        components = {f"loss_{name}": loss.item() for name, loss in per_target_losses.items()}
        components["loss_total_aux"] = loss_total.item()
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
    """Training loop for any multi-head position network."""

    def __init__(self, model, optimizer, scheduler, criterion, device,
                 target_names, patience=15, scheduler_per_batch=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.target_names = target_names
        self.patience = patience
        self.scheduler_per_batch = scheduler_per_batch
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

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

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = {k: v.to(self.device) for k, v in y_batch.items()}

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
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
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = {k: v.to(self.device) for k, v in y_batch.items()}

                    preds = self.model(X_batch)
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

            # --- Early Stopping ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
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
            if (epoch + 1) % 10 == 0:
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
