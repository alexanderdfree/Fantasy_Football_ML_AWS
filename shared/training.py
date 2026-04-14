import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class MultiTargetLoss(nn.Module):
    """Weighted Huber loss across multiple targets plus an auxiliary total loss."""

    def __init__(self, target_names: list[str], weights: dict, w_total: float = 0.5,
                 huber_deltas: dict = None):
        super().__init__()
        self.target_names = target_names
        self.weights = weights
        self.w_total = w_total
        # Per-target Huber deltas allow different MSE-to-MAE thresholds per target
        if huber_deltas is None:
            huber_deltas = {}
        self.huber_losses = {
            name: nn.HuberLoss(delta=huber_deltas.get(name, 1.0))
            for name in target_names
        }
        self.huber_total = nn.HuberLoss(delta=huber_deltas.get("total", 1.0))

    def forward(self, preds: dict, targets: dict) -> tuple:
        losses = {}
        combined = torch.tensor(0.0, device=next(iter(preds.values())).device)
        for name in self.target_names:
            loss = self.huber_losses[name](preds[name], targets[name])
            losses[f"loss_{name}"] = loss.item()
            combined = combined + self.weights[name] * loss

        loss_total = self.huber_total(preds["total"], targets["total"])
        losses["loss_total_aux"] = loss_total.item()
        combined = combined + self.w_total * loss_total
        losses["loss_combined"] = combined.item()

        return combined, losses


class MultiTargetDataset(Dataset):
    def __init__(self, X: np.ndarray, y_dict: dict):
        self.X = torch.FloatTensor(X)
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


def make_multi_target_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    train_ds = MultiTargetDataset(X_train, y_train_dict)
    val_ds = MultiTargetDataset(X_val, y_val_dict)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader


class MultiHeadTrainer:
    """Training loop for multi-head networks. Generic across positions."""

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
        history = {
            "train_loss": [], "val_loss": [],
            **{f"val_loss_{t}": [] for t in self.target_names},
            **{f"val_mae_{t}": [] for t in all_keys},
            "val_rmse_total": [],
        }

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            n_train = 0
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
                n_train += 1

            avg_train = epoch_train_loss / n_train
            history["train_loss"].append(avg_train)

            # Validation
            self.model.eval()
            all_preds = {k: [] for k in all_keys}
            all_targets = {k: [] for k in all_keys}
            epoch_val_loss = 0.0
            comp_accum = {}
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = {k: v.to(self.device) for k, v in y_batch.items()}
                    preds = self.model(X_batch)
                    loss, components = self.criterion(preds, y_batch)
                    epoch_val_loss += loss.item()
                    for k in components:
                        comp_accum[k] = comp_accum.get(k, 0) + components[k]
                    n_val += 1
                    for k in all_keys:
                        all_preds[k].append(preds[k].cpu().numpy())
                        all_targets[k].append(y_batch[k].cpu().numpy())

            avg_val = epoch_val_loss / n_val
            history["val_loss"].append(avg_val)

            for t in self.target_names:
                history[f"val_loss_{t}"].append(
                    comp_accum.get(f"loss_{t}", 0) / n_val
                )

            for k in all_keys:
                y_p = np.concatenate(all_preds[k])
                y_t = np.concatenate(all_targets[k])
                mae = np.mean(np.abs(y_p - y_t))
                history[f"val_mae_{k}"].append(mae)
                if k == "total":
                    history["val_rmse_total"].append(np.sqrt(np.mean((y_p - y_t) ** 2)))

            if not self.scheduler_per_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val)
                else:
                    self.scheduler.step()

            # Early stopping
            if avg_val < self.best_val_loss:
                self.best_val_loss = avg_val
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

            if (epoch + 1) % 10 == 0:
                target_maes = " | ".join(
                    f"{t}: {history[f'val_mae_{t}'][-1]:.3f}"
                    for t in self.target_names
                )
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                    f"MAE total: {history['val_mae_total'][-1]:.3f} | {target_maes}"
                )

        return history

    def plot_training_curves(self, history: dict, save_path: str, pos_label: str = "") -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(history["train_loss"], label="Train Loss")
        axes[0, 0].plot(history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Combined Loss")
        axes[0, 0].set_title(f"{pos_label} Training & Validation Loss")
        axes[0, 0].legend()

        for t in self.target_names:
            axes[0, 1].plot(history[f"val_loss_{t}"], label=t)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Huber Loss")
        axes[0, 1].set_title(f"{pos_label} Per-Target Validation Loss")
        axes[0, 1].legend()

        for t in self.target_names:
            axes[1, 0].plot(history[f"val_mae_{t}"], label=t)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].set_title(f"{pos_label} Per-Target Validation MAE")
        axes[1, 0].legend()

        axes[1, 1].plot(history["val_mae_total"], label="Total MAE")
        axes[1, 1].plot(history["val_rmse_total"], label="Total RMSE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Error")
        axes[1, 1].set_title(f"{pos_label} Total Fantasy Points Metrics")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
