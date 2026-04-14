import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class MultiTargetLoss(nn.Module):
    """Combined loss for multi-head RB network.

    Loss = w1 * MSE(rushing_floor) + w2 * MSE(receiving_floor) + w3 * MSE(td_points)
           + w_total * MSE(total_pred, total_actual)
    """

    def __init__(
        self,
        w_rushing: float = 1.0,
        w_receiving: float = 1.0,
        w_td: float = 1.0,
        w_total: float = 0.5,
    ):
        super().__init__()
        self.w_rushing = w_rushing
        self.w_receiving = w_receiving
        self.w_td = w_td
        self.w_total = w_total
        self.mse = nn.MSELoss()

    def forward(self, preds: dict, targets: dict) -> tuple:
        loss_rushing = self.mse(preds["rushing_floor"], targets["rushing_floor"])
        loss_receiving = self.mse(preds["receiving_floor"], targets["receiving_floor"])
        loss_td = self.mse(preds["td_points"], targets["td_points"])
        loss_total = self.mse(preds["total"], targets["total"])

        combined = (
            self.w_rushing * loss_rushing
            + self.w_receiving * loss_receiving
            + self.w_td * loss_td
            + self.w_total * loss_total
        )

        components = {
            "loss_rushing": loss_rushing.item(),
            "loss_receiving": loss_receiving.item(),
            "loss_td": loss_td.item(),
            "loss_total_aux": loss_total.item(),
            "loss_combined": combined.item(),
        }

        return combined, components


class RBMultiTargetDataset(Dataset):
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


def make_rb_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    """Create DataLoaders for multi-target RB training."""
    train_ds = RBMultiTargetDataset(X_train, y_train_dict)
    val_ds = RBMultiTargetDataset(X_val, y_val_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader


class RBMultiHeadTrainer:
    """Training loop for multi-head RB network."""

    def __init__(self, model, optimizer, scheduler, criterion, device, patience=15):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def train(self, train_loader, val_loader, n_epochs) -> dict:
        history = {k: [] for k in [
            "train_loss", "val_loss",
            "val_loss_rushing", "val_loss_receiving", "val_loss_td",
            "val_mae_total", "val_mae_rushing", "val_mae_receiving", "val_mae_td",
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
                self.optimizer.step()

                epoch_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = epoch_train_loss / n_train_batches
            history["train_loss"].append(avg_train_loss)

            # --- Validation pass ---
            self.model.eval()
            all_preds = {k: [] for k in ["rushing_floor", "receiving_floor", "td_points", "total"]}
            all_targets = {k: [] for k in ["rushing_floor", "receiving_floor", "td_points", "total"]}
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

                    for k in all_preds:
                        all_preds[k].append(preds[k].cpu().numpy())
                        all_targets[k].append(y_batch[k].cpu().numpy())

            avg_val_loss = epoch_val_loss / n_val_batches
            history["val_loss"].append(avg_val_loss)

            # Per-target val losses
            history["val_loss_rushing"].append(
                val_components_accum.get("loss_rushing", 0) / n_val_batches
            )
            history["val_loss_receiving"].append(
                val_components_accum.get("loss_receiving", 0) / n_val_batches
            )
            history["val_loss_td"].append(
                val_components_accum.get("loss_td", 0) / n_val_batches
            )

            # Compute MAE per target
            for k in ["rushing_floor", "receiving_floor", "td_points", "total"]:
                y_pred_all = np.concatenate(all_preds[k])
                y_true_all = np.concatenate(all_targets[k])
                mae = np.mean(np.abs(y_pred_all - y_true_all))
                if k == "rushing_floor":
                    history["val_mae_rushing"].append(mae)
                elif k == "receiving_floor":
                    history["val_mae_receiving"].append(mae)
                elif k == "td_points":
                    history["val_mae_td"].append(mae)
                elif k == "total":
                    history["val_mae_total"].append(mae)
                    history["val_rmse_total"].append(
                        np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
                    )

            # --- LR Scheduler ---
            self.scheduler.step(avg_val_loss)

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
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {avg_val_loss:.4f} | "
                    f"MAE total: {history['val_mae_total'][-1]:.3f} | "
                    f"rush: {history['val_mae_rushing'][-1]:.3f} | "
                    f"recv: {history['val_mae_receiving'][-1]:.3f} | "
                    f"td: {history['val_mae_td'][-1]:.3f}"
                )

        return history

    def plot_training_curves(self, history: dict, save_path: str) -> None:
        """Multi-panel figure for RB multi-head training."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Overall loss
        axes[0, 0].plot(history["train_loss"], label="Train Loss")
        axes[0, 0].plot(history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Combined Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()

        # Panel 2: Per-target val losses
        axes[0, 1].plot(history["val_loss_rushing"], label="Rushing Floor")
        axes[0, 1].plot(history["val_loss_receiving"], label="Receiving Floor")
        axes[0, 1].plot(history["val_loss_td"], label="TD Points")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE Loss")
        axes[0, 1].set_title("Per-Target Validation Loss")
        axes[0, 1].legend()

        # Panel 3: Per-target MAE
        axes[1, 0].plot(history["val_mae_rushing"], label="Rushing Floor")
        axes[1, 0].plot(history["val_mae_receiving"], label="Receiving Floor")
        axes[1, 0].plot(history["val_mae_td"], label="TD Points")
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
