import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
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
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
        }

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            history["train_loss"].append(epoch_loss / n_batches)

            # Validation
            self.model.eval()
            all_preds, all_true = [], []
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    preds = self.model(X_batch)
                    val_loss += self.criterion(preds, y_batch).item()
                    n_val += 1
                    all_preds.append(preds.cpu().numpy())
                    all_true.append(y_batch.cpu().numpy())

            avg_val_loss = val_loss / n_val
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_true)
            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(np.mean(np.abs(y_pred - y_true)))
            history["val_rmse"].append(np.sqrt(np.mean((y_pred - y_true) ** 2)))

            # LR scheduler
            self.scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Train: {history['train_loss'][-1]:.4f} | "
                    f"Val: {avg_val_loss:.4f} | "
                    f"MAE: {history['val_mae'][-1]:.3f}"
                )

        # Restore best weights. Early-stopping does this already and breaks,
        # so we only need to handle the "ran to completion" path where the
        # final epoch's weights may be worse than best_val_loss.
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def plot_training_curves(self, history: dict, save_path: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()

        ax2.plot(history["val_mae"], label="Val MAE")
        ax2.plot(history["val_rmse"], label="Val RMSE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Error")
        ax2.set_title("Validation Metrics")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create DataLoaders from pre-scaled numpy arrays."""
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    return train_loader, val_loader
