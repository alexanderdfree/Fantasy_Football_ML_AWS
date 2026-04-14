"""WR-specific multi-head neural network for fantasy point decomposition."""

import torch
import torch.nn as nn
import numpy as np


class WRMultiHeadNet(nn.Module):
    """Multi-head neural network for WR fantasy point decomposition.

    Architecture:
        Input (~130 features)
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x N
            -> Head 1 (receiving_floor): Linear -> ReLU -> Linear -> squeeze
            -> Head 2 (rushing_floor):   Linear -> ReLU -> Linear -> squeeze
            -> Head 3 (td_points):       Linear -> ReLU -> Linear -> squeeze

    Total prediction = head1 + head2 + head3
    """

    def __init__(
        self,
        input_dim: int,
        backbone_layers: list = None,
        head_hidden: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        if backbone_layers is None:
            backbone_layers = [128, 96, 48]

        # === Shared Backbone ===
        backbone_blocks = []
        prev_dim = input_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]

        # === Output Heads ===
        self.receiving_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.rushing_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.td_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            Dictionary with keys:
                "receiving_floor": shape (batch_size,)
                "rushing_floor": shape (batch_size,)
                "td_points": shape (batch_size,)
                "total": shape (batch_size,) -- sum of the 3 heads
        """
        shared = self.backbone(x)

        receiving = self.receiving_head(shared).squeeze(-1)
        rushing = self.rushing_head(shared).squeeze(-1)
        td = self.td_head(shared).squeeze(-1)

        # Clamp to non-negative: these targets are physically >= 0
        if not self.training:
            receiving = torch.clamp(receiving, min=0)
            rushing = torch.clamp(rushing, min=0)
            td = torch.clamp(td, min=0)

        return {
            "receiving_floor": receiving,
            "rushing_floor": rushing,
            "td_points": td,
            "total": receiving + rushing + td,
        }

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        """Convenience method for inference from numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}
