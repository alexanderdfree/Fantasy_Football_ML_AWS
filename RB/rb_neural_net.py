import torch
import torch.nn as nn
import numpy as np


class RBMultiHeadNet(nn.Module):
    """Multi-head neural network for RB fantasy point decomposition.

    Architecture:
        Input (~122 features)
            -> Shared backbone [Linear -> BN -> ReLU -> Dropout] x2
            -> Head 1 (rushing_floor):  Linear -> ReLU -> Linear -> squeeze
            -> Head 2 (receiving_floor): Linear -> ReLU -> Linear -> squeeze
            -> Head 3 (td_points):      Linear -> ReLU -> Linear -> squeeze

    Total prediction = head1 + head2 + head3 + fumble_adjustment
    """

    def __init__(
        self,
        input_dim: int,
        backbone_layers: list = None,
        head_hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone_layers is None:
            backbone_layers = [128, 64]

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
        self.rushing_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.receiving_head = nn.Sequential(
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
                "rushing_floor": shape (batch_size,)
                "receiving_floor": shape (batch_size,)
                "td_points": shape (batch_size,)
                "total": shape (batch_size,) -- sum of the 3 heads
        """
        shared = self.backbone(x)

        rushing = self.rushing_head(shared).squeeze(-1)
        receiving = self.receiving_head(shared).squeeze(-1)
        td = self.td_head(shared).squeeze(-1)

        return {
            "rushing_floor": rushing,
            "receiving_floor": receiving,
            "td_points": td,
            "total": rushing + receiving + td,
        }

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        """Convenience method for inference from numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}
