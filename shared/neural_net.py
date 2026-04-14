import torch
import torch.nn as nn
import numpy as np


class MultiHeadNet(nn.Module):
    """Generic multi-head neural network for fantasy point decomposition.

    Shared backbone feeds into N independent heads, one per target.
    """

    def __init__(
        self,
        input_dim: int,
        target_names: list[str],
        backbone_layers: list = None,
        head_hidden: int = 32,
        dropout: float = 0.3,
        head_hidden_overrides: dict = None,
    ):
        super().__init__()
        if backbone_layers is None:
            backbone_layers = [128, 64]
        if head_hidden_overrides is None:
            head_hidden_overrides = {}

        self.target_names = target_names

        # Shared backbone
        blocks = []
        prev_dim = input_dim
        for hidden_dim in backbone_layers:
            blocks.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)

        backbone_out = backbone_layers[-1]

        # Per-target heads (with optional per-head hidden size overrides)
        self.heads = nn.ModuleDict()
        for name in target_names:
            h = head_hidden_overrides.get(name, head_hidden)
            self.heads[name] = nn.Sequential(
                nn.Linear(backbone_out, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.backbone(x)
        preds = {}
        for name in self.target_names:
            val = self.heads[name](shared).squeeze(-1)
            if not self.training:
                val = torch.clamp(val, min=0)
            preds[name] = val
        preds["total"] = sum(preds[name] for name in self.target_names)
        return preds

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}
