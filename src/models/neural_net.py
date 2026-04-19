import torch
import torch.nn as nn


class FantasyPointsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
