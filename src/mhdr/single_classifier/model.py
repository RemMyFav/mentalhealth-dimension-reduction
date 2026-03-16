import torch
import torch.nn as nn


class LinearRankClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.net(x)