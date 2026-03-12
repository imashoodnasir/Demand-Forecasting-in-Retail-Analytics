from __future__ import annotations
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x_hist: torch.Tensor, x_future_cov: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x_hist)
        return self.head(out[:, -1, :])


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, input_length: int, horizon: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * input_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x_hist: torch.Tensor, x_future_cov: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x_hist.reshape(x_hist.size(0), -1))
