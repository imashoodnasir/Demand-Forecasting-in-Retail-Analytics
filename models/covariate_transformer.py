from __future__ import annotations
import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding, LearnablePositionalEncoding


class CovariateAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_cov_dim: int,
        input_length: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        learnable_pe: bool = False,
        use_covariates: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.future_cov_dim = future_cov_dim
        self.use_covariates = use_covariates
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = LearnablePositionalEncoding(d_model, max_len=input_length) if learnable_pe else PositionalEncoding(d_model, max_len=input_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        fusion_dim = d_model + horizon * future_cov_dim if future_cov_dim > 0 and use_covariates else d_model
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, horizon),
        )

    def forward(self, x_hist: torch.Tensor, x_future_cov: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_projection(x_hist)
        x = self.position(x)
        z = self.encoder(x)
        pooled = z[:, -1, :]
        if self.future_cov_dim > 0 and self.use_covariates and x_future_cov is not None and x_future_cov.numel() > 0:
            pooled = torch.cat([pooled, x_future_cov.reshape(x_future_cov.size(0), -1)], dim=1)
        return self.head(pooled)
