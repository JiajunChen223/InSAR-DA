from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 224,
        dropout: float = 0.1,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, int(max_length), int(d_model)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.network = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(int(num_layers), 1),
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.out_dim = int(d_model)

    def forward(self, x: torch.Tensor, *, return_sequence: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds transformer max_length={self.position_embedding.shape[1]}.")
        embedded = self.input_proj(x) + self.position_embedding[:, : x.shape[1]]
        token_features = self.network(self.dropout(embedded))
        pooled_features = token_features[:, -1]
        if return_sequence:
            return pooled_features, token_features
        return pooled_features
