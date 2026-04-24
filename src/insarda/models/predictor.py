from __future__ import annotations

import torch
import torch.nn as nn

from insarda.config import ModelConfig
from insarda.models.transformer import TransformerEncoder


class ForecastModel(nn.Module):
    def __init__(self, input_dim: int, horizon: int, model_cfg: ModelConfig, backbone: str = "transformer") -> None:
        super().__init__()
        self.backbone = str(backbone).strip().lower()
        if self.backbone != "transformer":
            raise ValueError("Only the formal transformer backbone is supported.")
        self.encoder = TransformerEncoder(
            input_dim=int(input_dim),
            d_model=int(model_cfg.transformer_d_model),
            nhead=int(model_cfg.transformer_nhead),
            num_layers=int(model_cfg.transformer_num_layers),
            dim_feedforward=int(model_cfg.transformer_ff_dim),
            dropout=float(model_cfg.dropout),
        )
        self.head = nn.Linear(int(self.encoder.out_dim), int(horizon))

    def encode(self, x: torch.Tensor, *, return_sequence: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_sequence:
            return self.encoder(x)
        return self.encoder(x, return_sequence=True)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _, token_features = self.encode(x, return_sequence=True)
        return token_features

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if return_tokens:
            features, token_features = self.encode(x, return_sequence=True)
        else:
            features = self.encode(x)
        prediction = self.predict_from_features(features)
        if return_features and return_tokens:
            return prediction, features, token_features
        if return_features:
            return prediction, features
        if return_tokens:
            return prediction, token_features
        return prediction
