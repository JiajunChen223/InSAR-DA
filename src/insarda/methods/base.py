from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from insarda.config import MethodConfig


Batch = dict[str, torch.Tensor]


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * float(ctx.scale), None


class GradientReversal(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(tensor, self.scale)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(int(feature_dim // 2), 32)
        self.network = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class LossOutput:
    total: torch.Tensor
    metrics: dict[str, float]


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * weight).sum() / denom


def samplewise_masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    denom = weight.sum(dim=1).clamp_min(1.0)
    return ((((prediction - target) ** 2) * weight).sum(dim=1) / denom).float()


class FormalMethod(nn.Module):
    name = "formal_method"
    uses_source = True
    freeze_model = False
    uses_ema = False
    ema_tracks_method = False
    uses_anchor = False
    uses_target_labeled = False
    uses_target_unlabeled = False
    uses_source_pretraining = False
    post_pretrain_uses_source = False
    post_pretrain_uses_target_labeled = True
    post_pretrain_uses_target_unlabeled = False
    selection_loader_name = "target_val"

    def __init__(self, method_cfg: MethodConfig) -> None:
        super().__init__()
        self.method_cfg = method_cfg

    def compute_loss(
        self,
        model: nn.Module,
        source_batch: Batch,
        target_labeled_batch: Batch | None,
        target_unlabeled_batch: Batch | None,
        ema_teacher: nn.Module | None,
        anchor_teacher: nn.Module | None,
    ) -> LossOutput:
        raise NotImplementedError

    def prepare_for_evaluation(
        self,
        model: nn.Module,
        *,
        source_loader=None,
        device: torch.device | None = None,
        training_cfg=None,
    ) -> None:
        del model, source_loader, device, training_cfg

    def on_epoch_end(self, *, epoch: int, epoch_record: dict[str, Any]) -> dict[str, Any]:
        del epoch, epoch_record
        return {}

    def predict_batch(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch["x"])

    @staticmethod
    def _domain_loss(
        discriminator: DomainDiscriminator,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if source_features.numel() == 0 or target_features.numel() == 0:
            return source_features.new_tensor(0.0)
        grl = GradientReversal(scale=scale)
        combined = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat(
            [
                torch.zeros(source_features.shape[0], device=source_features.device),
                torch.ones(target_features.shape[0], device=target_features.device),
            ],
            dim=0,
        )
        logits = discriminator(grl(combined))
        return F.binary_cross_entropy_with_logits(logits, labels)


def resolve_ema_model(ema_teacher: Any) -> nn.Module | None:
    if ema_teacher is None:
        return None
    return getattr(ema_teacher, "teacher", ema_teacher)


def resolve_ema_method(ema_teacher: Any) -> nn.Module | None:
    if ema_teacher is None:
        return None
    return getattr(ema_teacher, "method_teacher", None)
