from __future__ import annotations

import torch
import torch.nn as nn

from insarda.methods.base import DomainDiscriminator, FormalMethod, LossOutput, masked_mse_loss
from insarda.methods.coral_utils import safe_scalar_loss, stable_float


class SSDANNMethod(FormalMethod):
    name = "ss_dann"
    uses_target_labeled = True
    uses_target_unlabeled = True

    def __init__(self, feature_dim: int, method_cfg, **kwargs) -> None:
        del kwargs
        super().__init__(method_cfg)
        self.feature_norm = nn.LayerNorm(int(feature_dim))
        self.domain_discriminator = DomainDiscriminator(int(feature_dim))

    def _normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        safe_features = torch.nan_to_num(stable_float(features), nan=0.0, posinf=1e3, neginf=-1e3)
        normalized = self.feature_norm(safe_features)
        return normalized.to(device=features.device, dtype=features.dtype)

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ):
        del ema_teacher, anchor_teacher
        source_prediction, source_features = model(source_batch["x"], return_features=True)
        source_loss = masked_mse_loss(source_prediction, source_batch["y"], source_batch["y_mask"])
        source_loss = safe_scalar_loss(source_loss, source_prediction)

        target_supervision_loss = source_loss.new_tensor(0.0)
        target_feature_splits: list[torch.Tensor] = []
        if target_labeled_batch is not None and target_labeled_batch["x"].shape[0] > 0:
            target_prediction_l, target_features_l = model(target_labeled_batch["x"], return_features=True)
            target_supervision_loss = masked_mse_loss(
                target_prediction_l,
                target_labeled_batch["y"],
                target_labeled_batch["y_mask"],
            )
            target_feature_splits.append(target_features_l)
        if target_unlabeled_batch is not None and target_unlabeled_batch["x"].shape[0] > 0:
            target_features_u = model.encode(target_unlabeled_batch["x"])
            target_feature_splits.append(target_features_u)
        target_supervision_loss = safe_scalar_loss(target_supervision_loss, source_prediction)

        domain_loss = source_loss.new_tensor(0.0)
        domain_scale = 0.0
        if target_feature_splits:
            normalized_source = self._normalized_features(source_features)
            domain_scale = 1.5
            domain_terms = [
                safe_scalar_loss(
                    self._domain_loss(
                        discriminator=self.domain_discriminator,
                        source_features=normalized_source,
                        target_features=self._normalized_features(target_features),
                        scale=domain_scale,
                    ),
                    source_prediction,
                )
                for target_features in target_feature_splits
            ]
            domain_loss = safe_scalar_loss(
                torch.stack(domain_terms).mean(),
                source_prediction,
            )

        total = (
            float(self.method_cfg.source_supervision_weight) * source_loss
            + float(self.method_cfg.target_supervision_weight) * target_supervision_loss
            + float(self.method_cfg.domain_weight) * domain_loss
        )
        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "domain_loss": float(domain_loss.detach().cpu().item()),
                "domain_scale": float(domain_scale),
                "target_domain_splits": float(len(target_feature_splits)),
                "total_loss": float(total.detach().cpu().item()),
            },
        )
