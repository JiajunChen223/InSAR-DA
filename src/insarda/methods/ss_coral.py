from __future__ import annotations

import torch
import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss
from insarda.methods.coral_utils import safe_scalar_loss, stable_float, weighted_coral_loss


def plain_coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.numel() == 0 or target_features.numel() == 0:
        reference = source_features if source_features.numel() > 0 else target_features
        return reference.new_zeros(())
    source = torch.nan_to_num(stable_float(source_features), nan=0.0, posinf=1e3, neginf=-1e3)
    target = torch.nan_to_num(stable_float(target_features), nan=0.0, posinf=1e3, neginf=-1e3)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_norm = max(int(source.shape[0]) - 1, 1)
    target_norm = max(int(target.shape[0]) - 1, 1)
    source_cov = source_centered.transpose(0, 1) @ source_centered / float(source_norm)
    target_cov = target_centered.transpose(0, 1) @ target_centered / float(target_norm)
    feature_dim = max(int(source.shape[-1]), 1)
    return ((source_cov - target_cov) ** 2).sum() / float(4 * feature_dim * feature_dim)


class SSCORALMethod(FormalMethod):
    name = "ss_coral"
    uses_target_labeled = True
    uses_target_unlabeled = True

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

        target_supervision_loss = source_loss.new_zeros(())
        target_feature_parts: list[torch.Tensor] = []
        if target_labeled_batch is not None and target_labeled_batch["x"].shape[0] > 0:
            target_prediction_l, target_features_l = model(target_labeled_batch["x"], return_features=True)
            target_supervision_loss = masked_mse_loss(
                target_prediction_l,
                target_labeled_batch["y"],
                target_labeled_batch["y_mask"],
            )
            target_feature_parts.append(target_features_l)
        if target_unlabeled_batch is not None and target_unlabeled_batch["x"].shape[0] > 0:
            target_feature_parts.append(model.encode(target_unlabeled_batch["x"]))

        coral_loss = source_loss.new_zeros(())
        if target_feature_parts:
            target_alignment_features = (
                target_feature_parts[0]
                if len(target_feature_parts) == 1
                else torch.cat(target_feature_parts, dim=0)
            )
            coral_loss = safe_scalar_loss(
                weighted_coral_loss(
                    stable_float(source_features),
                    stable_float(target_alignment_features),
                ),
                source_loss,
            )

        total = (
            float(self.method_cfg.source_supervision_weight) * source_loss
            + float(self.method_cfg.target_supervision_weight) * target_supervision_loss
            + float(self.method_cfg.domain_weight) * coral_loss
        )
        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "coral_loss": float(coral_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            },
        )


__all__ = ["SSCORALMethod", "plain_coral_loss"]
