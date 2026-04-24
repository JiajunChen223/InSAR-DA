from __future__ import annotations

import torch
import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss, resolve_ema_model
from insarda.methods.coral_utils import augment_sequence, safe_scalar_loss


class SSMTMethod(FormalMethod):
    name = "ss_mt"
    uses_ema = True
    selection_uses_ema = True
    uses_target_labeled = True
    uses_target_unlabeled = True

    def _consistency_loss(
        self,
        model: nn.Module,
        ema_teacher: nn.Module | None,
        target_unlabeled_batch,
    ) -> tuple[torch.Tensor, float, float]:
        teacher_model = resolve_ema_model(ema_teacher)
        if (
            teacher_model is None
            or target_unlabeled_batch is None
            or target_unlabeled_batch["x"].shape[0] == 0
        ):
            reference = next(model.parameters())
            return reference.new_zeros(()), 0.0, 0.0
        weak_x = augment_sequence(target_unlabeled_batch["x"], noise_std=0.004, drop_prob=0.01, scale_std=0.01)
        strong_x = augment_sequence(target_unlabeled_batch["x"], noise_std=0.012, drop_prob=0.05, scale_std=0.03)
        with torch.no_grad():
            teacher_prediction = teacher_model(weak_x)
        student_prediction = model(strong_x)
        prediction_var = teacher_prediction.detach().float().std(dim=1)
        confidence = torch.exp(-prediction_var)
        threshold = torch.quantile(
            confidence,
            q=max(0.0, min(1.0, 1.0 - float(self.method_cfg.confidence_quantile))),
        )
        keep_mask = (confidence >= threshold).float()
        confidence_peak = confidence.max().detach().clamp_min(float(threshold.detach().cpu().item()) + 1e-6)
        confidence_weight = torch.where(
            keep_mask > 0.0,
            0.5 + 0.5 * ((confidence - threshold) / (confidence_peak - threshold + 1e-6)).clamp(min=0.0, max=1.0),
            torch.zeros_like(confidence),
        )
        weight = confidence_weight.unsqueeze(1)
        horizon = max(int(student_prediction.shape[1]), 1)
        loss = (((student_prediction - teacher_prediction) ** 2) * weight).sum() / (
            weight.sum().clamp_min(1.0) * float(horizon)
        )
        return (
            safe_scalar_loss(loss, student_prediction),
            float(keep_mask.mean().detach().cpu().item()),
            float(confidence.mean().detach().cpu().item()),
        )

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ):
        del anchor_teacher
        source_prediction = model(source_batch["x"])
        source_loss = masked_mse_loss(source_prediction, source_batch["y"], source_batch["y_mask"])

        target_supervision_loss = source_loss.new_tensor(0.0)
        if target_labeled_batch is not None and target_labeled_batch["x"].shape[0] > 0:
            target_prediction = model(target_labeled_batch["x"])
            target_supervision_loss = masked_mse_loss(
                target_prediction,
                target_labeled_batch["y"],
                target_labeled_batch["y_mask"],
            )

        target_consistency_loss, keep_fraction, confidence_mean = self._consistency_loss(
            model,
            ema_teacher,
            target_unlabeled_batch,
        )
        total = (
            float(self.method_cfg.source_supervision_weight) * source_loss
            + float(self.method_cfg.target_supervision_weight) * target_supervision_loss
            + float(self.method_cfg.target_consistency_weight) * target_consistency_loss
        )
        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "target_consistency_loss": float(target_consistency_loss.detach().cpu().item()),
                "consistency_keep_fraction": keep_fraction,
                "consistency_confidence_mean": confidence_mean,
                "total_loss": float(total.detach().cpu().item()),
            },
        )
