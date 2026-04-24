from __future__ import annotations

import torch.nn as nn

from insarda.methods.base import FormalMethod, LossOutput, masked_mse_loss


class STJointMethod(FormalMethod):
    name = "st_joint"
    uses_target_labeled = True

    def compute_loss(
        self,
        model: nn.Module,
        source_batch,
        target_labeled_batch,
        target_unlabeled_batch,
        ema_teacher,
        anchor_teacher,
    ):
        del target_unlabeled_batch, ema_teacher, anchor_teacher
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

        total = (
            float(self.method_cfg.source_supervision_weight) * source_loss
            + float(self.method_cfg.target_supervision_weight) * target_supervision_loss
        )
        return LossOutput(
            total=total,
            metrics={
                "source_loss": float(source_loss.detach().cpu().item()),
                "target_supervision_loss": float(target_supervision_loss.detach().cpu().item()),
                "total_loss": float(total.detach().cpu().item()),
            },
        )
